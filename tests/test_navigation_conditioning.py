"""Navigation geometry, old checkpoints, and conditioning of the real planner."""

import copy
import io
import pickle
import sys

import numpy as np
import pytest
import torch

from lead.adapt.adapt_decoder import AdaptDecoder
from lead.adapt.navigation_conditioning import NavigationEncoder, NavigationModulation
from lead.training.config_training import TrainingConfig


class SmallConfig(TrainingConfig):
    """Real ADAPT modules/codebook, with small CPU dimensions and no CLI parsing."""

    carla_root = "data/carla_leaderboard2"
    use_mixed_precision_training = False
    use_adapt_decoder = True
    use_radars = radar_detection = use_radar_detection = False
    kinematic_embed_dim = transfuser_token_dim = 16
    decoder_ffn_dim = 32
    decoder_num_layers = decoder_num_heads = 2
    decoder_dropout = 0.0
    kinematic_vocab_size = 8
    num_history_poses = 3
    num_way_points_prediction = num_route_points_prediction = 4
    max_sequence_length = 16

    def __init__(self):
        pass

    @property
    def device(self):
        return torch.device("cpu")


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def config(tmp_path):
    vocabulary = tmp_path / "vocabulary.pkl"
    centroids = np.zeros((8, 3), dtype=np.float32)
    centroids[:, 0] = np.arange(8) * 0.5
    with vocabulary.open("wb") as stream:
        pickle.dump({"centroids": centroids, "info": {}}, stream)
    value = SmallConfig()
    value.kdisks_vocab_path = str(vocabulary)
    return value


def navigation(batch=2):
    return {
        "target_point_previous": torch.tensor([[-3.0, 0.0]]).repeat(batch, 1),
        "target_point": torch.tensor([[0.0, 0.0]]).repeat(batch, 1),
        "target_point_next": torch.tensor([[3.0, 4.0]]).repeat(batch, 1),
        "command": torch.tensor([[1.0, 0, 0, 0, 0, 0]]).repeat(batch, 1),
        "next_command": torch.tensor([[0.0, 0, 0, 1, 0, 0]]).repeat(batch, 1),
    }


def decoder(config, enabled=False):
    value = copy.copy(config)
    value.adapt_navigation_conditioning = enabled
    return AdaptDecoder(8, value, torch.device("cpu"))


def sample():
    data = navigation()
    data.update(
        speed=torch.tensor([3.0, 4.0]),
        past_positions=torch.tensor([[[-1.0, 0.0], [-0.5, 0.0], [0.0, 0.0]]]).repeat(
            2,
            1,
            1,
        ),
        past_yaws=torch.zeros(2, 3),
        future_waypoints=torch.tensor(
            [[[0.5, 0.0], [1.0, 0.1], [1.5, 0.3], [2.0, 0.6]]],
        ).repeat(2, 1, 1),
        future_yaws=torch.tensor([[0.0, 0.1, 0.2, 0.3]]).repeat(2, 1),
        route=torch.tensor([[[2.5, 0.0], [3.5, 0.1], [4.5, 0.3], [5.5, 0.6]]]).repeat(
            2,
            1,
            1,
        ),
        target_speed=torch.tensor([4.0, 5.0]),
        brake=torch.zeros(2, dtype=torch.bool),
    )
    return data


def call(model, bev, data):
    return model(bev, None, None, data, {})


def test_geometry_uses_physical_directions_and_ordered_targets():
    encoder = NavigationEncoder(16, 6, [[200, 50]])
    features = encoder.geometry_features(navigation(1))
    expected_geometry = torch.tensor(
        [
            [
                -3 / 200,
                0,
                0,
                0,
                3 / 200,
                4 / 50,
                3 / 200,
                0,
                3 / 200,
                4 / 50,
                3 / 50,
                5 / 50,
                1,
                0,
                0.6,
                0.8,
                1,
                1,
                3 / 50,
                0,
                5 / 50,
            ],
        ],
    )
    assert features.shape == (1, 33)
    torch.testing.assert_close(features[:, :21], expected_geometry)
    torch.testing.assert_close(features[:, 21:27], navigation(1)["command"])
    torch.testing.assert_close(features[:, 27:], navigation(1)["next_command"])
    swapped = navigation(1)
    swapped["target_point_previous"], swapped["target_point_next"] = (
        swapped["target_point_next"],
        swapped["target_point_previous"],
    )
    assert not torch.equal(encoder(swapped), encoder(navigation(1)))


def test_duplicate_targets_have_finite_features_and_gradients():
    encoder = NavigationEncoder(16, 6, [200, 50])
    data = navigation()
    for key in encoder.TARGET_KEYS:
        data[key] = torch.zeros(2, 2, requires_grad=True)
    features = encoder.geometry_features(data)
    assert torch.equal(features[:, :21], torch.zeros(2, 21))
    encoder(data).square().sum().backward()
    for key in encoder.TARGET_KEYS:
        assert torch.isfinite(data[key].grad).all()


@pytest.mark.parametrize(
    "normalization",
    [[0, 50], [-1, 50], [1], [1, 2, 3], [float("nan"), 50]],
)
def test_invalid_coordinate_scales_fail_at_construction(normalization):
    with pytest.raises(ValueError, match="normalization"):
        NavigationEncoder(16, 6, normalization)


def test_missing_next_command_is_not_silently_replaced():
    encoder = NavigationEncoder(16, 6, [200, 50])
    data = navigation()
    del data["next_command"]
    with pytest.raises(KeyError, match="next_command"):
        encoder(data)
    data = navigation()
    data["next_command"] = torch.ones(2, 4)
    with pytest.raises(ValueError, match="next_command"):
        encoder(data)


def test_modulation_starts_as_identity_but_connections_can_learn():
    module = NavigationModulation(16)
    queries, nav = torch.randn(2, 5, 16), torch.randn(2, 16)
    assert torch.equal(module(queries, nav), queries)
    module(queries, nav).square().mean().backward()
    assert module.affine.weight.grad.norm() > 0


def test_host_navigation_moves_to_the_encoder_device():
    # Meta exercises the cross-device contract even in CPU-only CI. Legacy
    # training supplies CPU batches unless optional CUDA prefetching is enabled.
    encoder = NavigationEncoder(16, 6, [200, 50]).to("meta")
    assert encoder(navigation()).device.type == "meta"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA integration")
@pytest.mark.parametrize("prefetched", [False, True])
@pytest.mark.parametrize("training", [False, True])
def test_cuda_bfloat16_planning_accepts_host_or_prefetched_navigation(
    config,
    prefetched,
    training,
):
    from lead.adapt.transfuser_utils import patch_norm_fp32

    model = patch_norm_fp32(decoder(config, enabled=True)).cuda().train(training)
    data = navigation()
    if prefetched:
        data = {key: tensor.cuda() for key, tensor in data.items()}
    context = torch.randn(2, 9, 16, device="cuda", requires_grad=training)
    with torch.set_grad_enabled(training), torch.autocast("cuda", dtype=torch.bfloat16):
        expected = model._plan_decoder(model._plan_query.expand(2, -1, -1), context)
        actual = model._decode_plan_queries(context, data)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        loss = model._route_decoder(actual).float().square().mean()
    assert torch.isfinite(actual).all()
    if training:
        loss.backward()
        assert torch.isfinite(context.grad).all()
        for modulation in model._navigation_modulations:
            assert torch.isfinite(modulation.affine.weight.grad).all()
            assert modulation.affine.weight.grad.norm() > 0


def test_old_and_enabled_checkpoints_load_with_expected_keys(config):
    legacy = decoder(config)
    assert not any("_navigation_" in key for key in legacy.state_dict())
    legacy_clone = decoder(config)
    legacy_clone.load_state_dict(legacy.state_dict(), strict=True)
    enabled = decoder(config, enabled=True)
    result = enabled.load_state_dict(legacy.state_dict(), strict=False)
    assert result.missing_keys
    assert all(key.startswith("_navigation_") for key in result.missing_keys)
    assert not result.unexpected_keys
    checkpoint = io.BytesIO()
    torch.save(enabled.state_dict(), checkpoint)
    checkpoint.seek(0)
    restored = decoder(config, enabled=True)
    restored.load_state_dict(torch.load(checkpoint, weights_only=True), strict=True)
    bev, data = torch.randn(2, 8, 2, 3), sample()
    with torch.inference_mode():
        original = call(enabled.eval(), bev, data)
        reloaded = call(restored.eval(), bev, data)
    for key in ("pred_route", "pred_target_speed_distribution", "trajectory"):
        torch.testing.assert_close(original[key], reloaded[key], atol=0, rtol=0)


@pytest.mark.parametrize("training", [False, True])
def test_identity_warm_start_preserves_full_decoder_predictions(config, training):
    torch.manual_seed(71)
    legacy = decoder(config).train(training)
    conditioned = decoder(config, enabled=True).train(training)
    conditioned.load_state_dict(legacy.state_dict(), strict=False)
    bev, data = torch.randn(2, 8, 2, 3), sample()
    reference = call(legacy, bev, data)
    actual = call(conditioned, bev, data)
    for key in (
        "pred_route",
        "pred_target_speed_distribution",
        "trajectory",
        "output_logits",
    ):
        torch.testing.assert_close(actual[key], reference[key], atol=0, rtol=0)


def test_route_and_speed_losses_train_all_navigation_connections(config):
    model = decoder(config, enabled=True).train()
    bev, data = torch.randn(2, 8, 2, 3), sample()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    # First update opens the zero-initialized adapters; subsequent updates train
    # the joint encoder too. Use the real route/speed losses, not an aux nav loss.
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        prediction = call(model, bev, data)
        losses = {}
        model.compute_loss(data, prediction, losses, {})
        (losses["loss_spatial_route"] + losses["loss_target_speed"]).backward()
        optimizer.step()
    for name, parameter in model.named_parameters():
        if "_navigation_" in name:
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
            assert parameter.grad.norm() > 0, name


def test_learned_next_command_conditions_planning_without_changing_ar(config):
    model = decoder(config, enabled=True).eval()
    torch.nn.init.normal_(model._navigation_query_init.weight, std=0.02)
    for modulation in model._navigation_modulations:
        torch.nn.init.normal_(modulation.affine.weight, std=0.02)
    bev, data = torch.randn(2, 8, 2, 3), sample()
    other = copy.deepcopy(data)
    other["next_command"] = other["next_command"].roll(1, dims=-1)
    with torch.inference_mode():
        first, second = call(model, bev, data), call(model, bev, other)
    assert not torch.equal(first["pred_route"], second["pred_route"])
    assert not torch.equal(
        first["pred_target_speed_distribution"],
        second["pred_target_speed_distribution"],
    )
    torch.testing.assert_close(
        first["trajectory"],
        second["trajectory"],
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        first["output_logits"],
        second["output_logits"],
        atol=0,
        rtol=0,
    )


def test_future_labels_do_not_condition_the_executed_planning_head(config):
    model = decoder(config, enabled=True).train()
    torch.nn.init.normal_(model._navigation_query_init.weight, std=0.02)
    bev, data = torch.randn(2, 8, 2, 3), sample()
    other = copy.deepcopy(data)
    other["future_waypoints"] *= 4
    other["future_yaws"] += 0.5
    first, second = call(model, bev, data), call(model, bev, other)
    for key in ("pred_route", "pred_target_speed_distribution"):
        torch.testing.assert_close(first[key], second[key], atol=0, rtol=0)


def test_navigation_encoder_compiles_without_graph_breaks():
    encoder = NavigationEncoder(16, 6, [200, 50])
    compiled = torch.compile(encoder, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(navigation()), encoder(navigation()))


def test_conditioned_planning_compiles_without_graph_breaks(config):
    model = decoder(config, enabled=True)
    context = torch.randn(2, 9, 16)
    compiled = torch.compile(
        model._decode_plan_queries,
        backend="eager",
        fullgraph=True,
    )
    torch.testing.assert_close(
        compiled(context, navigation()),
        model._decode_plan_queries(context, navigation()),
    )


def test_bfloat16_training_with_production_norm_patching(config):
    from lead.adapt.transfuser_utils import patch_norm_fp32

    model = patch_norm_fp32(decoder(config, enabled=True).train())
    context = torch.randn(2, 9, 16, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert (
            model._navigation_encoder.geometry_features(navigation()).dtype
            == torch.float32
        )
        queries = model._decode_plan_queries(context, navigation())
        prediction = model._route_decoder(queries)
        loss = prediction.float().square().mean()
    loss.backward()
    assert torch.isfinite(prediction).all()
    assert torch.isfinite(context.grad).all()
    for modulation in model._navigation_modulations:
        assert torch.isfinite(modulation.affine.weight.grad).all()
        assert modulation.affine.weight.grad.norm() > 0


def test_config_defaults_and_serialization(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["test_navigation_conditioning"])
    monkeypatch.delenv("LEAD_TRAINING_CONFIG", raising=False)
    assert TrainingConfig({}).adapt_navigation_conditioning is False
    config = TrainingConfig({"adapt_navigation_conditioning": True})
    assert config.training_dict()["adapt_navigation_conditioning"] is True
    restored = TrainingConfig(config.training_dict())
    assert restored.adapt_navigation_conditioning is True


def test_conditioning_rejects_missing_navigation_configuration(config):
    config._loaded_config = {"use_discrete_command": False}
    with pytest.raises(ValueError, match="requires previous/current/next"):
        decoder(config, enabled=True)
