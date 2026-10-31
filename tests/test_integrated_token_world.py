"""Combined context must preserve a proper, causal first-step distribution."""

import copy
import pickle

import numpy as np
import pytest
import torch
from test_navigation_conditioning import SmallConfig, sample

from lead.adapt.adapt_decoder import AdaptDecoder
from lead.common.constants import SourceDataset


@pytest.fixture
def config(tmp_path):
    config = SmallConfig()
    config.adapt_history_free = True
    config.adapt_rolling_tokenization = True
    config.adapt_rolling_causal_first = True
    config.adapt_waypoints_from_tokens = True
    config.adapt_navigation_conditioning = True
    config.adapt_token_navigation_conditioning = True
    config.adapt_world_model = True
    config.world_num_steps = 4
    config.world_num_modes = 3
    config.world_hidden_dim = config.radar_token_dim = 16
    config.use_scheduled_sampling = False
    codes = np.zeros((8, 3), np.float32)
    codes[:, 0] = np.arange(8) * 0.5
    codes[3:, 1] = np.arange(5) * 0.15
    path = tmp_path / "codes.pkl"
    path.write_bytes(
        pickle.dumps(
            {
                "centroids": codes,
                "info": {
                    "frame": "body",
                    "heading_weight": config.kdisks_heading_weight,
                },
            },
        ),
    )
    config.kdisks_vocab_path = str(path)
    return config


def call(model, data, memory, confidence):
    return model(torch.ones(2, 8, 2, 2), None, None, data, {}, memory, confidence)


def test_combined_identity_warmstart_preserves_histfree_checkpoint(config):
    old = copy.copy(config)
    old.adapt_world_model = old.adapt_token_navigation_conditioning = False
    baseline = AdaptDecoder(8, old, torch.device("cpu")).eval()
    combined = AdaptDecoder(8, config, torch.device("cpu")).eval()
    result = combined.load_state_dict(baseline.state_dict(), strict=False)
    assert not result.unexpected_keys
    assert all(
        key.startswith(("_world_fusion.", "_token_navigation_modulation."))
        for key in result.missing_keys
    )
    data = sample()
    with torch.no_grad():
        before = call(baseline, data, None, None)
        after = call(combined, data, torch.randn(2, 6, 16), torch.rand(2, 6))
    for key in (
        "output_logits",
        "pred_future_waypoints",
        "pred_route",
        "pred_target_speed_distribution",
    ):
        torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)


def test_first_token_target_cannot_depend_on_later_future(config):
    model = AdaptDecoder(8, config, torch.device("cpu")).train()
    data = sample()
    memory, confidence = torch.randn(2, 6, 16), torch.rand(2, 6)
    first = call(model, data, memory, confidence)
    changed = copy.deepcopy(data)
    changed["future_waypoints"][:, 1:] += 3
    changed["future_yaws"][:, 1:] += 1
    second = call(model, changed, memory, confidence)
    torch.testing.assert_close(
        first["future_token_ids"][:, 0],
        second["future_token_ids"][:, 0],
    )
    torch.testing.assert_close(
        first["output_logits"][:, 0],
        second["output_logits"][:, 0],
        rtol=0,
        atol=0,
    )
    delta = torch.cat(
        [data["future_waypoints"][:, 0], data["future_yaws"][:, :1]],
        dim=-1,
    )
    nearest = model.kdisks_model.encode(delta).reshape(-1)
    torch.testing.assert_close(first["future_token_ids"][:, 0], nearest)


def test_token_logits_use_navigation_and_world_but_not_history_or_labels(config):
    model = AdaptDecoder(8, config, torch.device("cpu")).eval()
    with torch.no_grad():
        model._token_navigation_modulation.affine.weight.normal_(std=0.1)
        model._world_fusion.attention.out_proj.weight.normal_(std=0.1)
    data = sample()
    memory, confidence = torch.randn(2, 6, 16), torch.rand(2, 6)
    original = call(model, data, memory, confidence)["output_logits"]
    changed = copy.deepcopy(data)
    changed["past_positions"].fill_(float("nan"))
    changed["past_yaws"].fill_(float("nan"))
    for key in ("future_waypoints", "future_yaws"):
        del changed[key]
    changed["world_future_positions"] = torch.full((2, 2, 4, 2), float("nan"))
    torch.testing.assert_close(
        original,
        call(model, changed, memory, confidence)["output_logits"],
        rtol=0,
        atol=0,
    )
    changed["target_point_next"] += 20
    assert not torch.equal(
        original,
        call(model, changed, memory, confidence)["output_logits"],
    )
    assert not torch.equal(
        original,
        call(model, data, -memory, confidence)["output_logits"],
    )


def test_ce_gradient_reaches_both_context_branches(config):
    model = AdaptDecoder(8, config, torch.device("cpu")).train()
    data = sample()
    out = call(model, data, torch.randn(2, 6, 16), torch.rand(2, 6))
    loss, log = {}, {}
    model.compute_loss(data, out, loss, log)
    loss["loss_kinematic_token"].backward()
    assert model._world_fusion.attention.out_proj.weight.grad.abs().sum() > 0
    assert model._token_navigation_modulation.affine.weight.grad.abs().sum() > 0
    weights = config.detailed_loss_weights(SourceDataset.CARLA, 0)
    assert weights["loss_trajectory"] == weights["loss_soft_frechet"] == 0


def test_token_only_freezing_does_not_activate_world_losses(config):
    config.adapt_train_token_decoder_only = True
    weights = config.detailed_loss_weights(SourceDataset.CARLA, 0)
    assert all(
        weight == 0 for key, weight in weights.items() if key != "loss_kinematic_token"
    )


def test_inference_context_ablations_preserve_weights_and_reject_training(config):
    model = AdaptDecoder(8, config, torch.device("cpu")).eval()
    with torch.no_grad():
        model._token_navigation_modulation.affine.weight.normal_(std=0.1)
        model._world_fusion.attention.out_proj.weight.normal_(std=0.1)
    data = sample()
    memory, confidence = torch.randn(2, 6, 16), torch.rand(2, 6)
    weights = {key: value.clone() for key, value in model.state_dict().items()}
    normal = call(model, data, memory, confidence)["output_logits"]
    model.disable_token_navigation_for_inference = True
    assert not torch.equal(
        normal,
        call(model, data, memory, confidence)["output_logits"],
    )
    model.disable_token_navigation_for_inference = False
    model._world_fusion.disabled_for_inference = True
    assert not torch.equal(
        normal,
        call(model, data, memory, confidence)["output_logits"],
    )
    model._world_fusion.disabled_for_inference = False
    torch.testing.assert_close(
        normal,
        call(model, data, memory, confidence)["output_logits"],
        rtol=0,
        atol=0,
    )
    assert all(
        torch.equal(value, model.state_dict()[key]) for key, value in weights.items()
    )
    model.train()
    model.disable_token_navigation_for_inference = True
    with pytest.raises(RuntimeError, match="only be disabled for inference"):
        call(model, data, memory, confidence)
    model.disable_token_navigation_for_inference = False
    model._world_fusion.disabled_for_inference = True
    with pytest.raises(RuntimeError, match="only be disabled for inference"):
        call(model, data, memory, confidence)
