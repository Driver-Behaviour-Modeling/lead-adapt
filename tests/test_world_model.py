"""Forecast identity, missing-label masking, and predicted-only planner memory."""

import copy
import inspect
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from lead.adapt.world_model import VehicleWorldModel


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def config():
    return SimpleNamespace(
        world_num_steps=3,
        world_num_modes=3,
        world_step_seconds=0.25,
        world_hidden_dim=16,
        world_position_scale=10.0,
        radar_token_dim=8,
        kinematic_embed_dim=12,
        max_speed=25.0,
    )


def inputs(batch=2, queries=3):
    features = torch.randn(batch, queries, 8)
    predictions = torch.randn(batch, queries, 4)
    predictions[..., :2] *= 5.0
    predictions[..., 2] = 3.0
    return features, predictions


def targets(config, batch=2, queries=3):
    detections = torch.zeros(batch, queries, 4)
    detections[..., 3] = 1.0
    return {
        "world_future_positions": torch.randn(
            batch,
            queries,
            config.world_num_steps,
            2,
        ),
        "world_future_mask": torch.ones(
            batch,
            queries,
            config.world_num_steps,
            dtype=torch.bool,
        ),
        "world_vehicle_mask": torch.ones(batch, queries, dtype=torch.bool),
        "radar_detections": detections,
    }


def identity_matching(batch=2, queries=3):
    indices = torch.arange(queries).repeat(batch, 1)
    return indices, indices


def losses(model, outputs, data, matching=None):
    loss, log = {}, {}
    if matching is None:
        matching = identity_matching(*outputs["future_positions"].shape[:2])
    model.compute_loss(outputs, data, matching, loss, log)
    return loss, log


def test_shapes_motion_bounds_and_interval_velocity(config):
    model = VehicleWorldModel(config)
    features, radar = inputs()
    # Saturate the output to exercise the diagonal vector-speed bound.
    with torch.no_grad():
        model.displacement_head.weight.zero_()
        model.displacement_head.bias.fill_(100)
    outputs = model(features, radar)
    assert outputs["future_positions"].shape == (2, 3, 3, 3, 2)
    assert outputs["future_velocities"].shape == (2, 3, 3, 3, 2)
    assert outputs["mode_logits"].shape == (2, 3, 3)
    assert outputs["vehicle_logits"].shape == (2, 3)
    assert outputs["memory"].shape == (2, 9, 12)
    assert outputs["memory_confidence"].shape == (2, 9)
    assert (outputs["future_velocities"].norm(dim=-1) <= 25.00001).all()
    anchors = radar[..., None, None, :2].expand(-1, -1, 3, 1, -1)
    displacements = torch.cat([anchors, outputs["future_positions"]], dim=-2).diff(
        dim=-2,
    )
    torch.testing.assert_close(
        displacements / config.world_step_seconds,
        outputs["future_velocities"],
        atol=1e-5,
        rtol=1e-5,
    )
    assert all(torch.isfinite(value).all() for value in outputs.values())


def test_actor_speed_bound_is_independent_of_ego_speed_normalization(config):
    config.world_max_speed = 40.0
    model = VehicleWorldModel(config)
    features, radar = inputs()
    radar[..., 2] = 25.0
    with torch.no_grad():
        model.displacement_head.weight.zero_()
        model.displacement_head.bias.fill_(100)
    encoded_inputs = []
    hook = model.actor_encoder[0].register_forward_pre_hook(
        lambda _module, args: encoded_inputs.append(args[0].detach().clone()),
    )
    try:
        outputs = model(features, radar)
    finally:
        hook.remove()
    speeds = outputs["future_velocities"].norm(dim=-1)
    torch.testing.assert_close(speeds, torch.full_like(speeds, 40.0))
    torch.testing.assert_close(
        encoded_inputs[0][..., config.radar_token_dim + 2],
        torch.ones_like(radar[..., 2]),
    )
    assert model.radar_speed_scale.item() == 25.0
    assert model.speed_scale.item() == 40.0


def test_forward_accepts_observations_only(config):
    parameters = tuple(inspect.signature(VehicleWorldModel.forward).parameters)
    assert parameters == ("self", "radar_features", "radar_predictions")
    model = VehicleWorldModel(config).eval()
    features, radar = inputs()
    initial = model(features, radar)
    data = targets(config)
    data["world_future_positions"].fill_(float("nan"))
    # No data/label dictionary enters forecasting or memory creation.
    actual = model(features, radar)
    for key in initial:
        torch.testing.assert_close(actual[key], initial[key], atol=0, rtol=0)


def test_memory_has_no_hidden_feature_shortcut(config):
    model = VehicleWorldModel(config)
    with torch.no_grad():
        for head in (model.displacement_head, model.mode_head, model.vehicle_head):
            head.weight.zero_()
    features, radar = inputs()
    first = model(features, radar)
    second = model(torch.randn_like(features) * 100, radar)
    for key in first:
        torch.testing.assert_close(first[key], second[key], atol=0, rtol=0)
    changed_memory, _ = model.encode_memory(
        first["future_positions"] + 5,
        first["future_velocities"],
        first["mode_logits"],
        first["vehicle_logits"],
        radar[..., 3],
    )
    assert not torch.allclose(changed_memory, first["memory"])


def test_confidence_is_probabilistic_and_all_memory_paths_are_differentiable(config):
    model = VehicleWorldModel(config)
    features, radar = inputs()
    radar.requires_grad_()
    outputs = model(features, radar)
    expected = (
        outputs["mode_logits"].softmax(-1)
        * outputs["vehicle_logits"].sigmoid().unsqueeze(-1)
        * radar[..., 3].sigmoid().unsqueeze(-1)
    )
    torch.testing.assert_close(outputs["memory_confidence"], expected.flatten(1, 2))
    # Nonuniform readout mimics planner attention and avoids summing LayerNorm.
    objective = (
        outputs["memory"]
        * outputs["memory_confidence"].unsqueeze(-1)
        * torch.randn_like(outputs["memory"])
    ).sum()
    objective.backward()
    for head in (model.displacement_head, model.mode_head, model.vehicle_head):
        assert torch.isfinite(head.weight.grad).all()
        assert head.weight.grad.abs().sum() > 0
    assert torch.isfinite(radar.grad).all()
    assert radar.grad[..., 3].abs().sum() > 0


def test_permuting_labels_and_shared_assignment_preserves_losses(config):
    model = VehicleWorldModel(config)
    outputs = model(*inputs())
    data = targets(config)
    # Use nontrivial prediction ordering to exercise both matching arrays.
    pred_index = torch.tensor([[2, 0, 1], [1, 2, 0]])
    gt_index = torch.tensor([[1, 2, 0], [2, 0, 1]])
    expected, expected_log = losses(model, outputs, data, (pred_index, gt_index))
    permutation = torch.tensor([2, 0, 1])
    inverse = permutation.argsort()
    permuted = {key: value[:, permutation] for key, value in data.items()}
    actual, actual_log = losses(
        model,
        outputs,
        permuted,
        (pred_index, inverse[gt_index]),
    )
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
    for key in expected_log:
        torch.testing.assert_close(actual_log[key], expected_log[key], atol=0, rtol=0)


def test_crossing_actors_keep_current_detector_identity(config):
    model = VehicleWorldModel(config)
    data = targets(config, batch=1, queries=2)
    data["world_future_positions"] = torch.tensor(
        [[[[-1.0, 0], [0, 0], [1, 0]], [[1.0, 0], [0, 0], [-1, 0]]]],
    )
    predicted = data["world_future_positions"].unsqueeze(2).repeat(1, 1, 3, 1, 1)
    outputs = {
        "future_positions": predicted.requires_grad_(),
        "mode_logits": torch.zeros(1, 2, 3, requires_grad=True),
        "vehicle_logits": torch.zeros(1, 2, requires_grad=True),
    }
    correct, _ = losses(model, outputs, data)
    assert correct["world_loss_position"] == 0
    wrong = copy.deepcopy(data)
    wrong["world_future_positions"] = wrong["world_future_positions"][:, [1, 0]]
    inconsistent, _ = losses(model, outputs, wrong)
    assert inconsistent["world_loss_position"] > 0


def test_masked_nan_future_is_ignored_and_has_no_gradient(config):
    model = VehicleWorldModel(config)
    data = targets(config)
    data["world_future_mask"][:, :, 1:] = False
    outputs = model(*inputs())
    outputs["future_positions"].retain_grad()
    expected, _ = losses(model, outputs, data)
    data["world_future_positions"][:, :, 1:] = float("nan")
    actual, _ = losses(model, outputs, data)
    torch.testing.assert_close(
        actual["world_loss_position"],
        expected["world_loss_position"],
    )
    sum(actual.values()).backward()
    gradient = outputs["future_positions"].grad
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient[:, :, :, 1:]) == 0
    assert gradient[:, :, :, 0].abs().sum() > 0


@pytest.mark.parametrize("empty_kind", ["future", "nonvehicle", "padding"])
def test_empty_supervision_gives_finite_graph_connected_zero(config, empty_kind):
    model = VehicleWorldModel(config)
    outputs = model(*inputs())
    data = targets(config)
    if empty_kind == "future":
        data["world_future_mask"].zero_()
    elif empty_kind == "nonvehicle":
        data["world_vehicle_mask"].zero_()
    else:
        data["radar_detections"][..., 3].zero_()
    data["world_future_positions"].fill_(float("nan"))
    loss, log = losses(model, outputs, data)
    assert loss["world_loss_position"].item() == 0
    assert loss["world_loss_mode"].item() == 0
    assert log["metric/world_supervised_actors"].item() == 0
    assert all(torch.isfinite(value) for value in loss.values())
    assert all(torch.isfinite(value) for value in log.values())
    sum(loss.values()).backward()
    assert model.displacement_head.weight.grad is not None
    assert model.mode_head.weight.grad is not None
    assert torch.count_nonzero(model.displacement_head.weight.grad) == 0
    assert torch.count_nonzero(model.mode_head.weight.grad) == 0
    assert torch.isfinite(model.actor_encoder[0].weight.grad).all()


def test_missing_future_does_not_make_current_vehicle_negative(config):
    model = VehicleWorldModel(config)
    outputs = model(*inputs())
    data = targets(config)
    data["world_future_mask"].zero_()
    loss, _ = losses(model, outputs, data)
    expected = F.binary_cross_entropy_with_logits(
        outputs["vehicle_logits"],
        torch.ones_like(outputs["vehicle_logits"]),
    )
    torch.testing.assert_close(loss["world_loss_vehicle"], expected)


def test_position_loss_normalizes_each_actors_available_horizon(config):
    model = VehicleWorldModel(config)
    data = targets(config, batch=1, queries=2)
    data["world_future_positions"].zero_()
    data["world_future_mask"][0, 1, 1:] = False
    # Actor 0 has 1 m per-coordinate error; actor 1 has 3 m. Equal actor
    # weighting gives (1 + 3) / 2 / scale = 0.2 despite different horizons.
    positions = torch.ones(1, 2, 3, 3, 2)
    positions[0, 1] = 3
    outputs = {
        "future_positions": positions,
        "mode_logits": torch.zeros(1, 2, 3),
        "vehicle_logits": torch.zeros(1, 2),
    }
    loss, _ = losses(model, outputs, data)
    torch.testing.assert_close(loss["world_loss_position"], torch.tensor(0.2))


def test_winning_mode_uses_available_steps_and_metrics_use_last_valid_step(config):
    model = VehicleWorldModel(config)
    data = targets(config, batch=1, queries=1)
    data["world_future_positions"].zero_()
    data["world_future_mask"][0, 0] = torch.tensor([True, False, True])
    positions = torch.zeros(1, 1, 3, 3, 2)
    positions[0, 0, 0, :, 0] = torch.tensor([1, 1000, 3])
    positions[0, 0, 1, :, 0] = torch.tensor([4, 0, 4])
    positions[0, 0, 2, :, 0] = torch.tensor([5, 0, 5])
    outputs = {
        "future_positions": positions,
        "mode_logits": torch.tensor([[[0.0, 3.0, 0.0]]]),
        "vehicle_logits": torch.zeros(1, 1),
    }
    loss, log = losses(model, outputs, data)
    torch.testing.assert_close(log["metric/world_min_ade"], torch.tensor(2.0))
    torch.testing.assert_close(log["metric/world_mode_ade"], torch.tensor(4.0))
    torch.testing.assert_close(log["metric/world_min_fde"], torch.tensor(3.0))
    expected_ce = F.cross_entropy(
        outputs["mode_logits"].reshape(1, 3),
        torch.tensor([0]),
    )
    torch.testing.assert_close(loss["world_loss_mode"], expected_ce)


def test_serialization_preserves_forecasts_and_registered_geometry(config):
    model = VehicleWorldModel(config)
    restored = VehicleWorldModel(config)
    restored.load_state_dict(model.state_dict(), strict=True)
    features, radar = inputs()
    original, actual = model(features, radar), restored(features, radar)
    for key in original:
        torch.testing.assert_close(actual[key], original[key], atol=0, rtol=0)
    assert {
        "step_seconds",
        "position_scale",
        "speed_scale",
        "radar_speed_scale",
    } <= set(model.state_dict())


@pytest.mark.parametrize("prefetched", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_bfloat16_forward_and_loss_accept_host_or_prefetched_inputs(
    config,
    device,
    prefetched,
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA integration")
    from lead.adapt.transfuser_utils import patch_norm_fp32

    model = patch_norm_fp32(VehicleWorldModel(config)).to(device)
    features, radar = inputs()
    data = targets(config)
    if prefetched:
        features, radar = features.to(device), radar.to(device)
        data = {key: value.to(device) for key, value in data.items()}
    with torch.autocast(device, dtype=torch.bfloat16):
        outputs = model(features, radar)
        loss, _ = losses(model, outputs, data)
        objective = (
            sum(loss.values())
            + (
                outputs["memory"].float() * torch.randn_like(outputs["memory"].float())
            ).mean()
        )
    assert outputs["future_positions"].dtype == torch.float32
    assert outputs["future_positions"].device.type == device
    assert all(torch.isfinite(value).all() for value in outputs.values())
    assert all(value.dtype == torch.float32 for value in loss.values())
    objective.backward()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


@pytest.mark.parametrize(
    "name,value",
    [
        ("world_num_steps", 0),
        ("world_num_modes", 0),
        ("world_hidden_dim", 1.5),
        ("world_step_seconds", 0.0),
        ("world_position_scale", float("nan")),
        ("max_speed", -1.0),
        ("world_max_speed", -1.0),
        ("world_max_speed", float("inf")),
    ],
)
def test_invalid_geometry_configuration_fails_early(config, name, value):
    setattr(config, name, value)
    with pytest.raises(ValueError, match=name):
        VehicleWorldModel(config)


def test_malformed_targets_are_not_silently_broadcast(config):
    model = VehicleWorldModel(config)
    outputs = model(*inputs())
    data = targets(config)
    data["world_future_positions"] = data["world_future_positions"][:, :1]
    with pytest.raises(ValueError, match="world_future_positions"):
        losses(model, outputs, data)
