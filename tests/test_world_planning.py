"""Predicted futures condition both ego heads without exposing actor labels."""

import copy
import pickle

import numpy as np
import pytest
import torch
from test_navigation_conditioning import SmallConfig, sample

from lead.adapt.adapt_decoder import AdaptDecoder
from lead.adapt.transfuser_utils import patch_norm_fp32
from lead.adapt.world_attention import FutureContextFusion
from lead.adapt.world_model import VehicleWorldModel
from lead.common.constants import SourceDataset


@pytest.fixture
def config(tmp_path):
    value = SmallConfig()
    value.adapt_navigation_conditioning = True
    value.adapt_world_model = True
    value.world_num_steps = 4
    value.world_num_modes = 3
    value.world_hidden_dim = value.radar_token_dim = 16
    path = tmp_path / "vocab.pkl"
    centroids = np.zeros((8, 3), np.float32)
    centroids[:, 0] = np.arange(8) * 0.5
    path.write_bytes(pickle.dumps(dict(centroids=centroids, info={})))
    value.kdisks_vocab_path = str(path)
    return value


def call(model, data, memory=None, confidence=None):
    return model(torch.ones(2, 8, 2, 2), None, None, data, {}, memory, confidence)


@pytest.mark.parametrize("training", [False, True])
def test_world_identity_preserves_nav_checkpoint(config, training):
    legacy_config = copy.copy(config)
    legacy_config.adapt_world_model = False
    legacy = AdaptDecoder(8, legacy_config, torch.device("cpu")).train(training)
    extended = AdaptDecoder(8, config, torch.device("cpu")).train(training)
    result = extended.load_state_dict(legacy.state_dict(), strict=False)
    assert result.missing_keys and all(
        k.startswith("_world_fusion.") for k in result.missing_keys
    )
    assert result.unexpected_keys == []
    assert not any("world" in k for k in legacy.state_dict())
    data = sample()
    with torch.no_grad():
        expected = call(legacy, data)
        actual = call(extended, data, torch.randn(2, 6, 16), torch.rand(2, 6))
    for field in [
        "pred_route",
        "pred_target_speed_distribution",
        "pred_future_waypoints",
        "output_logits",
    ]:
        torch.testing.assert_close(actual[field], expected[field], rtol=0, atol=0)
    with pytest.raises(ValueError, match="predicted actor futures"):
        call(extended, data)
    restored = AdaptDecoder(8, config, torch.device("cpu"))
    restored.load_state_dict(extended.state_dict(), strict=True)


def test_actor_labels_cannot_change_planner_forward(config):
    model = AdaptDecoder(8, config, torch.device("cpu")).eval()
    with torch.no_grad():
        model._world_fusion.attention.out_proj.weight.normal_(std=0.05)
    data = sample()
    data.update(
        world_future_positions=torch.randn(2, 2, 4, 2),
        world_future_mask=torch.ones(2, 2, 4, dtype=torch.bool),
    )
    memory, confidence = torch.randn(2, 6, 16), torch.rand(2, 6)
    with torch.no_grad():
        original = call(model, data, memory, confidence)
        data["world_future_positions"].fill_(float("nan"))
        data["world_future_mask"].zero_()
        changed_labels = call(model, data, memory, confidence)
        changed_memory = call(model, data, torch.randn_like(memory), confidence)
    for field in [
        "pred_route",
        "pred_target_speed_distribution",
        "pred_future_waypoints",
    ]:
        torch.testing.assert_close(
            original[field],
            changed_labels[field],
            rtol=0,
            atol=0,
        )
        assert not torch.equal(original[field], changed_memory[field])


def test_planning_gradient_reaches_predicted_future_states(config):
    planner = AdaptDecoder(8, config, torch.device("cpu")).train()
    world = VehicleWorldModel(config)
    with torch.no_grad():
        planner._world_fusion.attention.out_proj.weight.normal_(std=0.05)
    futures = world(torch.randn(2, 2, 16), torch.randn(2, 2, 4))
    futures["future_positions"].retain_grad()
    data = sample()
    output = call(planner, data, futures["memory"], futures["memory_confidence"])
    losses = {}
    planner.compute_loss(data, output, losses, {})
    (losses["loss_spatial_route"] + losses["loss_target_speed"]).backward()
    grad = futures["future_positions"].grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    assert world.displacement_head.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_future_fusion_bf16_empty_confidence(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    fusion = patch_norm_fp32(FutureContextFusion(16, 2)).to(device)
    context = torch.randn(2, 5, 16, device=device, requires_grad=True)
    memory = torch.randn(2, 6, 16, device=device, requires_grad=True)
    confidence = torch.zeros(2, 6, device=device)
    with torch.autocast(device, dtype=torch.bfloat16):
        output = fusion(context, memory, confidence)
    torch.testing.assert_close(output, context, rtol=0, atol=0)
    output.square().mean().backward()
    assert torch.isfinite(fusion.attention.out_proj.weight.grad).all()
    assert fusion.attention.out_proj.weight.grad.abs().sum() > 0
    assert torch.isfinite(memory.grad).all()


def test_world_losses_are_registered_only_when_enabled(config):
    weights = config.detailed_loss_weights(SourceDataset.CARLA, 0)
    assert weights["world_loss_position"] == 1.0
    assert weights["world_loss_mode"] == weights["world_loss_vehicle"] == 0.1
    config.adapt_world_model = False
    assert not any(
        key.startswith("world_loss_")
        for key in config.detailed_loss_weights(SourceDataset.CARLA, 0)
    )
