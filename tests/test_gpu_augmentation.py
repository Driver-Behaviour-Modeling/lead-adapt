"""The opt-in device recipe preserves the uint8 model-input contract."""

import os

import pytest
import torch

from lead.training.gpu_augmentation import augment_rgb_batch


def test_zero_probability_preserves_pixels_without_consuming_randomness():
    rgb = torch.randint(0, 256, (2, 3, 12, 16), dtype=torch.uint8)
    state = torch.get_rng_state()
    assert augment_rgb_batch(rgb, 0.0) is rgb
    torch.testing.assert_close(torch.get_rng_state(), state)


@pytest.mark.parametrize("probability", [-0.1, 1.1, float("nan")])
def test_invalid_probability_is_rejected(probability):
    with pytest.raises(ValueError, match="probability"):
        augment_rgb_batch(torch.zeros(1, 3, 12, 16, dtype=torch.uint8), probability)


def test_input_contract_rejects_normalized_or_malformed_images():
    with pytest.raises(TypeError, match="uint8"):
        augment_rgb_batch(torch.zeros(1, 3, 12, 16), 0.2)
    with pytest.raises(ValueError, match="shape"):
        augment_rgb_batch(torch.zeros(1, 12, 16, 3, dtype=torch.uint8), 0.2)


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_full_recipe_is_seeded_bounded_and_does_not_mutate_input(device_name):
    if device_name == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device_name = os.environ.get("LEAD_TEST_CUDA_DEVICE", "cuda:0")
    device = torch.device(device_name)
    rgb = torch.arange(3 * 12 * 16, dtype=torch.int64).remainder(256)
    rgb = (
        rgb.to(dtype=torch.uint8, device=device)
        .reshape(1, 3, 12, 16)
        .repeat(3, 1, 1, 1)
    )
    original = rgb.clone()
    torch.manual_seed(4321)
    first = augment_rgb_batch(rgb, 1.0)
    torch.manual_seed(4321)
    second = augment_rgb_batch(rgb, 1.0)
    assert first.device == device
    assert first.dtype == torch.uint8
    assert first.shape == rgb.shape
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(rgb, original)
    assert not torch.equal(first, rgb)
    # Gates/parameters are sampled per item, not shared across the whole batch.
    assert not torch.equal(first[0], first[1])
