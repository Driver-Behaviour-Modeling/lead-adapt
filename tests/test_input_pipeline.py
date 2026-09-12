"""The optional input pipeline must not alter the legacy training path."""

from types import SimpleNamespace

import pytest
import torch

from lead.training.input_pipeline import prepared_training_batches


def config(**overrides):
    values = dict(
        cuda_prefetch=False,
        gpu_color_augmentation=False,
        use_color_aug=True,
        use_color_aug_prob=0.0,
        use_carla_data=True,
        use_navsim_data=False,
        use_waymo_e2e_data=False,
        device=torch.device("cpu"),
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_disabled_pipeline_preserves_batch_identity_and_metadata():
    batch = {"rgb": torch.ones(2, 3, 16, 16, dtype=torch.uint8), "path": ["a", "b"]}
    with prepared_training_batches([batch], config()) as batches:
        assert next(batches) is batch


def test_zero_probability_augmentation_preserves_pixels_and_labels():
    pixels = torch.randint(0, 256, (2, 3, 16, 16), dtype=torch.uint8)
    label = torch.tensor([3, 4])
    batch = {"rgb": pixels.clone(), "label": label}
    with prepared_training_batches(
        [batch],
        config(gpu_color_augmentation=True),
    ) as batches:
        result = next(batches)
    torch.testing.assert_close(result["rgb"], pixels)
    assert result["label"] is label


def test_unsupported_mixed_dataset_is_rejected_before_consumption():
    def source():
        raise AssertionError("Must not read the dataset")
        yield  # pragma: no cover

    with pytest.raises(ValueError, match="CARLA-only"):
        with prepared_training_batches(
            source(),
            config(gpu_color_augmentation=True, use_navsim_data=True),
        ):
            pass
