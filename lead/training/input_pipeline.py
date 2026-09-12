"""Optional training batch preparation shared by training and profiling."""

from contextlib import contextmanager, nullcontext

from lead.training.batch_pipeline import DeviceBatchIterator
from lead.training.gpu_augmentation import augment_rgb_batch


@contextmanager
def prepared_training_batches(batches, config):
    """Preserve the legacy path unless GPU transfer/augmentation is selected."""
    augment = config.gpu_color_augmentation and config.use_color_aug
    if augment and (
        not config.use_carla_data or config.use_navsim_data or config.use_waymo_e2e_data
    ):
        raise ValueError(
            "GPU color augmentation currently supports CARLA-only training",
        )
    manager = (
        DeviceBatchIterator(batches, config.device, enabled=config.cuda_prefetch)
        if config.cuda_prefetch or augment
        else nullcontext(iter(batches))
    )

    def prepared(iterator):
        for batch in iterator:
            if augment and batch.get("rgb") is not None:
                batch["rgb"] = augment_rgb_batch(
                    batch["rgb"],
                    config.use_color_aug_prob,
                )
            yield batch

    with manager as iterator:
        yield prepared(iterator)
