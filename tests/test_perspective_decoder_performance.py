"""Compatibility checks for the optional cheaper perspective decoder path."""

import copy
import io

import pytest
import torch
import torch.nn.functional as F

from lead.adapt.perspective_decoder import PerspectiveDecoder as AdaptDecoder
from lead.common.constants import SourceDataset
from lead.tfv6.perspective_decoder import PerspectiveDecoder as TransfuserDecoder
from lead.training.config_training import TrainingConfig


class SmallConfig(TrainingConfig):
    """Avoid environment/CLI overrides and keep geometry tests small on CPU."""

    final_image_height = 64
    final_image_width = 96
    deconv_channel_num_0 = 8
    deconv_channel_num_1 = 6
    deconv_channel_num_2 = 4
    upsample_perspective_logits = False
    upsample_mode = "bilinear"

    def __init__(self):
        pass


class LegacyConfig(SmallConfig):
    """Simulate an older serialized configuration without the new options."""

    def __getattribute__(self, name):
        if name in ("upsample_mode", "upsample_perspective_logits"):
            raise AttributeError(name)
        return super().__getattribute__(name)


@pytest.fixture(scope="module", autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make_decoder(decoder_class, config, modality):
    return decoder_class(
        config=config,
        in_channels=8,
        out_channels=3 if modality == "semantic" else 1,
        perspective_upsample_factor=32,
        modality=modality,
        device=torch.device("cpu"),
        source_data=int(SourceDataset.CARLA),
    )


def original_forward(decoder, features):
    """Original ordering, deliberately independent of the new option branches."""
    output = decoder.deconv1(features)
    output = F.interpolate(
        output,
        scale_factor=decoder.scale_factor_0,
        mode="bilinear",
        align_corners=False,
    )
    output = decoder.deconv2(output)
    output = F.interpolate(
        output,
        scale_factor=decoder.scale_factor_1,
        mode="bilinear",
        align_corners=False,
    )
    output = decoder.deconv3(output)
    expected_size = (
        decoder.config.final_image_height,
        decoder.config.final_image_width,
    )
    if output.shape[-2:] != expected_size:
        output = F.interpolate(
            output,
            size=expected_size,
            mode="bilinear",
            align_corners=False,
        )
    return output.squeeze(1) if decoder.modality == "depth" else output


@pytest.mark.parametrize("decoder_class", [AdaptDecoder, TransfuserDecoder])
@pytest.mark.parametrize("modality", ["semantic", "depth"])
@pytest.mark.parametrize("config_class", [SmallConfig, LegacyConfig])
@pytest.mark.parametrize("repair_size", [False, True])
def test_default_matches_original_outputs_and_gradients(
    decoder_class,
    modality,
    config_class,
    repair_size,
):
    torch.manual_seed(12)
    config = config_class()
    if repair_size:
        config.final_image_height = 63
        config.final_image_width = 95
    decoder = make_decoder(decoder_class, config, modality).double()
    original = copy.deepcopy(decoder)
    features = torch.randn(2, 8, 2, 3, dtype=torch.float64, requires_grad=True)
    old_features = features.detach().clone().requires_grad_()
    actual = decoder({}, features, {})
    expected = original_forward(original, old_features)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    cotangent = torch.randn_like(actual)
    actual.backward(cotangent)
    expected.backward(cotangent)
    torch.testing.assert_close(features.grad, old_features.grad, rtol=0, atol=0)
    for new_parameter, old_parameter in zip(
        decoder.parameters(),
        original.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(
            new_parameter.grad,
            old_parameter.grad,
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize("decoder_class", [AdaptDecoder, TransfuserDecoder])
@pytest.mark.parametrize("modality", ["semantic", "depth"])
@pytest.mark.parametrize("mode", ["bilinear", "nearest"])
@pytest.mark.parametrize("upsample_logits", [False, True])
@pytest.mark.parametrize("repair_size", [False, True])
def test_options_load_legacy_weights_and_train(
    decoder_class,
    modality,
    mode,
    upsample_logits,
    repair_size,
):
    torch.manual_seed(22)
    legacy = make_decoder(decoder_class, LegacyConfig(), modality)
    checkpoint = io.BytesIO()
    torch.save(legacy.state_dict(), checkpoint)
    checkpoint.seek(0)
    config = SmallConfig()
    config.upsample_mode = mode
    config.upsample_perspective_logits = upsample_logits
    if repair_size:
        config.final_image_height = 63
        config.final_image_width = 95
    decoder = make_decoder(decoder_class, config, modality)
    state = torch.load(checkpoint, weights_only=True)
    decoder.load_state_dict(state, strict=True)
    # The pre-backport head had exactly these six convolutions and no buffers.
    legacy_keys = {
        f"deconv{block}.{layer}.{parameter}"
        for block in (1, 2, 3)
        for layer in (0, 2)
        for parameter in ("weight", "bias")
    }
    assert set(decoder.state_dict()) == legacy_keys
    features = torch.randn(2, 8, 2, 3, requires_grad=True)
    prediction = decoder({}, features, {})
    size = (2, config.final_image_height, config.final_image_width)
    if modality == "semantic":
        assert prediction.shape == (2, 3, *size[1:])
        loss = F.cross_entropy(prediction, torch.randint(3, size))
    else:
        assert prediction.shape == size
        loss = F.l1_loss(prediction, torch.rand(size))
    assert torch.isfinite(prediction).all()
    loss.backward()
    assert torch.isfinite(features.grad).all()
    assert features.grad.abs().sum() > 0
    for parameter in decoder.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


@pytest.mark.parametrize("decoder_class", [AdaptDecoder, TransfuserDecoder])
def test_invalid_resize_mode_fails_before_training(decoder_class):
    config = SmallConfig()
    config.upsample_mode = "bicubic"
    with pytest.raises(ValueError, match="bilinear.*nearest"):
        make_decoder(decoder_class, config, "semantic")


@pytest.mark.parametrize("decoder_class", [AdaptDecoder, TransfuserDecoder])
def test_large_geometry_mismatch_still_rejected(decoder_class):
    config = SmallConfig()
    config.final_image_height = 100
    config.upsample_perspective_logits = True
    decoder = make_decoder(decoder_class, config, "depth")
    with pytest.raises(ValueError, match="Output size mismatch too large"):
        decoder({}, torch.randn(1, 8, 2, 3), {})
