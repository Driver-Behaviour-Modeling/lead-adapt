"""Exact, opt-in evidence capture without model or simulator execution."""

import hashlib
import json
import random
from enum import IntEnum
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lead.inference import model_snapshots
from lead.inference.config_closed_loop import ClosedLoopConfig
from lead.inference.model_snapshots import (
    NAVIGATION_INPUT_KEYS,
    ModelSnapshots,
    rng_state,
    snapshot_metadata,
    validate_snapshot_steps,
)


def example_inputs():
    return {
        "rgb": torch.arange(48, dtype=torch.uint8).reshape(1, 3, 4, 4),
        "lidar": torch.ones((1, 1, 4, 4), dtype=torch.bfloat16),
        "town": np.array(["Town12"]),
        "speed": torch.tensor([[8.0]]),
        "target_point_previous": torch.tensor([[-1.0, 2.0]]),
        "target_point": torch.tensor([[10.0, -3.0]]),
        "target_point_next": torch.tensor([[20.0, -10.0]]),
        "command": torch.tensor([[1.0, 0, 0, 0, 0, 0]]),
        "next_command": torch.tensor([[0.0, 0, 0, 1, 0, 0]]),
    }


def alternatives(inputs):
    baseline = {key: inputs[key].clone() for key in NAVIGATION_INPUT_KEYS}
    return {"baseline": {"inputs": baseline, "navigation": {"pop_distance_m": 5.0}}}


def writer(tmp_path, steps=(3,)):
    return ModelSnapshots(
        tmp_path,
        steps,
        metadata={"checkpoint_directory": "/fixture", "training_config": {}},
        device="cpu",
        autocast_enabled=False,
        autocast_dtype=torch.bfloat16,
    )


def prepare(recorder, inputs=None):
    inputs = example_inputs() if inputs is None else inputs
    return recorder.prepare(
        step=3,
        timestamp_seconds=0.15,
        inputs=inputs,
        navigation_alternatives=alternatives(inputs),
    )


def raw_prediction():
    return SimpleNamespace(
        pred_route=torch.tensor([[[2.0, -0.25], [4.0, -2.0]]]),
        pred_future_waypoints=torch.tensor([[[0.4, -0.1], [0.8, -0.3]]]),
        pred_headings=torch.tensor([[-0.1, -0.2]]),
        pred_target_speed_distribution=torch.tensor([[-4.0, 2.0, 0.5]]),
        pred_target_speed_scalar=None,
    )


def test_selection_defaults_and_config_roundtrip(monkeypatch):
    monkeypatch.delenv("LEAD_CLOSED_LOOP_CONFIG", raising=False)
    assert ClosedLoopConfig().diagnostic_snapshot_steps == []
    assert (
        validate_snapshot_steps([], diagnostics_enabled=False, save_path=None)
        == frozenset()
    )
    monkeypatch.setenv("LEAD_CLOSED_LOOP_CONFIG", "diagnostic_snapshot_steps=[0,3,520]")
    steps = ClosedLoopConfig().diagnostic_snapshot_steps
    assert validate_snapshot_steps(
        steps,
        diagnostics_enabled=True,
        save_path="/fixture",
    ) == {0, 3, 520}


@pytest.mark.parametrize("steps", [None, "[3]", {3}, [True], [1.0], [-1], [3, 3]])
def test_invalid_selections_are_rejected(steps):
    with pytest.raises(ValueError, match="diagnostic_snapshot_steps"):
        validate_snapshot_steps(steps, diagnostics_enabled=True, save_path="/fixture")


@pytest.mark.parametrize(
    "enabled,path",
    [(False, "/fixture"), (True, None), (False, None)],
)
def test_explicit_capture_requires_diagnostics_and_output(enabled, path):
    with pytest.raises(ValueError, match="driving diagnostics and SAVE_PATH"):
        validate_snapshot_steps([3], diagnostics_enabled=enabled, save_path=path)


def test_weights_only_roundtrip_clones_inputs_and_raw_outputs(tmp_path):
    recorder = writer(tmp_path)
    inputs = example_inputs()
    snapshot = prepare(recorder, inputs)
    original_rgb = inputs["rgb"].clone()
    inputs["rgb"].zero_()
    inputs["target_point"].add_(1000)
    prediction = raw_prediction()
    original_logits = prediction.pred_target_speed_distribution.clone()
    path = recorder.finish(snapshot, [prediction])
    prediction.pred_target_speed_distribution.zero_()
    recorder.close()

    restored = torch.load(path, weights_only=True, map_location="cpu")
    assert restored["schema_version"] == 1
    assert restored["step"] == 3
    assert restored["timestamp_seconds"] == 0.15
    assert restored["inputs"]["town"] == ["Town12"]
    assert restored["inputs"]["lidar"].dtype == torch.bfloat16
    assert restored["inputs"]["rgb"].device.type == "cpu"
    torch.testing.assert_close(restored["inputs"]["rgb"], original_rgb, rtol=0, atol=0)
    torch.testing.assert_close(
        restored["model_predictions"][0]["pred_target_speed_distribution"],
        original_logits,
        rtol=0,
        atol=0,
    )
    assert restored["model_predictions"][0]["pred_target_speed_scalar"] is None
    assert restored["runtime"]["device"] == "cpu"
    assert restored["runtime"]["autocast_dtype"] == "torch.bfloat16"
    assert restored["rng_state"]["torch_cuda"] is None
    entry = json.loads(recorder.index_path.read_text())
    assert entry == {
        "schema_version": 1,
        "step": 3,
        "timestamp_seconds": 0.15,
        "file": "00003.pth",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
    }
    assert sorted(item.name for item in recorder.directory.iterdir()) == [
        "00003.pth",
        "index.jsonl",
    ]


def test_unselected_steps_do_no_capture_work(tmp_path, monkeypatch):
    recorder = writer(tmp_path)
    monkeypatch.setattr(
        model_snapshots,
        "cpu_snapshot_value",
        lambda value: pytest.fail("unexpected clone"),
    )
    assert (
        recorder.prepare(
            step=4,
            timestamp_seconds=float("nan"),
            inputs=None,
            navigation_alternatives=None,
        )
        is None
    )
    assert recorder.finish(None, None) is None
    recorder.close()


@pytest.mark.parametrize("change", ["value", "dtype", "shape", "missing"])
def test_navigation_baseline_must_match_actual_input_exactly(tmp_path, change):
    recorder = writer(tmp_path)
    inputs = example_inputs()
    variants = alternatives(inputs)
    navigation = variants["baseline"]["inputs"]
    if change == "value":
        navigation["target_point"].add_(1)
    elif change == "dtype":
        navigation["target_point"] = navigation["target_point"].double()
    elif change == "shape":
        navigation["target_point"] = navigation["target_point"].squeeze(0)
    else:
        del navigation["next_command"]
    with pytest.raises(ValueError, match="Snapshot baseline"):
        recorder.prepare(
            step=3,
            timestamp_seconds=0.15,
            inputs=inputs,
            navigation_alternatives=variants,
        )
    assert not list(recorder.directory.glob("*.pth"))
    recorder.close()


def test_snapshot_and_index_cannot_be_overwritten(tmp_path):
    recorder = writer(tmp_path)
    first = prepare(recorder)
    second = prepare(recorder)
    path = recorder.finish(first, [raw_prediction()])
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        recorder.finish(second, [raw_prediction()])
    assert path.read_bytes() == original
    assert len(recorder.index_path.read_text().splitlines()) == 1
    with pytest.raises(FileExistsError):
        prepare(recorder)
    recorder.close()
    with pytest.raises(FileExistsError):
        writer(tmp_path)


def test_failed_serialization_leaves_no_partial_snapshot(tmp_path, monkeypatch):
    recorder = writer(tmp_path)
    snapshot = prepare(recorder)

    def fail_save(data, destination):
        destination.write(b"partial")
        raise OSError("fixture disk failure")

    monkeypatch.setattr(torch, "save", fail_save)
    with pytest.raises(OSError, match="fixture disk failure"):
        recorder.finish(snapshot, [raw_prediction()])
    recorder.close()
    assert list(recorder.directory.iterdir()) == [recorder.index_path]
    assert recorder.index_path.read_text() == ""


def test_capture_records_rng_without_consuming_it(tmp_path):
    original = rng_state(torch.device("cpu"))
    recorder = writer(tmp_path)
    snapshot = prepare(recorder)
    captured = snapshot["rng_state"]
    assert captured["python"] == original["python"]
    assert captured["numpy"] == original["numpy"]
    assert torch.equal(captured["torch_cpu"], original["torch_cpu"])
    expected = (random.random(), np.random.random(), torch.rand(3))
    random.setstate(captured["python"])
    state = captured["numpy"]
    np.random.set_state(
        (
            state["bit_generator"],
            np.array(state["keys"], dtype=np.uint32),
            state["position"],
            state["has_gauss"],
            state["cached_gaussian"],
        ),
    )
    torch.set_rng_state(captured["torch_cpu"])
    actual = (random.random(), np.random.random(), torch.rand(3))
    assert actual[0:2] == expected[0:2]
    assert torch.equal(actual[2], expected[2])
    recorder.close()


def test_metadata_canonicalizes_effective_values_and_hashes_files(
    tmp_path,
    monkeypatch,
):
    class Kind(IntEnum):
        CARLA = 1

    class FixtureTraining:
        effective = {Kind.CARLA: (2, 3)}

        def training_dict(self):
            return {"effective": "old value", "_private": 5, "dtype": torch.bfloat16}

    monkeypatch.delenv("LEAD_CLOSED_LOOP_CONFIG", raising=False)
    (tmp_path / "config.json").write_text('{"fixture":true}')
    (tmp_path / "model0039.pth").write_bytes(b"fixture checkpoint")
    metadata = snapshot_metadata(tmp_path, FixtureTraining(), ClosedLoopConfig())
    assert metadata["training_config"] == {"effective": {"1": [2, 3]}}
    assert (
        metadata["config_file"]["sha256"]
        == hashlib.sha256((tmp_path / "config.json").read_bytes()).hexdigest()
    )
    assert metadata["model_files"][0]["name"] == "model0039.pth"
    assert "lead/inference/model_snapshots.py" in metadata["source_files"]
    assert {
        "brake_threshold",
        "lower_target_speed",
        "lower_target_speed_factor",
    } <= metadata["inference_config"].keys()
    recorder = ModelSnapshots(
        tmp_path,
        [3],
        metadata=metadata,
        device="cpu",
        autocast_enabled=False,
        autocast_dtype=torch.bfloat16,
    )
    path = recorder.finish(prepare(recorder), [raw_prediction()])
    recorder.close()
    assert torch.load(path, weights_only=True)["metadata"] == metadata


@pytest.mark.parametrize("raises", [False, True])
def test_decoder_hook_clones_actual_features_and_is_always_removed(tmp_path, raises):
    class Decoder(torch.nn.Module):
        def forward(self, bev_features, radar_features, radar_predictions, data, log):
            if raises:
                raise RuntimeError("fixture decoder failure")
            return bev_features + 1

    decoder = Decoder()
    recorder = writer(tmp_path)
    snapshot = prepare(recorder)
    features = {
        "bev_features": torch.ones(1, 2, 3, 3),
        "radar_features": torch.ones(1, 2, 4),
        "radar_predictions": None,
    }
    try:
        with recorder.capture_decoder_features(
            snapshot,
            [SimpleNamespace(adapt_decoder=decoder)],
        ):
            assert len(decoder._forward_pre_hooks) == 1
            output = decoder(**features, data={}, log={})
            assert torch.equal(output, features["bev_features"] + 1)
    except RuntimeError as error:
        assert raises and str(error) == "fixture decoder failure"
    assert len(decoder._forward_pre_hooks) == 0
    captured = snapshot["decoder_sensor_features"][0]
    features["bev_features"].zero_()
    assert torch.equal(captured["bev_features"], torch.ones(1, 2, 3, 3))
    assert captured["radar_predictions"] is None
    assert captured["rng_state"]["torch_cuda"] is None
    recorder.close()


def test_unselected_decoder_capture_registers_no_hooks(tmp_path):
    decoder = torch.nn.Identity()
    recorder = writer(tmp_path)
    with recorder.capture_decoder_features(
        None,
        [SimpleNamespace(adapt_decoder=decoder)],
    ):
        assert not decoder._forward_pre_hooks
    recorder.close()
