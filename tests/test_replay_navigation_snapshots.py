"""Replay rejects drift before interpreting navigation-only model differences."""

import json
import random
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lead.inference.model_snapshots import rng_state
from scripts import replay_navigation_snapshots as replay


def fake_forward(inputs):
    point = inputs["target_point"] + inputs["rgb"].mean()
    route = torch.stack((point, point * 2), dim=1)
    return [
        SimpleNamespace(
            pred_route=route,
            pred_future_waypoints=route * 2,
            pred_headings=route[..., 1],
            pred_target_speed_distribution=torch.cat((point, point[:, :1] * 0), dim=-1),
            pred_target_speed_scalar=point.sum(dim=-1),
        ),
    ]


def fake_decode(predictions):
    prediction = predictions[0]
    return (
        prediction.pred_route,
        prediction.pred_future_waypoints,
        prediction.pred_target_speed_scalar.reshape(1, 1),
        prediction.pred_target_speed_distribution.softmax(dim=-1),
        prediction.pred_headings,
    )


def snapshot():
    inputs = {
        "rgb": torch.ones(1, 3, 2, 2),
        "rasterized_lidar": torch.arange(4).reshape(1, 1, 2, 2),
        "radar": torch.ones(1, 2, 3),
        "past_positions": torch.arange(10).reshape(1, 5, 2).float(),
        "past_yaws": torch.zeros(1, 5),
        "speed": torch.tensor([[5.0]]),
        "town": [12],
        "new_sensor_unknown_to_script": {"samples": [torch.tensor([7.0])]},
        "target_point_previous": torch.tensor([[0.0, 0.0]]),
        "target_point": torch.tensor([[10.0, -10.0]]),
        "target_point_next": torch.tensor([[20.0, -10.0]]),
        "command": torch.tensor([[1.0, 0, 0, 0, 0, 0]]),
        "next_command": torch.tensor([[0.0, 0, 0, 1, 0, 0]]),
    }
    navigation = {key: deepcopy(inputs[key]) for key in replay.NAVIGATION_INPUT_KEYS}
    alternatives = {
        name: {"inputs": deepcopy(navigation), "navigation": {"name": name}}
        for name in replay.ALTERNATIVES
    }
    alternatives["localized_navigation"]["inputs"]["target_point"][0, 1] += 1
    alternatives["training_pop_distance"]["inputs"]["target_point"][0, 0] += 2
    return {
        "schema_version": 1,
        "step": 309,
        "timestamp_seconds": 15.5,
        "inputs": inputs,
        "navigation_alternatives": alternatives,
        "model_predictions": replay.prediction_values(fake_forward(inputs)),
        "rng_state": rng_state(torch.device("cpu")),
    }


def execute(capture, **overrides):
    arguments = {
        "forward": fake_forward,
        "decode": fake_decode,
        "weight_hash": lambda: "frozen-weights",
        "restore_rng": lambda state: None,
    }
    arguments.update(overrides)
    return replay.replay_snapshot(capture, **arguments)


def test_exact_replay_changes_only_navigation_and_restores_baseline():
    capture = snapshot()
    original = replay.content_hash(capture)
    calls = []

    def forward(inputs):
        calls.append(replay.input_identity(inputs))
        return fake_forward(inputs)

    report = execute(capture, forward=forward)
    assert report["status"] == "accepted_exact"
    assert report["mode"] == "full_model"
    assert report["full_sensor_replay_status"] == "accepted_exact"
    assert report["timestamp_seconds"] == 15.5
    assert len(calls) == 4
    assert calls[0] == calls[-1]
    assert "new_sensor_unknown_to_script" in report["preserved_inputs"]["fields"]
    for key in report["preserved_inputs"]["fields"]:
        assert len({call["fields"][key]["sha256"] for call in calls}) == 1
    assert report["checks"]["captured_baseline"]["bitwise_equal"]
    assert report["checks"]["restored_baseline"]["bitwise_equal"]
    delta = report["variants"]["localized_navigation"]["delta_from_baseline"]
    assert delta["route"]["max_point_distance_m"] == 2
    assert delta["future_waypoints"]["max_point_distance_m"] == 4
    assert replay.content_hash(capture) == original


def test_baseline_mismatch_prevents_every_alternative_forward():
    capture = snapshot()
    capture["model_predictions"][0]["pred_route"][0, 0, 0] += 0.125
    calls = []

    def forward(inputs):
        calls.append(1)
        return fake_forward(inputs)

    report = execute(capture, forward=forward)
    assert report["status"] == "rejected"
    assert report["variants"] == {}
    assert calls == [1]
    comparison = report["checks"]["captured_baseline"]["models"][0]["pred_route"]
    assert comparison["max_abs_error"] == 0.125
    assert comparison["mean_abs_error"] == 0.125 / 4


def test_explicit_tolerance_is_recorded_without_claiming_bitwise_parity():
    capture = snapshot()
    capture["model_predictions"][0]["pred_route"][0, 0, 0] += 0.125
    report = execute(capture, atol=0.125)
    assert report["status"] == "accepted_with_explicit_tolerance"
    assert report["tolerance"] == {"atol": 0.125, "rtol": 0.0}
    assert not report["checks"]["captured_baseline"]["bitwise_equal"]


def test_restored_baseline_drift_withholds_all_variant_results():
    count = 0

    def forward(inputs):
        nonlocal count
        count += 1
        prediction = fake_forward(inputs)
        if count == 4:
            prediction[0].pred_route += 1
        return prediction

    report = execute(snapshot(), forward=forward)
    assert report["status"] == "rejected"
    assert report["checks"]["captured_baseline"]["accepted"]
    assert not report["checks"]["restored_baseline"]["accepted"]
    assert report["variants"] == {}


def test_input_mutation_is_rejected_even_for_an_unrecognized_nested_sensor():
    def forward(inputs):
        inputs["new_sensor_unknown_to_script"]["samples"][0].add_(1)
        return fake_forward(inputs)

    with pytest.raises(ValueError, match="mutated an input"):
        execute(snapshot(), forward=forward)


def test_model_weight_or_buffer_mutation_is_rejected():
    calls = iter(["before", "after"])
    with pytest.raises(ValueError, match="mutated weights or buffers"):
        execute(snapshot(), weight_hash=lambda: next(calls))


@pytest.mark.parametrize("field", ["rgb", "past_positions", "new_setting"])
def test_alternative_cannot_change_sensory_or_history_fields(field):
    capture = snapshot()
    capture["navigation_alternatives"]["localized_navigation"]["inputs"][field] = (
        torch.zeros(1, 2)
    )
    with pytest.raises(ValueError, match="exactly the five"):
        execute(capture)


def test_captured_baseline_must_match_actual_navigation_tensors():
    capture = snapshot()
    capture["navigation_alternatives"]["baseline"]["inputs"]["target_point"] += 1
    with pytest.raises(ValueError, match="Captured baseline navigation differs"):
        execute(capture)


@pytest.mark.parametrize(
    "invalid",
    [torch.ones(2), torch.ones(1, 2).double(), torch.tensor([[float("nan"), 1]])],
)
def test_alternative_requires_finite_batched_fp32_navigation(invalid):
    capture = snapshot()
    capture["navigation_alternatives"]["localized_navigation"]["inputs"][
        "target_point"
    ] = invalid
    with pytest.raises(ValueError, match="Invalid batched fp32"):
        execute(capture)


def test_hash_preserves_bfloat16_dtype_shape_signed_zero_and_all_leaves():
    assert replay.content_hash(
        torch.ones(2, dtype=torch.bfloat16),
    ) != replay.content_hash(torch.ones(2))
    assert replay.content_hash(torch.ones(2)) != replay.content_hash(torch.ones(1, 2))
    assert replay.content_hash(torch.tensor([0.0])) != replay.content_hash(
        torch.tensor([-0.0]),
    )
    first = {"unknown": [torch.tensor(1, dtype=torch.bfloat16)], "town": [12]}
    second = deepcopy(first)
    second["unknown"][0].add_(1)
    assert replay.content_hash(first) != replay.content_hash(second)
    assert replay.content_hash({"a": 1, "b": 2}) == replay.content_hash(
        {"b": 2, "a": 1},
    )


def test_signed_zero_is_not_accepted_as_bitwise_equal_with_default_tolerance():
    result = replay.compare_tensor(torch.tensor([0.0]), torch.tensor([-0.0]))
    assert result["max_abs_error"] == 0
    assert not result["accepted"]
    assert not result["bitwise_equal"]


def test_mismatched_shapes_dtypes_nonfinite_and_missing_predictions_are_rejected():
    assert not replay.compare_tensor(torch.ones(1), torch.ones(1, 1))["accepted"]
    assert not replay.compare_tensor(torch.ones(1), torch.ones(1).double(), atol=1)[
        "accepted"
    ]
    assert not replay.compare_tensor(
        torch.tensor([float("nan")]),
        torch.tensor([float("nan")]),
    )["accepted"]
    assert not replay.compare_tensor(None, torch.ones(1))["accepted"]
    assert replay.compare_tensor(None, None)["accepted"]


@pytest.mark.parametrize("tolerance", [-1, float("inf"), float("nan")])
def test_invalid_tolerances_are_rejected(tolerance):
    with pytest.raises(ValueError, match="Tolerances"):
        execute(snapshot(), atol=tolerance)


def test_safe_serialized_roundtrip_keeps_all_inputs_and_rng(tmp_path):
    capture = snapshot()
    path = tmp_path / "snapshot.pth"
    torch.save(capture, path)
    loaded = torch.load(path, weights_only=True, map_location="cpu")
    assert replay.content_hash(capture) == replay.content_hash(loaded)
    assert execute(loaded)["status"] == "accepted_exact"


def test_restores_python_numpy_torch_rng_without_cuda():
    before = rng_state(torch.device("cpu"))
    expected = random.random(), np.random.rand(), torch.rand(3)
    replay.restore_rng_state(before, "cpu")
    assert random.random() == expected[0]
    assert np.random.rand() == expected[1]
    assert torch.equal(torch.rand(3), expected[2])


def fixture_manifest(tmp_path):
    project = tmp_path / "project"
    (project / "lead").mkdir(parents=True)
    (project / "scripts").mkdir()
    (project / "lead/model.py").write_text("model = 1\n")
    (project / "scripts/diagnose_turns.py").write_text("runner = 1\n")
    experiment = tmp_path / "capture"
    checkpoint = experiment / "checkpoint"
    checkpoint.mkdir(parents=True)
    (checkpoint / "model_selected.pth").write_bytes(b"fixed-weights")
    (checkpoint / "config.json").write_text("{}\n")
    codebook = project / "codebook.npy"
    codebook.write_bytes(b"fixed-codebook")
    manifest = {
        "schema_version": 1,
        "checkpoint": {"sha256": replay.file_sha256(checkpoint / "model_selected.pth")},
        "training_config": {"sha256": replay.file_sha256(checkpoint / "config.json")},
        "source": {
            "files": {
                str(path.relative_to(project)): replay.file_sha256(path)
                for path in project.rglob("*.py")
            },
        },
        "codebooks": [{"path": str(codebook), "sha256": replay.file_sha256(codebook)}],
    }
    path = experiment / "manifest.json"
    path.write_text(json.dumps(manifest))
    return project, checkpoint, path, manifest


@pytest.mark.parametrize(
    "changed",
    ["checkpoint", "config", "source", "codebook", "new_source", "extra_model"],
)
def test_identity_validation_rejects_weight_config_codebook_and_source_drift(
    tmp_path,
    changed,
):
    project, checkpoint, path, _ = fixture_manifest(tmp_path)
    assert replay.validate_manifest(path, project=project)[1] == checkpoint
    targets = {
        "checkpoint": checkpoint / "model_selected.pth",
        "config": checkpoint / "config.json",
        "source": project / "lead/model.py",
        "codebook": project / "codebook.npy",
        "new_source": project / "lead/extra.py",
        "extra_model": checkpoint / "model_extra.pth",
    }
    targets[changed].write_text("changed")
    with pytest.raises(ValueError):
        replay.validate_manifest(path, project=project)


def test_snapshot_index_detects_changed_snapshot_and_truncated_index(tmp_path):
    capture = snapshot()
    path = tmp_path / "00309.pth"
    torch.save(capture, path)
    entry = {
        "schema_version": 1,
        "size_bytes": path.stat().st_size,
        "file": path.name,
        "step": 309,
        "timestamp_seconds": 15.5,
        "sha256": replay.file_sha256(path),
    }
    index = tmp_path / "index.jsonl"
    index.write_text(json.dumps(entry) + "\n")
    assert replay.validate_snapshot_index(path, capture) == entry
    index.write_text(json.dumps(entry) + '\n{"truncated"')
    with pytest.raises(ValueError, match="Malformed snapshot index"):
        replay.validate_snapshot_index(path, capture)
    index.write_text(json.dumps(entry) + "\n")
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="schema/size|SHA-256 mismatch"):
        replay.validate_snapshot_index(path, capture)


def test_nonfinite_capture_is_rejected_and_keeps_valid_json_failure_evidence(tmp_path):
    capture = snapshot()
    capture["model_predictions"][0]["pred_route"][0, 0, 0] = float("nan")
    report = execute(capture)
    replay.write_json(tmp_path / "report.json", report)
    result = json.loads((tmp_path / "report.json").read_text())
    assert result["status"] == "rejected"
    assert not result["checks"]["captured_baseline"]["models"][0]["pred_route"][
        "finite"
    ]
    assert result["captured_raw_predictions"][0]["pred_route"][0][0][0] is None
    assert result["variants"] == {}


def test_output_directory_is_exclusive(tmp_path):
    args = SimpleNamespace(output=tmp_path)
    marker = tmp_path / "existing.txt"
    marker.write_text("keep")
    with pytest.raises(FileExistsError):
        replay.run(args)
    assert marker.read_text() == "keep"


@pytest.mark.parametrize(
    "continue_on_mismatch, expected_reports",
    [(False, 1), (True, 2)],
)
def test_continue_on_mismatch_keeps_overall_rejection(
    tmp_path,
    monkeypatch,
    continue_on_mismatch,
    expected_reports,
):
    _, checkpoint, manifest_path, manifest = fixture_manifest(tmp_path)
    paths = []
    for step in (309, 310):
        capture = snapshot()
        capture.update(step=step, metadata={"inference_config": {}}, runtime={})
        path = tmp_path / f"{step}.pth"
        torch.save(capture, path)
        paths.append(str(path))
    inference = SimpleNamespace(config_training=None, ensemble_planning_decoder=None)
    monkeypatch.setattr(
        replay,
        "validate_manifest",
        lambda path: (manifest, checkpoint),
    )
    monkeypatch.setattr(replay, "validate_capture_identity", lambda *args: None)
    monkeypatch.setattr(replay, "validate_snapshot_index", lambda *args: {})
    monkeypatch.setattr(replay, "configure_runtime", lambda *args: None)
    monkeypatch.setattr(replay, "create_inference", lambda *args: inference)
    monkeypatch.setattr(replay.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(replay.torch.cuda, "set_device", lambda device: None)

    def fake_replay(capture, **kwargs):
        return {
            "step": capture["step"],
            "status": "rejected" if capture["step"] == 309 else "accepted_exact",
            "rejection_reason": "Captured mismatch",
        }

    monkeypatch.setattr(replay, "replay_snapshot", fake_replay)
    args = SimpleNamespace(
        output=tmp_path / "replay",
        manifest=manifest_path,
        snapshots=paths,
        device="cuda:0",
        decoder_features=False,
        cpu_threads=4,
        atol=0.0,
        rtol=0.0,
        continue_on_mismatch=continue_on_mismatch,
    )
    result = replay.run(args)
    assert result["status"] == "rejected"
    assert len(result["snapshots"]) == expected_reports
    assert result["snapshots"][0]["status"] == "rejected"
    assert (args.output / "000_step_00309.json").exists()


def test_config_isolation_rejects_overrides_and_restores_argv_cwd_env(
    tmp_path,
    monkeypatch,
):
    for key in replay.CONFIG_ENVIRONMENT:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("LEAD_PROJECT_ROOT", "original")
    monkeypatch.setattr(replay.sys, "argv", ["tool", "--output", "run"])
    previous = Path.cwd()
    with replay.isolated_config_environment(tmp_path, torch.device("cuda:0")):
        assert replay.sys.argv == ["replay_navigation_snapshots"]
        assert Path.cwd() == tmp_path
        assert replay.os.environ["LEAD_PROJECT_ROOT"] == str(tmp_path)
    assert replay.sys.argv == ["tool", "--output", "run"]
    assert replay.os.environ["LEAD_PROJECT_ROOT"] == "original"
    assert Path.cwd() == previous
    monkeypatch.setenv("LEAD_TRAINING_CONFIG", "use_radar=false")
    with pytest.raises(ValueError, match="Unset ambient LEAD_TRAINING_CONFIG"):
        with replay.isolated_config_environment(tmp_path, torch.device("cuda:0")):
            pass


def test_explicit_feature_mode_never_claims_full_model_parity():
    report = execute(snapshot(), replay_mode="decoder_features")
    assert report["status"] == "accepted_exact"
    assert report["mode"] == "decoder_features"
    assert report["full_sensor_replay_status"] == "not_run_in_this_invocation"


def test_feature_adapter_calls_real_decoder_boundary_and_keeps_kwargs_frozen():
    capture = snapshot()
    features = {
        "bev_features": torch.ones(1, 2, 2, 2),
        "radar_features": torch.ones(1, 2, 3),
        "radar_predictions": None,
        "rng_state": rng_state(torch.device("cpu")),
    }
    capture["decoder_sensor_features"] = [features]
    called = []

    def actual_decoder(*, bev_features, radar_features, radar_predictions, data, log):
        called.append(replay.content_hash((bev_features, radar_features)))
        assert radar_predictions is None
        assert log == {}
        return replay.prediction_values(fake_forward(data))[0]

    inference = SimpleNamespace(
        nets=[SimpleNamespace(adapt_decoder=actual_decoder)],
        config_training=SimpleNamespace(
            torch_float_type=torch.float32,
            use_mixed_precision_training=False,
        ),
    )
    original_hash = replay.content_hash(features)
    forward, identity = replay.decoder_features_forward(inference, capture, "cpu")
    report = execute(capture, forward=forward, replay_mode="decoder_features")
    assert report["status"] == "accepted_exact"
    assert len(called) == 4
    assert len(set(called)) == 1
    assert replay.content_hash(features) == original_hash
    assert identity["sha256"] == replay.content_hash([features])
    assert identity["models"][0]["fields"]["radar_predictions"]["type"] == "NoneType"


def test_feature_adapter_rejects_decoder_feature_mutation():
    capture = snapshot()
    capture["decoder_sensor_features"] = [
        {
            "bev_features": torch.ones(1, 2, 2, 2),
            "radar_features": None,
            "radar_predictions": None,
            "rng_state": rng_state(torch.device("cpu")),
        },
    ]

    def mutating_decoder(*, bev_features, data, **kwargs):
        bev_features.add_(1)
        return replay.prediction_values(fake_forward(data))[0]

    inference = SimpleNamespace(
        nets=[SimpleNamespace(adapt_decoder=mutating_decoder)],
        config_training=SimpleNamespace(
            torch_float_type=torch.float32,
            use_mixed_precision_training=False,
        ),
    )
    forward, _ = replay.decoder_features_forward(inference, capture, "cpu")
    with pytest.raises(ValueError, match="mutated captured sensor features"):
        execute(capture, forward=forward, replay_mode="decoder_features")


def test_genuine_ensemble_decoder_applies_brake_threshold_and_speed_factor():
    from dataclasses import fields

    from lead.adapt.adapt import Prediction
    from lead.inference.open_loop_inference import OpenLoopInference

    inference = OpenLoopInference.__new__(OpenLoopInference)
    inference.device = torch.device("cpu")
    inference.config_training = SimpleNamespace(
        use_planning_decoder=False,
        use_adapt_decoder=True,
        target_speed_classes=[0.0, 2.0, 4.0],
    )
    inference.config_open_loop = SimpleNamespace(
        brake_threshold=0.9,
        lower_target_speed=False,
        lower_target_speed_factor=0.5,
    )
    values = {field.name: None for field in fields(Prediction)}
    values.update(replay.prediction_values(fake_forward(snapshot()["inputs"]))[0])
    values["pred_target_speed_distribution"] = torch.tensor([[10.0, 0.0, 0.0]])
    values["pred_target_speed_scalar"] = torch.tensor([999.0])
    prediction = Prediction(**values)
    route, waypoints, speed, probabilities, headings = (
        inference.ensemble_planning_decoder([prediction])
    )
    assert speed.item() == 0.0
    assert probabilities[0, 0] > 0.9
    assert torch.equal(route, prediction.pred_route)
    assert torch.equal(waypoints, prediction.pred_future_waypoints)
    assert torch.equal(headings, prediction.pred_headings)
    prediction.pred_target_speed_distribution = torch.tensor([[0.0, 0.0, 10.0]])
    regular_speed = inference.ensemble_planning_decoder([prediction])[2]
    inference.config_open_loop.lower_target_speed = True
    reduced_speed = inference.ensemble_planning_decoder([prediction])[2]
    assert reduced_speed.item() == regular_speed.item() * 0.5
    assert 1.99 < reduced_speed.item() <= 2.0
