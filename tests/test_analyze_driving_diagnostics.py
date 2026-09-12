"""Offline evidence extraction must preserve geometry, timestamps and uncertainty."""

import csv
import json
import math

import pytest

from scripts import analyze_driving_diagnostics as analyzer


def step(number=10, timestamp=1.5):
    return {
        "record_type": "step",
        "step": number,
        "timestamp_seconds": timestamp,
        "observed_state": {
            "filtered_state": [100, 200, 0, 3],
            "noisy_state": [100, 200, 0, 3],
            "compass_radians": math.pi / 2,
            "speed_mps": [[3.0]],
        },
        "navigation": {
            "command_id": 1,
            "next_command_id": 1,
            "target_position_source": "filtered_state",
            "planner_position_source": "filtered_state",
            "remaining_target_points_world": [[100, 210], [90, 210]],
        },
        "model_inputs": {"target_point": [[10, 0]], "target_point_next": [[20, 0]]},
        "predictions": {
            "pred_route": [[[5, 0], [10, 0]]],
            "pred_future_waypoints": [[[2, 0], [4, 0]]],
            "pred_target_speed_scalar": [[4]],
            "route_steer": 0.1,
            "waypoints_steer": 0.1,
        },
        "controller_control": {"steer": 0.1, "throttle": 0.5, "brake": 0},
        "executed_control": {"steer": 0.1, "throttle": 0.5, "brake": 0},
    }


def trace_file(tmp_path, records):
    trace = tmp_path / "trace.jsonl"
    metadata = {"record_type": "metadata", "schema_version": 1, "route_id": "14842"}
    trace.write_text(
        "\n".join(json.dumps(record) for record in [metadata, *records]) + "\n",
    )
    return trace


def test_observed_pose_projects_route_and_targets_without_using_ground_truth(tmp_path):
    record = step()
    record["offline_ground_truth"] = {
        "position_world_m": [999, 999, 999],
        "yaw_degrees": 0,
    }
    trace = trace_file(tmp_path, [record])
    output = tmp_path / "report"
    summary = analyzer.analyze(trace, output, selected_step=10, plot=True)
    snapshot = json.loads((output / "selected_step.json").read_text())
    geometry = snapshot["geometry"]
    assert geometry["predicted_route_estimated_world_m"] == [[100, 205], [100, 210]]
    assert geometry["model_targets_estimated_world_m"]["target_point"] == [[100, 210]]
    assert geometry["sparse_planner_targets_world_m"] == [[100, 210], [90, 210]]
    assert summary["route_id"] == "14842"
    assert summary["flagged_steps"] == 0
    assert "not ground truth" in geometry["projection"]


def test_timeline_preserves_steps_timestamps_progress_and_control_intervention(
    tmp_path,
):
    first, second = step(), step(14, 2.35)
    second["observed_state"]["filtered_state"] = [103, 204, 0, 3]
    second["observed_state"]["noisy_state"] = [103, 204, 0, 3]
    second["executed_control"] = {"steer": 0.1, "throttle": 0, "brake": 1}
    output = tmp_path / "report"
    summary = analyzer.analyze(trace_file(tmp_path, [first, second]), output)
    with (output / "timeline.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert [float(row["timestamp_seconds"]) for row in rows] == [1.5, 2.35]
    assert [float(row["step"]) for row in rows] == [10, 14]
    assert summary["observed_path_length_m"] == 5
    assert summary["elapsed_timestamp_seconds"] == pytest.approx(0.85)
    assert rows[1]["flags"] == "controller_modified"
    assert float(rows[1]["executed_control_brake"]) == 1


def test_large_bearing_gap_is_an_inspection_flag_not_wrong_turn_conclusion(tmp_path):
    record = step()
    record["predictions"]["pred_route"] = [[[0, 5], [0, 10]]]
    output = tmp_path / "report"
    summary = analyzer.analyze(trace_file(tmp_path, [record]), output)
    moments = json.loads((output / "candidate_moments.json").read_text())
    assert moments["moments"][0]["flags"] == "sparse_target_bearing_gap"
    assert moments["moments"][0]["route_target_bearing_gap_degrees"] == 90
    assert summary["flag_counts"] == {"sparse_target_bearing_gap": 1}
    assert "not causal or wrong-turn verdicts" in moments["purpose"]
    assert any(
        "not an exact desired trajectory" in note for note in summary["limitations"]
    )


def test_world_heading_change_accounts_for_ego_rotation_and_angle_wrap(tmp_path):
    first, second = step(), step(11, 1.55)
    first["observed_state"]["compass_radians"] = math.radians(179)
    second["observed_state"]["compass_radians"] = math.radians(-179)
    output = tmp_path / "report"
    summary = analyzer.analyze(trace_file(tmp_path, [first, second]), output)
    assert "plan_heading_change" not in summary["flag_counts"]
    assert analyzer.angle_difference(-179, 179) == 2


def test_malformed_middle_and_truncated_tail_preserve_complete_records(tmp_path):
    trace = trace_file(tmp_path, [step()])
    with trace.open("a") as stream:
        stream.write("not json\n")
        stream.write(json.dumps(step(11, 1.55)) + "\n")
        stream.write('{"record_type":"step",')
    summary = analyzer.analyze(trace, tmp_path / "report")
    assert summary["step_records"] == 2
    assert [(issue["line"], issue["kind"]) for issue in summary["issues"]] == [
        (3, "invalid_json"),
        (5, "truncated_tail"),
    ]


def test_missing_and_nonfinite_values_remain_missing_in_csv_and_json(tmp_path):
    record = step()
    record["observed_state"] = {
        "filtered_state": [float("nan"), 2],
        "speed_mps": [[float("inf")]],
    }
    record["predictions"] = {"pred_route": [[[None, 2], [3, None]]]}
    record["controller_control"] = None
    record["model_inputs"] = None
    output = tmp_path / "report"
    summary = analyzer.analyze(trace_file(tmp_path, [record]), output, plot=True)
    assert summary["read_counts"]["nonfinite_values"] == 2
    assert summary["missing_filtered_positions"] == 1
    assert summary["max_speed_mps"] is None
    for filename in ("summary.json", "selected_step.json", "candidate_moments.json"):
        assert "NaN" not in (output / filename).read_text()
        assert "Infinity" not in (output / filename).read_text()
    with (output / "timeline.csv").open() as stream:
        row = next(csv.DictReader(stream))
    assert row["speed_mps"] == ""
    assert row["route_lookahead_bearing_degrees"] == ""
    assert (output / "trajectory.png").stat().st_size > 0
    assert (output / "timeline.png").stat().st_size > 0


def test_duplicate_steps_and_reversed_time_are_reported_without_sorting(tmp_path):
    output = tmp_path / "report"
    trace = trace_file(tmp_path, [step(10, 2), step(10, 1)])
    summary = analyzer.analyze(trace, output)
    assert {issue["kind"] for issue in summary["issues"]} == {
        "duplicate_step",
        "nonincreasing_timestamp",
    }
    assert summary["first_timestamp_seconds"] == 2
    assert summary["last_timestamp_seconds"] == 1
    with pytest.raises(ValueError, match="exactly one recorded step"):
        analyzer.analyze(trace, output, selected_step=10)


def test_empty_trace_reports_missing_metadata_and_steps(tmp_path):
    trace = tmp_path / "empty.jsonl"
    trace.write_text("")
    summary = analyzer.analyze(trace, tmp_path / "report", plot=True)
    assert summary["step_records"] == 0
    assert {issue["kind"] for issue in summary["issues"]} == {
        "missing_metadata",
        "no_step_records",
    }
    with pytest.raises(ValueError, match="found 0"):
        analyzer.analyze(trace, tmp_path / "report", selected_step=20)


def test_short_horizon_does_not_trigger_sparse_target_gap_flag(tmp_path):
    record = step()
    record["predictions"]["pred_route"] = [[[0, 0.5], [0, 1]]]
    summary = analyzer.analyze(trace_file(tmp_path, [record]), tmp_path / "report")
    assert "sparse_target_bearing_gap" not in summary["flag_counts"]


def test_missing_projection_pose_keeps_geometry_explicitly_unavailable():
    record = step()
    del record["observed_state"]["compass_radians"]
    geometry = analyzer.selected_geometry(record)
    assert geometry["predicted_route_estimated_world_m"] == []
    assert geometry["sparse_planner_targets_world_m"] == [[100, 210], [90, 210]]


def test_history_observations_preserve_startup_padding_and_requested_ages(tmp_path):
    record = step()
    record["history"] = {
        "actual_tick_ages": [10, 10, 5, 0],
        "requested_tick_ages": [15, 10, 5, 0],
        "padding_mask": [True, False, False, False],
    }
    record["interventions"] = {
        "initial_braking": True,
        "stuck_detector": 0,
        "force_move": 0,
    }
    output = tmp_path / "report"
    analyzer.analyze(trace_file(tmp_path, [record]), output)
    with (output / "timeline.csv").open() as stream:
        row = next(csv.DictReader(stream))
    assert float(row["history_oldest_tick_age"]) == 10
    assert float(row["history_requested_oldest_tick_age"]) == 15
    assert float(row["history_newest_tick_age"]) == 0
    assert row["history_padding_count"] == "1"
    assert row["initial_braking"] == "True"
