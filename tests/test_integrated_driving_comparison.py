"""Prevent success-only denominators and comparisons across changed protocols."""

import json

import pytest

from scripts.compare_integrated_driving import compare
from scripts.diagnose_turns import file_hash


def bundle(path, scores, *, seed=0, omit=None):
    (path / "checkpoint").mkdir(parents=True)
    (path / "routes").mkdir()
    weights = path / "checkpoint/model_selected.pth"
    weights.write_bytes(b"fixture weights")
    config = {
        "adapt_history_free": True,
        "adapt_waypoints_from_tokens": True,
        "adapt_decoding": "beam",
        "adapt_beam_width": 8,
        "kinematic_vocab_size": 1234,
        "kdisks_heading_weight": 2.67,
    }
    config_path = path / "checkpoint/config.json"
    config_path.write_text(json.dumps(config))
    configuration = "steer_modality=waypoints throttle_modality=waypoints brake_modality=waypoints adapt_disable_world_fusion=false adapt_disable_token_navigation=false"
    rows = []
    for route_id, score in scores.items():
        xml = path / "routes" / f"{route_id}.xml"
        xml.write_text(f"original route/weather bytes {route_id}")
        rows.append({"id": route_id, "sha256": file_hash(xml)})
        if route_id == omit:
            continue
        folder = path / "baseline" / route_id
        (folder / "agent").mkdir(parents=True)
        record = {
            "route_id": f"RouteScenario_{route_id}_rep0",
            "status": "Completed",
            "scores": {"score_composed": score},
            "infractions": {"collisions_vehicle": []},
        }
        (folder / "result.json").write_text(
            json.dumps({"_checkpoint": {"records": [record]}}),
        )
        (folder / "attempt.json").write_text(
            json.dumps(
                {
                    "attempt": 1,
                    "status": "evaluator_finished",
                    "returncode": 0,
                    "closed_loop_config": configuration,
                },
            ),
        )
        (folder / "agent/driving_diagnostics.jsonl").write_text(
            json.dumps({"record_type": "step", "step": 0}) + "\n",
        )
    manifest = {
        "routes": rows,
        "variants": {"baseline": configuration},
        "seed": seed,
        "attempts_per_route_variant": 1,
        "repetitions": 1,
        "checkpoint": {"sha256": file_hash(weights)},
        "effective_training_config": {"sha256": file_hash(config_path)},
        "codebooks": [
            {"config_key": "kdisks_vocab_path", "sha256": "fixed vocabulary"},
        ],
        "source": {"files": {"lead/adapt/adapt.py": "fixed source"}},
        "execution": {
            "cuda_device": "fixed GPU",
            "graphics_adapter": 0,
            "cpu_threads": 4,
        },
    }
    (path / "manifest.json").write_text(json.dumps(manifest))
    return path


def test_complete_paired_results_report_the_actual_mean_change(tmp_path):
    baseline = bundle(tmp_path / "before", {"1": 60, "2": 100})
    combined = bundle(tmp_path / "after", {"1": 100, "2": 100})
    result = compare(baseline, combined)
    assert result["primary_comparison_valid"]
    assert result["validated_mean_delta"] == pytest.approx(20)
    assert not result["full_220_route_comparison"]


def test_missing_route_stays_in_denominator_and_cannot_prove_a_gain(tmp_path):
    baseline = bundle(tmp_path / "before", {"1": 60, "2": 60})
    combined = bundle(tmp_path / "after", {"1": 100, "2": 100}, omit="2")
    result = compare(baseline, combined)
    assert result["combined"]["baseline"]["mean_with_fixed_denominator"] == 50
    assert result["combined"]["baseline"]["valid_routes"] == 1
    assert not result["primary_comparison_valid"]
    assert result["validated_mean_delta"] is None


def test_seed_difference_invalidates_an_apparent_improvement(tmp_path):
    baseline = bundle(tmp_path / "before", {"1": 60})
    combined = bundle(tmp_path / "after", {"1": 100}, seed=1)
    result = compare(baseline, combined)
    assert "Evaluator seeds differ" in result["protocol_issues"]
    assert result["validated_mean_delta"] is None


def test_changed_staged_xml_or_checkpoint_is_detected(tmp_path):
    baseline = bundle(tmp_path / "before", {"1": 60})
    combined = bundle(tmp_path / "after", {"1": 100})
    (combined / "routes/1.xml").write_text("different weather")
    (combined / "checkpoint/model_selected.pth").write_bytes(b"different weights")
    result = compare(baseline, combined)
    assert not result["primary_comparison_valid"]
    assert any("Staged route changed" in error for error in result["protocol_issues"])
    assert any("checkpoint changed" in error for error in result["protocol_issues"])
