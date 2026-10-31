"""Exercise capture -> threshold artifact -> held-out evaluation boundaries."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts/calibrate_integrated_reference.py"
)


def capture(prefix, drive, epsilon=0.01):
    prefix.with_suffix(".json").write_text(
        json.dumps(
            {
                "members": [{"sha256": "test", "seed": 0}],
                "temperatures": [1],
                "epsilon": epsilon,
                "sigma": 0.1,
                "vocabulary_sha256": "test",
                "heading_weight": 2.67,
            },
        ),
    )
    np.savez_compressed(
        prefix.with_suffix(".npz"),
        score=np.linspace(-2, 2, 100),
        aleatoric=np.ones(100),
        epistemic=np.zeros(100),
        route=np.repeat(drive, 100),
        frame=np.arange(100),
        world_conflict_count=np.linspace(0, 1, 100),
    )


def run(*args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *map(str, args)],
        capture_output=True,
        text=True,
    )


def test_calibration_cli_rejects_overlap_and_parameter_change(tmp_path):
    cal, test, changed = [tmp_path / name for name in ("cal", "test", "changed")]
    capture(cal, "cal_drive")
    capture(test, "test_drive")
    capture(changed, "another_drive", epsilon=0.1)
    artifact = tmp_path / "nested/calibration.json"
    result = run("fit", "--features", cal, "--output", artifact, "--min-bin-size", 25)
    assert result.returncode == 0, result.stderr
    report = tmp_path / "results/evaluation.json"
    result = run(
        "evaluate",
        "--calibration",
        artifact,
        "--features",
        test,
        "--output",
        report,
    )
    assert result.returncode == 0, result.stderr
    saved = json.loads(report.read_text())
    assert "test_drive" in saved["drives"]
    assert report.with_suffix(".npz").is_file()
    result = run(
        "evaluate",
        "--calibration",
        artifact,
        "--features",
        cal,
        "--output",
        tmp_path / "overlap.json",
    )
    assert result.returncode != 0 and "overlap" in result.stderr
    result = run(
        "evaluate",
        "--calibration",
        artifact,
        "--features",
        changed,
        "--output",
        tmp_path / "changed_report.json",
    )
    assert result.returncode != 0 and "differ from calibration" in result.stderr


def test_parameter_fit_drives_cannot_fit_thresholds(tmp_path):
    prefix = tmp_path / "validation"
    capture(prefix, "fit_drive")
    meta_path = prefix.with_suffix(".json")
    metadata = json.loads(meta_path.read_text())
    metadata["parameter_fit_drives"] = ["fit_drive"]
    meta_path.write_text(json.dumps(metadata))
    result = run("fit", "--features", prefix, "--output", tmp_path / "bad.json")
    assert (
        result.returncode != 0 and "Parameter-fitting drives overlap" in result.stderr
    )
