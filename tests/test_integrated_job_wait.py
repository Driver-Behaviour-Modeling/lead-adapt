"""Reject PID reuse and incomplete training before automatic evaluation."""

import json
import os
import subprocess
import sys
from pathlib import Path

from scripts.after_integrated_training import live_process


def test_live_process_requires_the_original_process_start_time():
    ticks = Path(f"/proc/{os.getpid()}/stat").read_text().split()[21]
    assert live_process(os.getpid(), ticks)
    assert not live_process(os.getpid(), str(int(ticks) + 1))
    assert not live_process(-12345, ticks)


def test_terminal_incomplete_training_does_not_start_evaluation(tmp_path):
    training = tmp_path / "training"
    training.mkdir()
    (training / "launch.json").write_text(
        json.dumps({"launcher_pid": -12345, "process_start_ticks": "0"}),
    )
    (training / "config.json").write_text(json.dumps({"epochs": 5}))
    output = tmp_path / "job"
    result = subprocess.run(
        [
            sys.executable,
            str(
                Path(__file__).resolve().parents[1]
                / "scripts/after_integrated_training.py",
            ),
            "--training-dir",
            str(training),
            "--output-dir",
            str(output),
            "--partitions",
            str(tmp_path / "not-used.json"),
            "--reference-gpu",
            "unused",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2, result.stderr
    status = json.loads((output / "status.json").read_text())
    assert status["status"] == "training_incomplete"
    assert status["tasks"] == {}
