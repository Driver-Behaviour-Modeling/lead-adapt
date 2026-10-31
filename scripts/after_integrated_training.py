"""Wait for one identified training process, then run paired diagnostics.

Never restarts training. A missing final checkpoint stops this job. Evaluation
commands preserve their own logs and return codes, including failed attempts.
Finishing this job does not establish a full-benchmark improvement.
"""

import argparse
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def live_process(pid, start_ticks):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().split()
    except FileNotFoundError:
        return False
    return fields[21] == str(start_ticks) and fields[2] != "Z"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--partitions", type=Path, required=True)
    parser.add_argument("--reference-gpu", required=True)
    parser.add_argument(
        "--driving-gpu", default="GPU-4ec20303-3c97-fc1c-9499-e5539efe3717",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    training = args.training_dir.resolve()
    launch = json.loads((training / "launch.json").read_text())
    pid, ticks = launch["launcher_pid"], launch["process_start_ticks"]
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    output = args.output_dir.resolve()
    output.mkdir(parents=True)
    state = {
        "training_dir": str(training),
        "training_pid": pid,
        "training_start_ticks": ticks,
        "tasks": {},
    }

    def save(status):
        state["status"] = status
        state["updated_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        temporary = output / "status.tmp"
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(output / "status.json")

    save("waiting_for_training")
    while live_process(pid, ticks):
        save("waiting_for_training")
        time.sleep(30)
    config = json.loads((training / "config.json").read_text())
    epoch = config["epochs"] - 1
    required = [
        training / f"{prefix}_{epoch:04d}.pth"
        for prefix in ("model", "optimizer", "scheduler", "scaler")
    ]
    required.append(training / f"gradient_steps_skipped_{epoch:04d}.txt")
    if any(not path.is_file() or path.stat().st_size == 0 for path in required):
        state["error"] = (
            "Training process is terminal, but its final completed-epoch checkpoint is missing. No training restart or evaluation was attempted."
        )
        save("training_incomplete")
        return 2
    selected = output / "selected_checkpoint"
    selected.mkdir()
    os.link(required[0], selected / "model_selected.pth")
    shutil.copy2(training / "config.json", selected / "config.json")
    state["selected_epoch"] = epoch
    state["skipped_gradient_steps"] = int(required[-1].read_text().strip())
    digest = hashlib.sha256()
    with required[0].open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    state["selected_checkpoint_sha256"] = digest.hexdigest()
    state["execution_source_sha256"] = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((root / "lead").rglob("*.py"))
        + sorted((root / "scripts").glob("*integrated*"))
        if path.is_file()
    }
    save("evaluating")
    env = os.environ.copy()
    env.update(
        PYTHONPATH=str(root),
        OMP_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="1",
        PYTHONUNBUFFERED="1",
    )
    for name in (
        "LEAD_TRAINING_CONFIG",
        "LEAD_OPEN_LOOP_CONFIG",
        "LEAD_CLOSED_LOOP_CONFIG",
    ):
        env.pop(name, None)
    python = sys.executable
    reference_output = output / "reference"
    reference_env = env | {
        "CUDA_VISIBLE_DEVICES": args.reference_gpu,
        "CHECKPOINT_DIR": str(selected),
        "DATA_ROOT": json.loads(args.partitions.read_text())["data_root"],
        "PARTITIONS": str(args.partitions.resolve()),
        "OUTPUT_DIR": str(reference_output),
        "PYTHON_BIN": python,
        "BATCH_SIZE": "4",
        "WORKERS": "4",
        "FIT_DEVICE": "cuda",
    }

    def start(name, command, process_env):
        handle = (output / f"{name}.log").open("x")
        try:
            process = subprocess.Popen(
                command,
                cwd=root,
                env=process_env,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
        finally:
            handle.close()
        state["tasks"][name] = {
            "command": command,
            "pid": process.pid,
            "status": "running",
        }
        save("evaluating")
        return process

    def finish(name, process):
        code = process.wait()
        state["tasks"][name].update(
            returncode=code, status="completed" if code == 0 else "failed",
        )
        save("evaluating")
        return code

    # The two diagnostics use different GPUs and independent output directories.
    reference = start(
        "reference", ["bash", "scripts/evaluate_integrated_reference.sh"], reference_env,
    )
    baseline = root / "outputs/local_training/adapt_fresh_histfree_v1/model_0026.pth"
    cases = (("e26_driving", baseline, False), ("combined_driving", required[0], True))
    for name, checkpoint, ablations in cases:
        command = [
            python,
            "scripts/evaluate_integrated_driving.py",
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output / name),
            "--gpu",
            args.driving_gpu,
            "--run",
        ]
        if ablations:
            command.append("--ablations")
        process = start(name, command, env)
        finish(name, process)
    if finish("reference", reference) == 0:
        command = [
            python,
            "scripts/evaluate_integrated_perturbations.py",
            "--experiment-dir",
            str(reference_output),
            "--output",
            str(reference_output / "perturbations.json"),
        ]
        finish("perturbations", start("perturbations", command, env))
    successful = all(task.get("returncode") == 0 for task in state["tasks"].values())
    save("diagnostics_completed" if successful else "diagnostics_have_failures")
    return 0 if successful else 2


if __name__ == "__main__":
    raise SystemExit(main())
