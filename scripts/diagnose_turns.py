"""Prepare or run isolated, single-attempt Bench2Drive turn diagnostics.

Run with the project's Python environment. Without ``--run`` this only prepares
the experiment and checks imports; it never starts or connects to CARLA.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import signal
import socket
import subprocess
import sys
import time
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROUTES = ("14842", "3144", "3905", "3737", "3904", "25968")
VARIANTS = {
    "baseline": {},
    "history_aligned": {"adapt_history_mode": "training_aligned"},
    "localized_navigation": {"navigation_position_source": "planner"},
    "training_pop_distance": {"navigation_pop_distance_mode": "training"},
}
BASE_CONFIG = {
    "record_driving_diagnostics": "true",
    "adapt_history_mode": "legacy",
    "navigation_position_source": "legacy",
    "navigation_pop_distance_mode": "legacy",
    "debug_mode": "false",
}
CONFIG_ENVIRONMENTS = (
    "LEAD_TRAINING_CONFIG",
    "LEAD_OPEN_LOOP_CONFIG",
    "LEAD_CLOSED_LOOP_CONFIG",
    "LEAD_EXPERT_CONFIG",
)
THREAD_ENVIRONMENTS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def validate_snapshot_steps(steps: list[int] | None) -> None:
    if steps is None:
        return
    if any(type(step) is not int or step < 0 for step in steps):
        raise ValueError("Snapshot steps must be nonnegative integers.")
    if len(steps) != len(set(steps)):
        raise ValueError("Snapshot steps must be unique.")


def variant_config(name: str, snapshot_steps: list[int] | None = None) -> str:
    validate_snapshot_steps(snapshot_steps)
    values = BASE_CONFIG | VARIANTS[name]
    if snapshot_steps is not None:
        values["diagnostic_snapshot_steps"] = json.dumps(
            snapshot_steps, separators=(",", ":"),
        )
    return " ".join(f"{key}={value}" for key, value in values.items())


def checked_routes(route_dir: Path, route_ids: list[str]) -> list[dict]:
    if len(route_ids) != len(set(route_ids)):
        raise ValueError("Route IDs must be unique; each route gets one attempt.")
    routes = []
    for route_id in route_ids:
        if not route_id.isdigit():
            raise ValueError(f"Expected a numeric Bench2Drive route ID: {route_id!r}")
        source = route_dir / f"{route_id}.xml"
        tree = ET.parse(source)
        nodes = tree.getroot().findall("route")
        if len(nodes) != 1 or nodes[0].get("id") != route_id:
            raise ValueError(f"{source} must contain exactly route {route_id}.")
        node = nodes[0]
        for tag in ("waypoints", "scenarios", "weathers"):
            if node.find(tag) is None:
                raise ValueError(f"{source} is missing its original {tag}.")
        routes.append(
            {
                "id": route_id,
                "source": str(source.resolve()),
                "sha256": file_hash(source),
                "town": node.get("town"),
                "scenarios": [
                    n.get("type") for n in node.findall("scenarios/scenario")
                ],
                "reference_outcome": "failed_turn"
                if route_id in DEFAULT_ROUTES[:4]
                else (
                    "successful_turn"
                    if route_id in DEFAULT_ROUTES[4:]
                    else "unspecified"
                ),
            },
        )
    return routes


def source_snapshot(project: Path) -> dict:
    files = sorted((project / "lead").rglob("*.py"))
    files += [project / "scripts/diagnose_turns.py"]
    hashes = {str(p.relative_to(project)): file_hash(p) for p in files if p.is_file()}
    result = {"files": hashes}
    for name, arguments in (
        ("git_head", ["rev-parse", "HEAD"]),
        ("git_status", ["status", "--short"]),
    ):
        completed = subprocess.run(
            ["git", "-C", str(project), *arguments],
            capture_output=True,
            text=True,
            check=False,
        )
        result[name] = completed.stdout.strip() if completed.returncode == 0 else None
    return result


def prepare_experiment(
    output: Path,
    checkpoint: Path,
    route_dir: Path,
    route_ids: list[str],
    variants: list[str],
    *,
    project: Path = PROJECT_ROOT,
    seed: int = 0,
    snapshot_steps: list[int] | None = None,
) -> dict:
    """Create an exclusive new bundle; preserve input XML bytes and model weights."""
    if output.exists():
        raise FileExistsError(
            f"Refusing to reuse an existing output directory: {output}",
        )
    if not variants or len(variants) != len(set(variants)):
        raise ValueError("Choose at least one variant, without duplicates.")
    for variant in variants:
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant: {variant}")
    validate_snapshot_steps(snapshot_steps)
    routes = checked_routes(route_dir, route_ids)
    if not routes:
        raise ValueError("Choose at least one route.")
    checkpoint = checkpoint.resolve(strict=True)
    config_path = checkpoint.parent / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    codebooks = []
    for name, value in config.items():
        if isinstance(value, str) and ("vocab_path" in name or "codebook_path" in name):
            path = Path(value)
            if not path.is_absolute():
                path = project / path
            if not path.is_file():
                raise FileNotFoundError(
                    f"Configured codebook {name} is missing: {path}",
                )
            codebooks.append(
                {
                    "config_key": name,
                    "path": str(path.resolve()),
                    "sha256": file_hash(path),
                },
            )
    manifest = {
        "schema_version": 1,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "output": str(output.resolve()),
        "project_root": str(project.resolve()),
        "checkpoint": {"path": str(checkpoint), "sha256": file_hash(checkpoint)},
        "training_config": {"path": str(config_path), "sha256": file_hash(config_path)},
        "codebooks": codebooks,
        "source": source_snapshot(project),
        "routes": routes,
        "variants": {name: variant_config(name, snapshot_steps) for name in variants},
        "snapshot_steps": snapshot_steps,
        "seed": seed,
        "repetitions": 1,
        "attempts_per_route_variant": 1,
        "status": "prepared",
        "interpretation": "A small diagnostic subset, not an estimate of the 220-route benchmark score.",
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "routes").mkdir()
    (output / "checkpoint").mkdir()
    shutil.copyfile(config_path, output / "checkpoint/config.json")
    if (
        file_hash(output / "checkpoint/config.json")
        != manifest["training_config"]["sha256"]
    ):
        raise RuntimeError("Training config changed while the experiment was prepared.")
    # The inference loader ensembles every model*.pth in its directory. Stage
    # exactly one explicitly selected checkpoint, without copying large weights.
    (output / "checkpoint/model_selected.pth").symlink_to(checkpoint)
    for route in routes:
        target = output / "routes" / f"{route['id']}.xml"
        shutil.copyfile(route["source"], target)
        if file_hash(target) != route["sha256"]:
            raise RuntimeError(f"Route changed during copying: {route['source']}")
    write_json(output / "manifest.json", manifest)
    return manifest


def evaluation_environment(
    manifest: dict,
    gpu: str,
    carla_root: Path,
    python: str = sys.executable,
    cpu_threads: int = 4,
) -> dict[str, str]:
    env = os.environ.copy()
    project = Path(manifest["project_root"])
    for name in CONFIG_ENVIRONMENTS:
        env.pop(name, None)
    paths = [
        project,
        project / "3rd_party/Bench2Drive/leaderboard",
        project / "3rd_party/Bench2Drive/scenario_runner",
        carla_root / "PythonAPI/carla",
    ]
    env.update(
        LEAD_PROJECT_ROOT=str(project),
        PYTHONPATH=os.pathsep.join(map(str, paths)),
        SCENARIO_RUNNER_ROOT=str(paths[2]),
        LEADERBOARD_ROOT=str(paths[1]),
        CARLA_ROOT=str(carla_root),
        CUDA_VISIBLE_DEVICES=gpu,
        IS_BENCH2DRIVE="1",
        PLANNER_TYPE="only_traj",
        EVALUATION_DATASET="bench2drive",
        PYTHONUNBUFFERED="1",
        PYTHONHASHSEED=str(manifest["seed"]),
    )
    # An absolute interpreter path does not activate its conda environment.
    # Include matching executable tools (in particular ffmpeg) explicitly.
    interpreter = Path(shutil.which(python) or python).resolve()
    env["PATH"] = str(interpreter.parent) + os.pathsep + env.get("PATH", "")
    if cpu_threads > 0:
        for name in THREAD_ENVIRONMENTS:
            env[name] = str(cpu_threads)
    return env


def torch_thread_setup(cpu_threads: int) -> str:
    if cpu_threads == 0:
        return ""
    return (
        f"torch.set_num_threads({cpu_threads}); "
        f"torch.set_num_interop_threads({cpu_threads}); "
    )


def check_prerequisites(
    python: str,
    carla_root: Path,
    env: dict[str, str],
    cpu_threads: int = 4,
    snapshot_steps: list[int] | None = None,
) -> dict:
    launcher = carla_root / "CarlaUE4.sh"
    binary = carla_root / "CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping"
    missing = [str(path) for path in (launcher, binary) if not path.is_file()]
    script = (
        "import importlib.metadata, json, shutil; import torch; "
        + torch_thread_setup(cpu_threads)
        + "import carla; "
        "import leaderboard.leaderboard_evaluator; import lead.inference.sensor_agent; "
        "from lead.inference.config_closed_loop import ClosedLoopConfig; "
        "config = ClosedLoopConfig(); "
        "assert config.record_driving_diagnostics is True; "
        "assert shutil.which('ffmpeg'), 'ffmpeg is missing from the evaluation PATH'; "
        "import torch; "
        "print(json.dumps({'carla': importlib.metadata.version('carla'), "
        "'ffmpeg': shutil.which('ffmpeg'), "
        "'torch': torch.__version__, 'cuda_available': torch.cuda.is_available(), "
        "'torch_intraop_threads': torch.get_num_threads(), "
        "'torch_interop_threads': torch.get_num_interop_threads(), "
        "'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})); "
        "raise SystemExit(0 if torch.cuda.is_available() else 1)"
    )
    try:
        probe = subprocess.run(
            [python, "-c", script],
            cwd=env["LEAD_PROJECT_ROOT"],
            env=env
            | {"LEAD_CLOSED_LOOP_CONFIG": variant_config("baseline", snapshot_steps)},
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        imports = {
            "returncode": probe.returncode,
            "stdout": probe.stdout,
            "stderr": probe.stderr,
        }
    except (OSError, subprocess.TimeoutExpired) as error:
        imports = {"returncode": -1, "error": str(error)}
    return {
        "ready": not missing and imports["returncode"] == 0,
        "missing_files": missing,
        "imports": imports,
    }


def inspect_snapshot_outputs(
    run_dir: Path, requested_steps: list[int], observed_steps: set[int],
) -> dict:
    """Verify published snapshots without loading their Torch payloads."""
    directory = run_dir / "agent/model_snapshots"
    index = directory / "index.jsonl"
    errors = []
    saved = {}
    seen_steps = set()
    if index.is_file():
        try:
            with index.open() as stream:
                for line_number, line in enumerate(stream, 1):
                    try:
                        row = json.loads(line)
                        if not isinstance(row, dict) or row.get("schema_version") != 1:
                            raise ValueError(
                                "Expected a schema_version=1 snapshot index object.",
                            )
                        step = row.get("step")
                        if type(step) is not int or step < 0:
                            raise ValueError(
                                "Snapshot step must be a nonnegative integer.",
                            )
                        if step in seen_steps:
                            raise ValueError(f"Duplicate snapshot step: {step}")
                        seen_steps.add(step)
                        if step not in requested_steps:
                            raise ValueError(f"Snapshot step was not requested: {step}")
                        filename = row.get("file")
                        if filename != f"{step:05d}.pth":
                            raise ValueError(
                                f"Unexpected snapshot filename: {filename!r}",
                            )
                        path = directory / filename
                        if path.is_symlink():
                            raise ValueError(
                                "A published snapshot must not be a symbolic link.",
                            )
                        size = row.get("size_bytes")
                        if (
                            type(size) is not int
                            or size <= 0
                            or path.stat().st_size != size
                        ):
                            raise ValueError(f"Snapshot size mismatch: {filename}")
                        digest = row.get("sha256")
                        if not isinstance(digest, str) or file_hash(path) != digest:
                            raise ValueError(f"Snapshot SHA-256 mismatch: {filename}")
                        saved[step] = row | {"path": str(path)}
                    except (OSError, ValueError, TypeError) as error:
                        errors.append(f"Snapshot index line {line_number}: {error}")
        except OSError as error:
            errors.append(f"Cannot read snapshot index: {error}")
    reached = [step for step in requested_steps if step in observed_steps]
    missing = [step for step in requested_steps if step not in saved]
    missing_reached = [step for step in reached if step not in saved]
    unobserved = [step for step in requested_steps if step not in observed_steps]
    if missing_reached:
        errors.append(
            f"Requested driving steps have no verified snapshot: {missing_reached}",
        )
    return {
        "snapshot_requested_steps": requested_steps,
        "snapshot_reached_steps": reached,
        "snapshot_saved_steps": sorted(saved),
        "snapshot_missing_steps": missing,
        "snapshot_missing_reached_steps": missing_reached,
        "snapshot_unobserved_steps": unobserved,
        "snapshot_capture_complete": not missing and not errors,
        "snapshot_index_path": str(index),
        "snapshot_index_present": index.is_file(),
        "snapshot_index_sha256": file_hash(index) if index.is_file() else None,
        "snapshot_files": [saved[step] for step in sorted(saved)],
        "snapshot_errors": errors,
    }


def inspect_attempt_outputs(
    run_dir: Path, snapshot_steps: list[int] | None = None,
) -> dict:
    """Distinguish a driving failure from broken execution or empty logging."""
    errors = []
    observed_steps = set()
    result = {"has_route_result": False, "diagnostics_step_records": 0}
    try:
        records = json.loads((run_dir / "result.json").read_text())["_checkpoint"][
            "records"
        ]
        result["records"] = records
        result["has_route_result"] = (
            isinstance(records, list)
            and len(records) == 1
            and isinstance(records[0], dict)
        )
        if not result["has_route_result"]:
            errors.append("Expected exactly one official route record.")
        else:
            status = str(records[0].get("status", "")).lower()
            if (
                not status
                or status == "started"
                or any(
                    marker in status
                    for marker in (
                        "agent crashed",
                        "agent couldn't be set up",
                        "simulation crashed",
                        "sensors were invalid",
                        "rejected",
                    )
                )
            ):
                errors.append(
                    f"Official evaluator reported an execution failure: {status}",
                )
    except (OSError, ValueError, KeyError, TypeError) as error:
        errors.append(f"Cannot read official route result: {error}")
    diagnostics = run_dir / "agent/driving_diagnostics.jsonl"
    result["diagnostics_path"] = str(diagnostics)
    result["diagnostics_bytes"] = (
        diagnostics.stat().st_size if diagnostics.is_file() else 0
    )
    try:
        with diagnostics.open() as stream:
            for line in stream:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("A diagnostic record must be a JSON object.")
                if record.get("record_type") == "step":
                    result["diagnostics_step_records"] += 1
                    if type(record.get("step")) is int:
                        observed_steps.add(record["step"])
    except (OSError, ValueError) as error:
        errors.append(f"Cannot parse complete driving diagnostics: {error}")
    if result["diagnostics_step_records"] == 0:
        errors.append("No driving step was recorded (metadata alone is insufficient).")
    if snapshot_steps is not None:
        result.update(inspect_snapshot_outputs(run_dir, snapshot_steps, observed_steps))
        errors.extend(result["snapshot_errors"])
    result["output_errors"] = errors
    return result


def available_ports() -> tuple[int, int]:
    """Check CARLA's three adjacent ports and a separate traffic manager port."""
    for _ in range(200):
        port = random.SystemRandom().randrange(15000, 55000)
        held = []
        try:
            for candidate in (port, port + 1, port + 2, port + 3):
                stream = socket.socket()
                held.append(stream)
                stream.bind(("127.0.0.1", candidate))
            return port, port + 3
        except OSError:
            continue
        finally:
            for stream in held:
                stream.close()
    raise RuntimeError("Unable to find four available local ports.")


def stop_process_group(process: subprocess.Popen, grace_seconds: float = 10) -> None:
    """Only signal this runner's new session, including reparented CARLA children."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        process.poll()
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.1)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=10)


def wait_for_server(process: subprocess.Popen, port: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"CARLA exited during startup: {process.returncode}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except OSError:
            time.sleep(1)
    raise TimeoutError(f"CARLA did not open port {port} within {timeout} seconds.")


def evaluator_command(
    python: str,
    project: Path,
    run_dir: Path,
    output: Path,
    route_id: str,
    port: int,
    tm_port: int,
    seed: int,
    cpu_threads: int = 4,
) -> list[str]:
    # Seed policy-side Python, NumPy and Torch, as well as TrafficManager. The
    # official runner only seeds TrafficManager; document the extra seeds here.
    wrapper = (
        "import random, runpy, sys; import numpy as np; import torch; "
        + torch_thread_setup(cpu_threads)
        + f"random.seed({seed}); np.random.seed({seed}); torch.manual_seed({seed}); "
        "sys.argv = sys.argv[1:]; runpy.run_path(sys.argv[0], run_name='__main__')"
    )
    return [
        python,
        "-c",
        wrapper,
        str(
            project
            / "3rd_party/Bench2Drive/leaderboard/leaderboard/leaderboard_evaluator.py",
        ),
        f"--routes={output / 'routes' / (route_id + '.xml')}",
        "--track=SENSORS",
        f"--checkpoint={run_dir / 'result.json'}",
        f"--agent={project / 'lead/inference/sensor_agent.py'}",
        f"--agent-config={output / 'checkpoint'}",
        "--debug=0",
        "--resume=False",
        f"--port={port}",
        f"--traffic-manager-port={tm_port}",
        "--timeout=120",
        f"--debug-checkpoint={run_dir / 'live_results.txt'}",
        f"--traffic-manager-seed={seed}",
        "--repetitions=1",
    ]


def run_one(
    manifest: dict, variant: str, route_id: str, args: argparse.Namespace, env: dict,
) -> dict:
    output = Path(manifest["output"])
    project = Path(manifest["project_root"])
    run_dir = output / variant / route_id
    run_dir.mkdir(parents=True, exist_ok=False)
    port, tm_port = available_ports()
    env = env | {
        "LEAD_CLOSED_LOOP_CONFIG": manifest["variants"][variant],
        "SAVE_PATH": str(run_dir / "agent"),
        "BENCHMARK_ROUTE_ID": route_id,
        "EVALUATION_OUTPUT_DIR": str(run_dir),
    }
    carla_command = [
        "bash",
        str(args.carla_root / "CarlaUE4.sh"),
        f"--world-port={port}",
        "-nosound",
        f"-graphicsadapter={args.graphics_adapter}",
        "-RenderOffScreen",
    ]
    command = evaluator_command(
        args.python,
        project,
        run_dir,
        output,
        route_id,
        port,
        tm_port,
        manifest["seed"],
        args.cpu_threads,
    )
    result = {
        "variant": variant,
        "route_id": route_id,
        "attempt": 1,
        "carla_command": carla_command,
        "evaluator_command": command,
        "closed_loop_config": env["LEAD_CLOSED_LOOP_CONFIG"],
        "status": "starting",
    }
    write_json(run_dir / "attempt.json", result)
    started = time.monotonic()
    carla_process = evaluator = None
    try:
        with (
            (run_dir / "carla.log").open("x") as carla_log,
            (run_dir / "evaluator.log").open("x") as evaluator_log,
        ):
            carla_process = subprocess.Popen(
                carla_command,
                cwd=project,
                env=env,
                stdout=carla_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            wait_for_server(carla_process, port, args.boot_timeout)
            evaluator = subprocess.Popen(
                command,
                cwd=project,
                env=env,
                stdout=evaluator_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            result["returncode"] = evaluator.wait(timeout=args.route_timeout)
            result["status"] = (
                "evaluator_finished" if evaluator.returncode == 0 else "evaluator_error"
            )
    except (OSError, RuntimeError, TimeoutError, subprocess.TimeoutExpired) as error:
        result["status"] = "infrastructure_error"
        result["error"] = str(error)
    except KeyboardInterrupt:
        result["status"] = "interrupted"
        raise
    finally:
        termination_requested = False

        def defer_termination(_signal, _frame):
            nonlocal termination_requested
            termination_requested = True

        # A termination arriving during cleanup must neither leak the second
        # child process nor silently proceed to the next experiment afterward.
        handlers = {
            sig: signal.signal(sig, defer_termination)
            for sig in (signal.SIGTERM, signal.SIGINT)
        }
        try:
            for process in (evaluator, carla_process):
                if process is not None:
                    stop_process_group(process)
            result["wall_seconds"] = time.monotonic() - started
            result.update(
                inspect_attempt_outputs(run_dir, manifest.get("snapshot_steps")),
            )
            if result["status"] == "evaluator_finished" and result["output_errors"]:
                result["status"] = "infrastructure_error"
            if termination_requested:
                result["status"] = "interrupted"
            write_json(run_dir / "attempt.json", result)
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
        if termination_requested:
            raise KeyboardInterrupt
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="store_true",
        help="Start fresh CARLA processes and execute the prepared cases.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/local_training/adapt_posttrain_radar_ext2/model_0039.pth",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Must not exist; defaults to a timestamped diagnostic directory.",
    )
    parser.add_argument(
        "--route-dir",
        type=Path,
        default=PROJECT_ROOT / "data/benchmark_routes/bench2drive",
    )
    parser.add_argument("--routes", nargs="+", default=list(DEFAULT_ROUTES))
    parser.add_argument(
        "--variants", nargs="+", choices=list(VARIANTS), default=["baseline"],
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--carla-root", type=Path, default=PROJECT_ROOT / "3rd_party/CARLA_0915",
    )
    parser.add_argument(
        "--gpu",
        default="0",
        help="One CUDA device index or UUID; Vulkan adapter selection is separate.",
    )
    parser.add_argument(
        "--graphics-adapter",
        type=int,
        default=0,
        help="CARLA Vulkan adapter index; verify on this machine.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--snapshot-steps",
        nargs="+",
        type=int,
        help="Save model inputs at these unique nonnegative agent steps.",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=4,
        help="Limit Torch/BLAS/Numba pools before imports; 0 retains inherited defaults.",
    )
    parser.add_argument("--boot-timeout", type=float, default=240)
    parser.add_argument("--route-timeout", type=float, default=900)
    args = parser.parse_args(argv)
    try:
        validate_snapshot_steps(args.snapshot_steps)
    except ValueError as error:
        parser.error(str(error))
    if args.seed < 0 or args.seed >= 2**32:
        parser.error("--seed must be between 0 and 2**32 - 1.")
    if args.cpu_threads < 0:
        parser.error(
            "--cpu-threads must be nonnegative (0 retains inherited defaults).",
        )
    if args.boot_timeout <= 0 or args.route_timeout <= 0:
        parser.error("Timeouts must be positive.")
    if not args.gpu or "," in args.gpu or args.graphics_adapter < 0:
        parser.error("Choose one CUDA GPU and a nonnegative Vulkan adapter index.")
    if args.output is None:
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        args.output = (
            PROJECT_ROOT
            / "outputs/diagnostics/turns"
            / f"{stamp}_{uuid.uuid4().hex[:8]}"
        )
    args.output = args.output.resolve()
    args.carla_root = args.carla_root.resolve()
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    inherited = [key for key in CONFIG_ENVIRONMENTS if os.environ.get(key, "").strip()]
    if inherited:
        raise ValueError(
            f"Unset configuration overrides for a controlled comparison: {', '.join(inherited)}",
        )
    manifest = prepare_experiment(
        args.output,
        args.checkpoint,
        args.route_dir,
        args.routes,
        args.variants,
        seed=args.seed,
        snapshot_steps=args.snapshot_steps,
    )
    env = evaluation_environment(
        manifest, args.gpu, args.carla_root, args.python, args.cpu_threads,
    )
    manifest["execution"] = {
        "python": args.python,
        "carla_root": str(args.carla_root),
        "cuda_device": args.gpu,
        "graphics_adapter": args.graphics_adapter,
        "boot_timeout": args.boot_timeout,
        "route_timeout": args.route_timeout,
        "mode": "run" if args.run else "dry_run",
    }
    manifest["execution"].update(
        cpu_threads=args.cpu_threads,
        thread_environment={key: env.get(key) for key in THREAD_ENVIRONMENTS},
        torch_thread_policy="explicit_intraop_and_interop"
        if args.cpu_threads > 0
        else "inherited_defaults",
    )
    manifest["prerequisites"] = check_prerequisites(
        args.python, args.carla_root, env, args.cpu_threads, args.snapshot_steps,
    )
    write_json(args.output / "manifest.json", manifest)
    print(
        f"Prepared {len(args.routes)} routes x {len(args.variants)} variants at {args.output}",
        flush=True,
    )
    if not manifest["prerequisites"]["ready"]:
        print(
            "Prerequisite checks failed; see manifest.json. No simulator started.",
            flush=True,
        )
        return 2
    if not args.run:
        print(
            "Dry run complete. Add --run with a fresh output path to execute.",
            flush=True,
        )
        return 0
    manifest["status"] = "running"
    write_json(args.output / "manifest.json", manifest)
    results = []

    def terminate(_signal, _frame):
        # Unwind normally so CARLA's separate process group is cleaned up if a
        # scheduler stops this runner.
        raise KeyboardInterrupt

    previous_handler = signal.signal(signal.SIGTERM, terminate)
    try:
        for route_id in args.routes:
            for variant in args.variants:
                print(f"Running {route_id}: {variant}, single attempt", flush=True)
                result = run_one(manifest, variant, route_id, args, env)
                results.append(result)
                write_json(args.output / "summary.json", results)
                print(f"Finished {route_id}: {variant}: {result['status']}", flush=True)
                if result.get("snapshot_capture_complete") is False:
                    print(
                        "Snapshot capture incomplete: "
                        f"missing={result['snapshot_missing_steps']}; "
                        f"unobserved driving steps={result['snapshot_unobserved_steps']}",
                        flush=True,
                    )
                if (
                    result["status"] != "evaluator_finished"
                    or not result["has_route_result"]
                    or result["diagnostics_step_records"] == 0
                ):
                    manifest["status"] = "infrastructure_error"
                    print(
                        "Stopping after an execution/output error; no automatic retry.",
                        flush=True,
                    )
                    return 1
        manifest["status"] = "finished"
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
        if manifest["status"] not in ("finished", "infrastructure_error"):
            manifest["status"] = "interrupted"
        write_json(args.output / "manifest.json", manifest)
    return (
        0
        if all(
            r["has_route_result"] and r["diagnostics_step_records"] > 0 for r in results
        )
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
