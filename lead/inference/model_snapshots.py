"""Selected, exact model inputs and raw outputs for frozen-checkpoint replay."""

import hashlib
import json
import math
import os
import platform
import random
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

NAVIGATION_INPUT_KEYS = (
    "target_point_previous",
    "target_point",
    "target_point_next",
    "command",
    "next_command",
)
RAW_PREDICTION_KEYS = (
    "pred_route",
    "pred_future_waypoints",
    "pred_headings",
    "pred_target_speed_distribution",
    "pred_target_speed_scalar",
)


def validate_snapshot_steps(
    steps,
    *,
    diagnostics_enabled: bool,
    save_path,
) -> frozenset:
    """Validate explicit selections without changing or silently coercing them."""
    if not isinstance(steps, list | tuple):
        raise ValueError("diagnostic_snapshot_steps must be a list of integers")
    if any(type(step) is not int or step < 0 for step in steps):
        raise ValueError("diagnostic_snapshot_steps must contain nonnegative integers")
    if len(steps) != len(set(steps)):
        raise ValueError("diagnostic_snapshot_steps must not contain duplicates")
    if steps and (not diagnostics_enabled or save_path is None):
        raise ValueError("Model snapshots require driving diagnostics and SAVE_PATH")
    return frozenset(steps)


def cpu_snapshot_value(value):
    """Clone tensors and convert other leaves to weights-only-loadable values."""
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", copy=True)
    if isinstance(value, np.ndarray):
        return cpu_snapshot_value(value.tolist())
    if isinstance(value, np.generic):
        return cpu_snapshot_value(value.item())
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("Snapshot dictionary keys must be strings")
        return {key: cpu_snapshot_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_snapshot_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_snapshot_value(item) for item in value)
    if isinstance(value, Path | torch.device | torch.dtype):
        return str(value)
    if value is None or type(value) in (str, bool, int, float):
        return value
    raise TypeError(f"Unsupported snapshot value: {type(value).__name__}")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_metadata(
    checkpoint_directory,
    training_config,
    closed_loop_config,
) -> dict:
    """Capture effective configuration and exact local checkpoint/source identity."""
    directory = Path(checkpoint_directory).absolute()

    def identity(path):
        return {
            "name": path.name,
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
        }

    training_values = {}
    for key, value in training_config.training_dict().items():
        if key.startswith("_"):
            continue
        try:
            # Use the effective property value, including config overrides.
            value = getattr(training_config, key, value)
            value = json.loads(json.dumps(value))
        except (TypeError, ValueError):
            continue
        training_values[key] = value
    project = Path(__file__).resolve().parents[2]
    source_paths = sorted((project / "lead").rglob("*.py"))
    source_paths.append(project / "scripts/diagnose_turns.py")
    return {
        "checkpoint_directory": str(directory),
        "model_files": [
            identity(path) for path in sorted(directory.glob("model*.pth"))
        ],
        "config_file": identity(directory / "config.json"),
        "training_config": training_values,
        "inference_config": {
            key: getattr(closed_loop_config, key)
            for key in (
                "adapt_history_mode",
                "navigation_position_source",
                "navigation_pop_distance_mode",
                "use_kalman_filter",
                "route_planner_min_distance",
                "sensor_agent_pop_distance_adaptive",
                "sensor_agent_skip_distant_target_point",
                "sensor_agent_skip_distant_target_point_threshold",
                "brake_threshold",
                "lower_target_speed",
                "lower_target_speed_factor",
                "steer_modality",
                "throttle_modality",
                "brake_modality",
            )
        },
        "source_files": {
            str(path.relative_to(project)): file_sha256(path)
            for path in source_paths
            if path.is_file()
        },
        "prediction_semantics": {
            "model_predictions": "Raw per-model outputs before ensembling and controls",
            "pred_target_speed_distribution": "Logits, not probabilities",
            "timestamp_seconds": "Evaluator elapsed game time; first initialization step has no forward",
        },
    }


def runtime_state(
    device: torch.device,
    *,
    autocast_enabled: bool,
    autocast_dtype,
) -> dict:
    """Settings used by OpenLoopInference.forward, captured outside its autocast scope."""
    return {
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else "cpu",
        "device_capability": list(torch.cuda.get_device_capability(device))
        if device.type == "cuda"
        else None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "autocast_enabled": bool(autocast_enabled),
        "autocast_dtype": str(autocast_dtype),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_enabled": torch.backends.cudnn.enabled,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_allow_fp16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        "cuda_allow_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        "flash_sdp_enabled": torch.backends.cuda.flash_sdp_enabled(),
        "mem_efficient_sdp_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
        "math_sdp_enabled": torch.backends.cuda.math_sdp_enabled(),
        "cudnn_sdp_enabled": torch.backends.cuda.cudnn_sdp_enabled(),
        "num_threads": torch.get_num_threads(),
        "num_interop_threads": torch.get_num_interop_threads(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def rng_state(device: torch.device) -> dict:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "keys": numpy_state[1].tolist(),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": torch.cuda.get_rng_state(device).cpu().clone()
        if device.type == "cuda"
        else None,
        "cuda_device_index": (
            device.index if device.index is not None else torch.cuda.current_device()
        )
        if device.type == "cuda"
        else None,
    }


class ModelSnapshots:
    """Publish complete selected frames exclusively, then append their SHA index."""

    def __init__(
        self,
        directory,
        steps,
        *,
        metadata,
        device,
        autocast_enabled,
        autocast_dtype,
    ):
        self.steps = validate_snapshot_steps(
            steps,
            diagnostics_enabled=True,
            save_path=directory,
        )
        self.metadata = cpu_snapshot_value(metadata)
        self.device = torch.device(device)
        self.directory = Path(directory) / "model_snapshots"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.index_path = self.directory / "index.jsonl"
        self._index = self.index_path.open("x", encoding="utf-8")
        self.autocast_enabled = autocast_enabled
        self.autocast_dtype = autocast_dtype

    def selected(self, step: int) -> bool:
        return step in self.steps

    def prepare(
        self,
        *,
        step: int,
        timestamp_seconds: float,
        inputs: dict,
        navigation_alternatives: dict,
    ):
        if not self.selected(step):
            return None
        if not math.isfinite(timestamp_seconds):
            raise ValueError("Snapshot timestamp must be finite")
        path = self.directory / f"{step:05d}.pth"
        if path.exists():
            raise FileExistsError(f"Refusing to replace model snapshot: {path}")
        copied_inputs = cpu_snapshot_value(inputs)
        alternatives = cpu_snapshot_value(navigation_alternatives)
        baseline = alternatives.get("baseline", {}).get("inputs", {})
        if set(baseline) != set(NAVIGATION_INPUT_KEYS):
            raise ValueError(
                "Snapshot baseline must contain exactly the five navigation inputs",
            )
        for key in NAVIGATION_INPUT_KEYS:
            if (
                not isinstance(baseline[key], torch.Tensor)
                or baseline[key].dtype != copied_inputs[key].dtype
                or baseline[key].shape != copied_inputs[key].shape
                or not torch.equal(baseline[key], copied_inputs[key])
            ):
                raise ValueError(
                    f"Snapshot baseline differs from actual model input: {key}",
                )
        result = {
            "schema_version": 1,
            "step": step,
            "timestamp_seconds": float(timestamp_seconds),
            "inputs": copied_inputs,
            "navigation_alternatives": alternatives,
            "metadata": self.metadata,
            "runtime": runtime_state(
                self.device,
                autocast_enabled=self.autocast_enabled,
                autocast_dtype=self.autocast_dtype,
            ),
        }
        # Last operation before returning to the forward: capture, never reseed.
        result["rng_state"] = rng_state(self.device)
        return result

    @contextmanager
    def capture_decoder_features(self, snapshot, nets):
        """Capture actual ADAPT sensor inputs without replacing forward arguments."""
        if snapshot is None:
            yield
            return
        captured = [None for _ in nets]
        handles = []

        def hook_for(index):
            def capture(module, args, kwargs):
                if captured[index] is not None:
                    raise RuntimeError(
                        "ADAPT decoder ran more than once in a selected forward",
                    )
                names = ("bev_features", "radar_features", "radar_predictions")
                captured[index] = {
                    key: cpu_snapshot_value(
                        kwargs[key] if key in kwargs else args[position],
                    )
                    for position, key in enumerate(names)
                }
                captured[index]["rng_state"] = rng_state(self.device)

            return capture

        try:
            for index, net in enumerate(nets):
                decoder = getattr(net, "adapt_decoder", None)
                if decoder is not None:
                    handles.append(
                        decoder.register_forward_pre_hook(
                            hook_for(index),
                            with_kwargs=True,
                        ),
                    )
            snapshot["decoder_sensor_features"] = captured
            yield
        finally:
            for handle in handles:
                handle.remove()

    def finish(self, snapshot, model_predictions) -> Path | None:
        if snapshot is None:
            return None
        if self._index.closed:
            raise ValueError("Cannot publish a snapshot after its index is closed")
        snapshot["model_predictions"] = [
            {
                key: cpu_snapshot_value(getattr(prediction, key, None))
                for key in RAW_PREDICTION_KEYS
            }
            for prediction in model_predictions
        ]
        if not snapshot["model_predictions"]:
            raise ValueError("A model snapshot requires raw per-model predictions")
        path = self.directory / f"{snapshot['step']:05d}.pth"
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=self.directory,
                prefix=".snapshot-",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                torch.save(snapshot, temporary)
                temporary.flush()
                os.fsync(temporary.fileno())
            entry = {
                "schema_version": 1,
                "step": snapshot["step"],
                "timestamp_seconds": snapshot["timestamp_seconds"],
                "file": path.name,
                "sha256": file_sha256(temporary_path),
                "size_bytes": temporary_path.stat().st_size,
            }
            # Same-filesystem hard linking atomically publishes a complete file
            # and fails if it already exists. os.replace would overwrite evidence.
            os.link(temporary_path, path)
            self._index.write(json.dumps(entry, allow_nan=False) + "\n")
            self._index.flush()
            os.fsync(self._index.fileno())
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        return path

    def close(self):
        self._index.close()
