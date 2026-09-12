"""Replay captured online inputs through the complete, frozen driving model.

Navigation alternatives are interpreted only after reproducing the captured raw
prediction and a restored baseline. This is an input-sensitivity experiment,
not an offline simulation of controls, interactions, or route score.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import platform
import random
import sys
from contextlib import contextmanager
from pathlib import Path

# Set library limits before importing NumPy/Torch in the standalone CLI. Importing
# this module for tests or other tools does not change their environment.
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
if __name__ == "__main__":
    _bootstrap = argparse.ArgumentParser(add_help=False)
    _bootstrap.add_argument("--cpu-threads", type=int, default=4)
    _early_args, _ = _bootstrap.parse_known_args()
    if _early_args.cpu_threads > 0:
        for _variable in THREAD_VARIABLES:
            os.environ[_variable] = str(_early_args.cpu_threads)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from lead.inference.model_snapshots import (  # noqa: E402
    NAVIGATION_INPUT_KEYS,
    RAW_PREDICTION_KEYS,
    file_sha256,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALTERNATIVES = {"baseline", "localized_navigation", "training_pop_distance"}
DECODED_KEYS = (
    "route",
    "future_waypoints",
    "target_speed_mps",
    "target_speed_probabilities",
    "future_headings_radians",
)
CONFIG_ENVIRONMENT = (
    "LEAD_TRAINING_CONFIG",
    "LEAD_OPEN_LOOP_CONFIG",
    "LEAD_CLOSED_LOOP_CONFIG",
    "LEAD_EXPERT_CONFIG",
)


def content_hash(value) -> str:
    """Hash every value, tensor byte, dtype and shape, independent of device.

    Contiguous logical values are hashed rather than allocation/stride details;
    tensor strides are recorded separately. Length-prefixing prevents ambiguous
    concatenations. BF16 and signed zero retain their exact bits.
    """
    digest = hashlib.sha256()

    def put(data):
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)

    def visit(item):
        if isinstance(item, torch.Tensor):
            if item.layout != torch.strided:
                raise ValueError("Only dense, strided snapshot tensors are supported")
            put(b"tensor")
            put(str(item.dtype).encode())
            put(json.dumps(list(item.shape)).encode())
            data = item.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
            put(data.numpy().tobytes())
        elif isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise ValueError("Snapshot dictionary keys must be strings")
            put(b"dict")
            put(str(len(item)).encode())
            for key in sorted(item):
                put(key.encode())
                visit(item[key])
        elif isinstance(item, list | tuple):
            put(type(item).__name__.encode())
            put(str(len(item)).encode())
            for child in item:
                visit(child)
        elif item is None or type(item) in (str, int, float, bool):
            put(type(item).__name__.encode())
            put(json.dumps(item, allow_nan=False).encode())
        else:
            raise TypeError(f"Unsupported snapshot value: {type(item).__name__}")

    visit(value)
    return digest.hexdigest()


def clone_to(value, device):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device=device, copy=True)
    if isinstance(value, dict):
        return {key: clone_to(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_to(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_to(item, device) for item in value)
    return value


def json_value(value):
    if isinstance(value, torch.Tensor):
        return json_value(value.detach().cpu().tolist())
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    path.write_text(json.dumps(json_value(value), indent=2, allow_nan=False) + "\n")


def input_identity(inputs) -> dict:
    return {
        "sha256": content_hash(inputs),
        "fields": {
            key: {
                "sha256": content_hash(value),
                **(
                    {
                        "shape": list(value.shape),
                        "dtype": str(value.dtype),
                        "stride": list(value.stride()),
                    }
                    if isinstance(value, torch.Tensor)
                    else {"type": type(value).__name__}
                ),
            }
            for key, value in sorted(inputs.items())
        },
    }


def compare_tensor(reference, actual, *, atol=0.0, rtol=0.0) -> dict:
    if reference is None or actual is None:
        equal = reference is None and actual is None
        return {
            "bitwise_equal": equal,
            "accepted": equal,
            "max_abs_error": None,
            "mean_abs_error": None,
            "both_missing": equal,
        }
    if not isinstance(reference, torch.Tensor) or not isinstance(actual, torch.Tensor):
        raise ValueError("Raw predictions must be tensors or None")
    shape_equal = reference.shape == actual.shape
    dtype_equal = reference.dtype == actual.dtype
    finite = bool(torch.isfinite(reference).all() and torch.isfinite(actual).all())
    bitwise_equal = content_hash(reference) == content_hash(actual)
    result = {
        "bitwise_equal": bitwise_equal,
        "shape_equal": shape_equal,
        "dtype_equal": dtype_equal,
        "finite": finite,
        "reference_dtype": str(reference.dtype),
        "actual_dtype": str(actual.dtype),
        "reference_shape": list(reference.shape),
        "actual_shape": list(actual.shape),
        "max_abs_error": None,
        "mean_abs_error": None,
        "accepted": False,
    }
    if shape_equal and finite:
        ref, got = reference.detach().cpu().double(), actual.detach().cpu().double()
        error = (got - ref).abs()
        result["max_abs_error"] = error.max().item() if error.numel() else 0.0
        result["mean_abs_error"] = error.mean().item() if error.numel() else 0.0
        result["accepted"] = dtype_equal and (
            bitwise_equal
            if atol == rtol == 0
            else bool(torch.all(error <= atol + rtol * ref.abs()))
        )
    return result


def compare_predictions(reference, actual, *, atol=0.0, rtol=0.0) -> dict:
    if len(reference) != len(actual):
        return {
            "accepted": False,
            "bitwise_equal": False,
            "error": "Model count differs",
        }
    models = []
    for expected, observed in zip(reference, actual, strict=True):
        if set(expected) != set(RAW_PREDICTION_KEYS) or set(observed) != set(
            RAW_PREDICTION_KEYS,
        ):
            raise ValueError("Raw prediction keys do not match schema version 1")
        models.append(
            {
                key: compare_tensor(expected[key], observed[key], atol=atol, rtol=rtol)
                for key in RAW_PREDICTION_KEYS
            },
        )
    return {
        "accepted": all(
            value["accepted"] for model in models for value in model.values()
        ),
        "bitwise_equal": all(
            value["bitwise_equal"] for model in models for value in model.values()
        ),
        "models": models,
    }


def prediction_values(predictions):
    return [
        {key: clone_to(getattr(prediction, key), "cpu") for key in RAW_PREDICTION_KEYS}
        for prediction in predictions
    ]


def prediction_delta(reference, actual):
    """Coordinate-wise differences; headings are radians, not a branch verdict."""
    result = {}
    for key in DECODED_KEYS:
        ref, got = reference[key], actual[key]
        if ref is None or got is None:
            result[key] = None
            continue
        difference = got.double() - ref.double()
        item = {
            "delta": difference,
            "max_abs_delta": difference.abs().max().item(),
            "mean_abs_delta": difference.abs().mean().item(),
        }
        if key in {"route", "future_waypoints"}:
            distances = torch.linalg.vector_norm(difference, dim=-1)
            item.update(
                {
                    "point_distance_m": distances,
                    "max_point_distance_m": distances.max().item(),
                    "mean_point_distance_m": distances.mean().item(),
                },
            )
        result[key] = item
    return result


def validate_snapshot(snapshot):
    if snapshot.get("schema_version") != 1:
        raise ValueError("Unsupported snapshot schema_version")
    if type(snapshot.get("step")) is not int or snapshot["step"] < 0:
        raise ValueError("Snapshot step must be a nonnegative integer")
    if not isinstance(
        snapshot.get("timestamp_seconds"), int | float,
    ) or not math.isfinite(snapshot["timestamp_seconds"]):
        raise ValueError("Snapshot timestamp must be finite")
    inputs = snapshot["inputs"]
    if not isinstance(inputs, dict) or not inputs:
        raise ValueError("Snapshot inputs must be a nonempty dictionary")
    content_hash(inputs)  # Verify all leaves are supported, not just known fields.
    alternatives = snapshot["navigation_alternatives"]
    if set(alternatives) != ALTERNATIVES:
        raise ValueError(
            f"Expected captured navigation variants {sorted(ALTERNATIVES)}",
        )
    for name, alternative in alternatives.items():
        if set(alternative["inputs"]) != set(NAVIGATION_INPUT_KEYS):
            raise ValueError(f"{name} must replace exactly the five navigation tensors")
        for key, value in alternative["inputs"].items():
            if (
                not isinstance(value, torch.Tensor)
                or value.dtype != torch.float32
                or value.ndim != 2
                or value.shape[0] != 1
                or value.shape != inputs[key].shape
                or value.dtype != inputs[key].dtype
                or not torch.isfinite(value).all()
            ):
                raise ValueError(
                    f"Invalid batched fp32 navigation tensor: {name}.{key}",
                )
            if name == "baseline" and content_hash(value) != content_hash(inputs[key]):
                raise ValueError(f"Captured baseline navigation differs: {key}")
    if len(snapshot["model_predictions"]) != 1:
        raise ValueError("Replay requires exactly one captured model prediction")


def replay_snapshot(
    snapshot,
    *,
    forward,
    decode,
    weight_hash,
    restore_rng,
    device="cpu",
    atol=0.0,
    rtol=0.0,
    replay_mode="full_model",
):
    """Testable boundary; production callbacks run full nets and the real decoder.

    No variant forward runs if initial equivalence fails. Variant outputs are
    published only after the restored baseline and immutability checks pass.
    """
    if any(not math.isfinite(value) or value < 0 for value in (atol, rtol)):
        raise ValueError("Tolerances must be finite and nonnegative")
    if replay_mode not in {"full_model", "decoder_features"}:
        raise ValueError("Unknown replay mode")
    validate_snapshot(snapshot)
    source = snapshot["inputs"]
    original_hash = content_hash(source)
    weights_before = weight_hash()
    preserved = {
        key: value for key, value in source.items() if key not in NAVIGATION_INPUT_KEYS
    }
    preserved_hash = content_hash(preserved)
    report = {
        "step": snapshot["step"],
        "timestamp_seconds": snapshot["timestamp_seconds"],
        "mode": replay_mode,
        "full_sensor_replay_status": "pending"
        if replay_mode == "full_model"
        else "not_run_in_this_invocation",
        "status": "rejected",
        "tolerance": {"atol": atol, "rtol": rtol},
        "inputs": input_identity(source),
        "preserved_inputs": input_identity(preserved),
        "weights_and_buffers_sha256_before": weights_before,
        "captured_raw_predictions": snapshot["model_predictions"],
        "checks": {},
        "variants": {},
    }

    def run(nav=None):
        inputs = clone_to(source, device)
        if nav is not None:
            inputs.update(clone_to(nav, device))
        before = content_hash(inputs)
        preserved_now = {
            key: value
            for key, value in inputs.items()
            if key not in NAVIGATION_INPUT_KEYS
        }
        if content_hash(preserved_now) != preserved_hash:
            raise ValueError("A preserved sensory/history input changed before forward")
        restore_rng(snapshot["rng_state"])
        with torch.inference_mode():
            predictions = forward(inputs)
            raw = prediction_values(predictions)
            decoded = clone_to(
                dict(zip(DECODED_KEYS, decode(predictions), strict=True)), "cpu",
            )
        if content_hash(inputs) != before:
            raise ValueError("The model mutated an input tensor/value")
        if weight_hash() != weights_before:
            raise ValueError("The model mutated weights or buffers during replay")
        if content_hash(source) != original_hash:
            raise ValueError("The captured source inputs changed during replay")
        return raw, decoded, before

    raw, decoded, baseline_input_hash = run()
    comparison = compare_predictions(
        snapshot["model_predictions"], raw, atol=atol, rtol=rtol,
    )
    report["checks"]["captured_baseline"] = comparison
    report["replayed_baseline_raw_predictions"] = raw
    if not comparison["accepted"]:
        report["rejection_reason"] = (
            f"Unmodified {replay_mode} forward did not reproduce captured raw predictions"
        )
        if replay_mode == "full_model":
            report["full_sensor_replay_status"] = "rejected"
        return json_value(report)
    pending = {
        "baseline": {
            "inputs": {key: source[key] for key in NAVIGATION_INPUT_KEYS},
            "navigation": snapshot["navigation_alternatives"]["baseline"]["navigation"],
            "all_inputs_sha256": baseline_input_hash,
            "preserved_inputs_sha256": preserved_hash,
            "raw_predictions": raw,
            "decoded": decoded,
        },
    }
    for name in ("localized_navigation", "training_pop_distance"):
        alternative = snapshot["navigation_alternatives"][name]
        variant_raw, variant_decoded, variant_hash = run(alternative["inputs"])
        # Nonfinite predictions cannot support an interpretable delta report.
        if not compare_predictions(variant_raw, variant_raw)["accepted"]:
            raise ValueError(f"Nonfinite raw predictions in {name}")
        pending[name] = {
            **alternative,
            "all_inputs_sha256": variant_hash,
            "preserved_inputs_sha256": preserved_hash,
            "raw_predictions": variant_raw,
            "decoded": variant_decoded,
            "delta_from_baseline": prediction_delta(decoded, variant_decoded),
        }
    restored_raw, restored_decoded, restored_hash = run()
    restored = compare_predictions(raw, restored_raw, atol=atol, rtol=rtol)
    restored_captured = compare_predictions(
        snapshot["model_predictions"], restored_raw, atol=atol, rtol=rtol,
    )
    report["checks"].update(
        {
            "restored_baseline": restored,
            "restored_baseline_vs_capture": restored_captured,
            "restored_decoded_baseline": {
                key: compare_tensor(
                    decoded[key], restored_decoded[key], atol=atol, rtol=rtol,
                )
                for key in DECODED_KEYS
            },
            "restored_all_inputs_bitwise_equal": restored_hash == baseline_input_hash,
            "preserved_inputs_unchanged": True,
            "weights_and_buffers_unchanged": True,
        },
    )
    report["weights_and_buffers_sha256_after"] = weight_hash()
    if (
        not restored["accepted"]
        or not restored_captured["accepted"]
        or not all(
            item["accepted"]
            for item in report["checks"]["restored_decoded_baseline"].values()
        )
    ):
        report["rejection_reason"] = (
            "Restored baseline failed equivalence; alternatives withheld"
        )
        if replay_mode == "full_model":
            report["full_sensor_replay_status"] = "rejected"
        return json_value(report)
    report["status"] = (
        "accepted_exact"
        if all(
            item["bitwise_equal"] for item in (comparison, restored, restored_captured)
        )
        else "accepted_with_explicit_tolerance"
    )
    report["variants"] = pending
    if replay_mode == "full_model":
        report["full_sensor_replay_status"] = report["status"]
    return json_value(report)


def verify_identity(path, expected, label):
    actual = file_sha256(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch: {path}")
    return actual


def validate_manifest(manifest_path, *, project=PROJECT_ROOT):
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported manifest schema_version")
    checkpoint_directory = manifest_path.parent / "checkpoint"
    models = sorted(checkpoint_directory.glob("model*.pth"))
    if len(models) != 1:
        raise ValueError(
            "Staged checkpoint directory must contain exactly one model*.pth",
        )
    verify_identity(models[0], manifest["checkpoint"]["sha256"], "Checkpoint")
    verify_identity(
        checkpoint_directory / "config.json",
        manifest["training_config"]["sha256"],
        "Training config",
    )
    source_files = manifest["source"]["files"]
    current_paths = {
        str(path.relative_to(project)) for path in (project / "lead").rglob("*.py")
    }
    current_paths.add("scripts/diagnose_turns.py")
    if set(source_files) != current_paths:
        raise ValueError("Current source file set differs from captured manifest")
    for relative, expected in source_files.items():
        path = (project / relative).resolve()
        if not path.is_relative_to(project.resolve()):
            raise ValueError(f"Source path escapes project: {relative}")
        verify_identity(path, expected, "Source")
    for codebook in manifest["codebooks"]:
        verify_identity(Path(codebook["path"]), codebook["sha256"], "Codebook")
    return manifest, checkpoint_directory


def validate_capture_identity(snapshot, manifest, checkpoint_directory):
    metadata = snapshot["metadata"]
    if metadata["source_files"] != manifest["source"]["files"]:
        raise ValueError("Snapshot source identity differs from manifest")
    models = metadata["model_files"]
    staged = list(checkpoint_directory.glob("model*.pth"))
    if (
        len(models) != 1
        or models[0]["sha256"] != manifest["checkpoint"]["sha256"]
        or models[0]["name"] != staged[0].name
    ):
        raise ValueError("Snapshot checkpoint identity differs from staged model")
    if metadata["config_file"]["sha256"] != manifest["training_config"]["sha256"]:
        raise ValueError("Snapshot training config identity differs from manifest")


def validate_snapshot_index(path, snapshot):
    index = path.parent / "index.jsonl"
    entries = []
    for number, line in enumerate(index.read_text().splitlines(), 1):
        try:
            item = json.loads(line)
        except ValueError as error:
            raise ValueError(f"Malformed snapshot index at line {number}") from error
        if item.get("file") == path.name:
            entries.append(item)
    if len(entries) != 1:
        raise ValueError(f"Expected exactly one snapshot index entry for {path.name}")
    entry = entries[0]
    if (
        entry.get("schema_version") != 1
        or entry.get("size_bytes") != path.stat().st_size
    ):
        raise ValueError("Snapshot index schema/size differs from snapshot file")
    verify_identity(path, entry["sha256"], "Snapshot index")
    if (
        entry["step"] != snapshot["step"]
        or entry["timestamp_seconds"] != snapshot["timestamp_seconds"]
    ):
        raise ValueError("Snapshot timestamp/step differs from its index")
    return entry


def restore_rng_state(state, device):
    random.setstate(state["python"])
    value = state["numpy"]
    np.random.set_state(
        (
            value["bit_generator"],
            np.asarray(value["keys"], dtype=np.uint32),
            value["position"],
            value["has_gauss"],
            value["cached_gaussian"],
        ),
    )
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state["torch_cuda"] is not None:
        if torch.device(device).type != "cuda":
            raise ValueError("Captured CUDA RNG requires CUDA replay")
        torch.cuda.set_rng_state(state["torch_cuda"].cpu(), device)


def configure_runtime(runtime, device, cpu_threads):
    """Restore recorded numeric settings, reject unreplicated software/hardware."""
    expected_versions = {
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device_name": torch.cuda.get_device_name(device),
    }
    for key, actual in expected_versions.items():
        if runtime[key] != actual:
            raise ValueError(
                f"Runtime mismatch for {key}: capture={runtime[key]!r}, replay={actual!r}",
            )
    if (
        runtime["num_threads"] != cpu_threads
        or runtime["num_interop_threads"] != cpu_threads
    ):
        raise ValueError("--cpu-threads must match both captured Torch thread pools")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != runtime["cublas_workspace_config"]:
        raise ValueError(
            "CUBLAS_WORKSPACE_CONFIG differs; set it before starting replay",
        )
    if (
        "device_capability" in runtime
        and list(torch.cuda.get_device_capability(device))
        != runtime["device_capability"]
    ):
        raise ValueError("CUDA device capability differs from capture")
    torch.set_num_threads(cpu_threads)
    if torch.get_num_interop_threads() != cpu_threads:
        torch.set_num_interop_threads(cpu_threads)
    torch.set_float32_matmul_precision(runtime["float32_matmul_precision"])
    torch.backends.cuda.matmul.allow_tf32 = runtime["cuda_matmul_allow_tf32"]
    torch.backends.cudnn.allow_tf32 = runtime["cudnn_allow_tf32"]
    torch.backends.cudnn.benchmark = runtime["cudnn_benchmark"]
    torch.backends.cudnn.deterministic = runtime["cudnn_deterministic"]
    if "cudnn_enabled" in runtime:
        torch.backends.cudnn.enabled = runtime["cudnn_enabled"]
    torch.use_deterministic_algorithms(
        runtime["deterministic_algorithms"],
        warn_only=runtime["deterministic_warn_only"],
    )
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = runtime[
        "cuda_allow_fp16_reduced_precision_reduction"
    ]
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = runtime[
        "cuda_allow_bf16_reduced_precision_reduction"
    ]
    torch.backends.cuda.enable_flash_sdp(runtime["flash_sdp_enabled"])
    torch.backends.cuda.enable_mem_efficient_sdp(runtime["mem_efficient_sdp_enabled"])
    torch.backends.cuda.enable_math_sdp(runtime["math_sdp_enabled"])
    if "cudnn_sdp_enabled" in runtime:
        torch.backends.cuda.enable_cudnn_sdp(runtime["cudnn_sdp_enabled"])


@contextmanager
def isolated_config_environment(project, device):
    """Config constructors must not parse this CLI or ambient training overrides."""
    for key in CONFIG_ENVIRONMENT:
        if os.environ.get(key, "").strip():
            raise ValueError(f"Unset ambient {key} before snapshot replay")
    previous_argv, previous_directory = sys.argv, Path.cwd()
    previous = {key: os.environ.get(key) for key in ("LEAD_PROJECT_ROOT", "LOCAL_RANK")}
    try:
        sys.argv = ["replay_navigation_snapshots"]
        os.environ["LEAD_PROJECT_ROOT"] = str(project)
        os.environ["LOCAL_RANK"] = str(device.index or 0)
        os.chdir(project)
        yield
    finally:
        sys.argv = previous_argv
        os.chdir(previous_directory)
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def create_inference(snapshot, checkpoint_directory, device):
    # Heavy dependencies are imported only after identity validation.
    from lead.inference.config_open_loop import OpenLoopConfig
    from lead.inference.open_loop_inference import OpenLoopInference
    from lead.training.config_training import TrainingConfig

    config = TrainingConfig(
        json.loads((checkpoint_directory / "config.json").read_text()),
    )
    differing = {}
    for key, expected in snapshot["metadata"]["training_config"].items():
        actual = json.loads(json.dumps(getattr(config, key)))
        if actual != expected:
            differing[key] = {"capture": expected, "replay": actual}
    if differing:
        raise ValueError(
            f"Effective training configuration differs: {json.dumps(differing)}",
        )
    runtime = snapshot["runtime"]
    if (
        bool(config.use_mixed_precision_training) != runtime["autocast_enabled"]
        or str(config.torch_float_type) != runtime["autocast_dtype"]
    ):
        raise ValueError("Autocast configuration differs from captured runtime")
    open_config = OpenLoopConfig()
    for key in ("brake_threshold", "lower_target_speed", "lower_target_speed_factor"):
        setattr(open_config, key, snapshot["metadata"]["inference_config"][key])
    open_config.strict_weight_load = True
    inference = OpenLoopInference(
        config, open_config, str(checkpoint_directory), device,
    )
    if len(inference.nets) != 1:
        raise ValueError("Expected exactly one loaded model")
    for net in inference.nets:
        net.eval().requires_grad_(False)
    return inference


def model_state_hash(inference):
    return content_hash(
        {
            str(index): {
                "parameters": dict(net.named_parameters()),
                "buffers": dict(net.named_buffers()),
            }
            for index, net in enumerate(inference.nets)
        },
    )


def decoder_features_forward(inference, snapshot, device):
    """Call the actual ADAPT decoder with its captured pre-hook kwargs.

    This deliberately does not claim to reproduce the upstream image/LiDAR/radar
    feature computation. There is no automatic fallback to this mode.
    """
    from dataclasses import fields

    from lead.adapt.adapt import Prediction

    features = snapshot.get("decoder_sensor_features")
    keys = {"bev_features", "radar_features", "radar_predictions"}
    if not isinstance(features, list) or len(features) != len(inference.nets):
        raise ValueError("Decoder feature capture must align with every loaded model")
    for feature in features:
        if not isinstance(feature, dict) or set(feature) != keys | {"rng_state"}:
            raise ValueError("Missing captured decoder kwargs/RNG state")
    original_hash = content_hash(features)

    def forward(inputs):
        predictions = []
        for net, feature in zip(inference.nets, features, strict=True):
            kwargs = clone_to({key: feature[key] for key in keys}, device)
            before = content_hash(kwargs)
            restore_rng_state(feature["rng_state"], device)
            with torch.amp.autocast(
                device_type="cuda",
                dtype=inference.config_training.torch_float_type,
                enabled=inference.config_training.use_mixed_precision_training,
            ):
                output = net.adapt_decoder(**kwargs, data=inputs, log={})
            if (
                content_hash(kwargs) != before
                or content_hash(features) != original_hash
            ):
                raise ValueError("Decoder mutated captured sensor features")
            values = {field.name: None for field in fields(Prediction)}
            values.update({key: output[key] for key in RAW_PREDICTION_KEYS})
            predictions.append(Prediction(**values))
        return predictions

    return forward, {
        "sha256": original_hash,
        "models": [
            input_identity({key: feature[key] for key in keys}) for feature in features
        ],
    }


def expand_snapshots(patterns):
    paths = set()
    for pattern in patterns:
        matched = glob.glob(pattern)
        if not matched:
            raise ValueError(f"Snapshot pattern matched no files: {pattern}")
        paths.update(Path(path).resolve() for path in matched)
    return sorted(paths)


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    summary = {
        "schema_version": 1,
        "status": "rejected",
        "snapshots": [],
        "mode": "decoder_features" if args.decoder_features else "full_model",
        "manifest": str(args.manifest.resolve()),
        "tolerance": {"atol": args.atol, "rtol": args.rtol},
        "continue_on_mismatch": args.continue_on_mismatch,
        "limitations": [
            (
                "Actual-decoder input sensitivity using captured sensor features; sensor feature extraction is not replayed."
                if args.decoder_features
                else "Full-model input sensitivity at captured observed states."
            ),
            "No offline route score or PID controls are inferred.",
            "Alternative planners follow the recorded localization/route progression history, not a counterfactual rollout.",
            "Sparse navigation cues are not an exact desired trajectory or a causal module diagnosis.",
            "Exact output matching does not guarantee deterministic future simulator interactions.",
        ],
    }
    try:
        manifest, checkpoint_directory = validate_manifest(args.manifest.resolve())
        summary["manifest_sha256"] = file_sha256(args.manifest)
        summary["replay_script_sha256"] = file_sha256(Path(__file__))
        paths = expand_snapshots(args.snapshots)
        device = torch.device(args.device)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("The live OpenLoopInference loader requires a CUDA device")
        torch.cuda.set_device(device)
        inference = None
        first_metadata = first_runtime = None
        with isolated_config_environment(PROJECT_ROOT, device):
            for index, path in enumerate(paths):
                snapshot = torch.load(path, map_location="cpu", weights_only=True)
                validate_snapshot(snapshot)
                validate_capture_identity(snapshot, manifest, checkpoint_directory)
                index_entry = validate_snapshot_index(path, snapshot)
                if inference is None:
                    configure_runtime(snapshot["runtime"], device, args.cpu_threads)
                    inference = create_inference(snapshot, checkpoint_directory, device)
                    first_metadata, first_runtime = (
                        snapshot["metadata"],
                        snapshot["runtime"],
                    )
                    summary["capture_runtime"] = first_runtime
                    summary["inference_config"] = first_metadata["inference_config"]
                elif (
                    snapshot["metadata"] != first_metadata
                    or snapshot["runtime"] != first_runtime
                ):
                    raise ValueError(
                        "Selected snapshots must share the same capture metadata/runtime",
                    )

                def forward(inputs, inference=inference):
                    with torch.amp.autocast(
                        device_type="cuda",
                        dtype=inference.config_training.torch_float_type,
                        enabled=inference.config_training.use_mixed_precision_training,
                    ):
                        return [net(inputs) for net in inference.nets]

                feature_identity = None
                active_forward = forward
                if args.decoder_features:
                    active_forward, feature_identity = decoder_features_forward(
                        inference, snapshot, device,
                    )
                report = replay_snapshot(
                    snapshot,
                    forward=active_forward,
                    decode=inference.ensemble_planning_decoder,
                    weight_hash=lambda inference=inference: model_state_hash(inference),
                    restore_rng=lambda state: restore_rng_state(state, device),
                    device=device,
                    atol=args.atol,
                    rtol=args.rtol,
                    replay_mode=summary["mode"],
                )
                report.update(
                    {"snapshot_path": str(path), "snapshot_index": index_entry},
                )
                if feature_identity is not None:
                    report["captured_decoder_features"] = feature_identity
                filename = f"{index:03d}_step_{snapshot['step']:05d}.json"
                write_json(args.output / filename, report)
                summary["snapshots"].append(
                    {
                        "step": snapshot["step"],
                        "report": filename,
                        "status": report["status"],
                    },
                )
                print(
                    f"step {snapshot['step']} ({snapshot['timestamp_seconds']:.2f}s): {report['status']}",
                    flush=True,
                )
                if report["status"] == "rejected" and not args.continue_on_mismatch:
                    raise ValueError(report["rejection_reason"])
        # File identity is checked again after inference, including source/config/codebooks.
        validate_manifest(args.manifest.resolve())
        rejected = sum(item["status"] == "rejected" for item in summary["snapshots"])
        if rejected:
            summary["error"] = {
                "type": "PredictionMismatch",
                "message": f"{rejected} snapshot baseline checks rejected; see individual reports",
            }
        else:
            summary["status"] = "accepted"
    except Exception as error:
        summary["error"] = {"type": type(error).__name__, "message": str(error)}
    write_json(args.output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="Capture run manifest.json; uses its staged checkpoint/config",
    )
    parser.add_argument(
        "--snapshots",
        required=True,
        nargs="+",
        help="One or more snapshot paths or quoted globs",
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="New, exclusive output directory",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--decoder-features",
        action="store_true",
        help="Explicit diagnostic: run the real decoder on captured features; does not establish full-sensor parity",
    )
    parser.add_argument("--cpu-threads", default=4, type=int)
    parser.add_argument(
        "--continue-on-mismatch",
        action="store_true",
        help="Keep rejected-frame evidence and test remaining snapshots; never run alternatives for a rejected baseline",
    )
    parser.add_argument(
        "--atol",
        default=0.0,
        type=float,
        help="Explicit absolute tolerance; default requires bitwise equality",
    )
    parser.add_argument(
        "--rtol",
        default=0.0,
        type=float,
        help="Explicit relative tolerance; never increased automatically",
    )
    args = parser.parse_args()
    if args.cpu_threads < 1 or any(
        not math.isfinite(value) or value < 0 for value in (args.atol, args.rtol)
    ):
        parser.error("CPU threads must be positive; tolerances finite and nonnegative")
    try:
        summary = run(args)
    except FileExistsError:
        parser.error(f"Output already exists: {args.output}")
    print(
        json.dumps(
            {
                "status": summary["status"],
                "output": str(args.output),
                "error": summary.get("error"),
            },
        ),
        flush=True,
    )
    return 0 if summary["status"] == "accepted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
