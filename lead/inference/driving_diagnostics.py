"""Small append-only driving traces; no sensor images or policy-side feedback."""

import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import torch


def json_value(value):
    """Detach tensors and emit standards-compliant JSON, including missing values."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.dtype == torch.bfloat16:
            value = value.float()
        return json_value(value.numpy())
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if dataclasses.is_dataclass(value):
        return json_value(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, str | float | int | bool):
        return value
    raise TypeError(f"Unsupported diagnostic value: {type(value).__name__}")


class DrivingDiagnostics:
    """Flush every tick so a simulator failure retains the preceding decisions."""

    def __init__(self, directory: str | Path, metadata: dict):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "driving_diagnostics.jsonl"
        # Evaluation retries must use a fresh directory; never replace evidence.
        self._stream = self.path.open("x", encoding="utf-8")
        try:
            self.write(
                record_type="metadata",
                schema_version=1,
                coordinates={
                    "policy": "CARLA current ego: x forward, y right; meters/radians",
                    "states": (
                        "GPS-derived navigation frame; meters/radians; "
                        "origin may differ from CARLA world"
                    ),
                    "offline_ground_truth": "CARLA world; meters/degrees; logging only",
                    "nonfinite_values": "null",
                },
                **metadata,
            )
        except BaseException:
            self._stream.close()
            raise

    def write(self, **record):
        encoded = json.dumps(json_value(record), allow_nan=False, separators=(",", ":"))
        self._stream.write(encoded + "\n")
        self._stream.flush()

    def close(self):
        self._stream.close()


def control_values(control) -> dict:
    return {key: float(getattr(control, key)) for key in ("steer", "throttle", "brake")}


def prediction_values(prediction) -> dict:
    fields = (
        "pred_route",
        "pred_future_waypoints",
        "pred_future_headings",
        "pred_target_speed_scalar",
        "pred_target_speed_distribution",
        "route_steer",
        "target_speed_throttle",
        "target_speed_brake",
        "waypoints_steer",
        "waypoints_throttle",
        "waypoints_brake",
    )
    return {key: getattr(prediction, key, None) for key in fields}
