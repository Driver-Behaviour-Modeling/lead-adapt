"""Score actual recorded maneuvers around vehicle collisions using frozen sensors.

The reference model did not generate these older driving traces. Ground-truth
ego poses are used only to measure the observed future after each snapshot;
predicted other-vehicle futures use snapshot sensors only. Selected collision
cases and controls provide qualitative evidence, not a detection-rate benchmark
or a causal assignment of responsibility.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

from lead.adapt.transfuser_utils import patch_norm_fp32
from lead.common.constants import SourceDataset
from lead.scoring.calibration import SceneCalibration
from lead.scoring.reference import NormativeTokenScorer
from lead.scoring.world_interactions import interaction_features
from lead.training.batch_pipeline import move_batch_to_device
from lead.training.config_training import TrainingConfig
from lead.training.training_utils import create_model


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            value.update(chunk)
    return value.hexdigest()


def observed_future(rows, step, steps=8, dt=0.25):
    """Interpolate logged CARLA world poses into the snapshot's true ego frame."""
    rows = sorted(rows, key=lambda row: row["timestamp_seconds"])
    times = np.array([row["timestamp_seconds"] for row in rows])
    anchor = next(row for row in rows if row["step"] == step)
    start = anchor["timestamp_seconds"]
    target_times = start + dt * np.arange(1, steps + 1)
    if dt <= 0 or not np.isfinite(dt) or steps < 1 or np.any(np.diff(times) <= 0):
        raise ValueError(
            "Invalid prediction timing or duplicate/nonmonotonic trace times",
        )
    if target_times[-1] > times[-1] + 1e-6:
        raise ValueError("Complete observed future horizon is unavailable")
    period = np.median(np.diff(times))
    relevant = (times[:-1] >= start - 1e-6) & (times[:-1] < target_times[-1])
    if np.any(np.diff(times)[relevant] > 1.5 * period):
        raise ValueError("A logging gap crosses the observed future horizon")
    xy = np.array(
        [row["offline_ground_truth"]["position_world_m"][:2] for row in rows],
        dtype=float,
    )
    yaw = np.unwrap(
        np.deg2rad([row["offline_ground_truth"]["yaw_degrees"] for row in rows]),
    )
    if not np.isfinite(xy).all() or not np.isfinite(yaw).all():
        raise ValueError("Ground-truth ego poses must be finite")
    root_xy = np.asarray(anchor["offline_ground_truth"]["position_world_m"][:2])
    root_yaw = math.radians(anchor["offline_ground_truth"]["yaw_degrees"])
    delta = (
        np.stack(
            [np.interp(target_times, times, xy[:, axis]) for axis in range(2)], axis=-1,
        )
        - root_xy
    )
    rotation = np.array(
        [
            [math.cos(root_yaw), -math.sin(root_yaw)],
            [math.sin(root_yaw), math.cos(root_yaw)],
        ],
    )
    local_xy = delta @ rotation
    local_yaw = np.interp(target_times, times, yaw) - root_yaw
    return np.c_[local_xy, np.arctan2(np.sin(local_yaw), np.cos(local_yaw))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-experiment", type=Path, required=True)
    parser.add_argument("--agent-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if os.environ.get("LEAD_TRAINING_CONFIG"):
        raise ValueError("Unset model overrides to preserve reference identity")
    torch.set_num_threads(4)
    root = args.reference_experiment
    params_path = root / "fitted/parameters.json"
    parameters = json.loads(params_path.read_text())
    if len(parameters["members"]) != 1:
        raise ValueError(
            "This case-study scorer currently supports one reference member",
        )
    member = parameters["members"][0]
    checkpoint = Path(member["weights"])
    if digest(checkpoint) != member["sha256"]:
        raise ValueError("Reference weights changed since calibration")
    config = TrainingConfig(json.loads((checkpoint.parent / "config.json").read_text()))
    if not (
        config.adapt_history_free
        and config.adapt_rolling_causal_first
        and config.adapt_world_model
    ):
        raise ValueError(
            "Reference must use history-free causal-first targets and world modelling",
        )
    model = patch_norm_fp32(create_model(config)).cuda().eval()
    model.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True,
    )
    codes = model.adapt_decoder.kdisks_model.codebook.centroids
    if (
        hashlib.sha256(codes.cpu().numpy().tobytes()).hexdigest()
        != parameters["vocabulary_sha256"]
    ):
        raise ValueError("Calibrated vocabulary differs from the reference")
    scorer = NormativeTokenScorer(
        codes,
        parameters["heading_weight"],
        parameters["temperatures"],
        epsilon=parameters["epsilon"],
        sigma=parameters["sigma"],
    ).cuda()
    calibrations = {}
    for channel in ("score", "world_conflict_count"):
        saved = json.loads((root / f"{channel}_calibration.json").read_text())
        if saved["identity"]["scoring_parameters_sha256"] != digest(params_path):
            raise ValueError("Thresholds belong to other scoring parameters")
        calibrations[channel] = SceneCalibration.from_dict(saved["calibration"])
    report = {
        "reference_experiment": str(root.resolve()),
        "members": parameters["members"],
        "interpretation": __doc__,
        "cases": [],
    }
    sensor_keys = (
        "used_cameras",
        "pixels_per_meter",
        "min_x_meter",
        "max_x_meter",
        "min_y_meter",
        "max_y_meter",
        "max_speed",
        "waypoints_spacing",
        "num_way_points_prediction",
    )
    with torch.inference_mode():
        for directory in args.agent_dirs:
            records = [
                json.loads(line)
                for line in (directory / "driving_diagnostics.jsonl")
                .read_text()
                .splitlines()
            ]
            metadata = records[0]
            rows = [row for row in records if row["record_type"] == "step"]
            by_step = {row["step"]: row for row in rows}
            events = json.loads((directory / "infractions.json").read_text())[
                "infractions"
            ]
            collisions = [
                event for event in events if "COLLISION_VEHICLE" in event["event_type"]
            ]
            collision_times = [
                by_step[event["step"]]["timestamp_seconds"] for event in collisions
            ]
            case = {
                "agent_dir": str(directory.resolve()),
                "route_id": metadata["route_id"],
                "trace_sha256": digest(directory / "driving_diagnostics.jsonl"),
                "vehicle_collision_times": collision_times,
                "samples": [],
                "excluded_snapshots": [],
            }
            for path in sorted((directory / "model_snapshots").glob("*.pth")):
                snapshot = torch.load(path, map_location="cpu", weights_only=True)
                step = snapshot["step"]
                try:
                    future = observed_future(
                        rows, step, config.world_num_steps, config.world_step_seconds,
                    )
                except ValueError as error:
                    case["excluded_snapshots"].append(
                        {"step": step, "reason": str(error)},
                    )
                    continue
                current = by_step[step]
                if (
                    abs(current["timestamp_seconds"] - snapshot["timestamp_seconds"])
                    > 1e-6
                ):
                    raise ValueError(
                        "Sensor snapshot and pose trace timestamps disagree",
                    )
                previous_config = snapshot["metadata"]["training_config"]
                mismatches = [
                    key
                    for key in sensor_keys
                    if previous_config.get(key) != getattr(config, key)
                ]
                if mismatches:
                    raise ValueError(
                        f"Snapshot sensor/input geometry differs on {mismatches}",
                    )
                data = move_batch_to_device(snapshot["inputs"], "cuda")
                ego = torch.as_tensor(future[None], device="cuda", dtype=torch.float32)
                data.update(
                    future_waypoints=ego[..., :2],
                    future_yaws=ego[..., 2],
                    source_dataset=torch.tensor([SourceDataset.CARLA], device="cuda"),
                )
                model.adapt_decoder.training = True
                try:
                    with torch.autocast(
                        "cuda",
                        dtype=config.torch_float_type,
                        enabled=config.use_mixed_precision_training,
                    ):
                        prediction = model(data)
                finally:
                    model.adapt_decoder.training = False
                statistics = scorer(
                    model._adapt_outputs["output_logits"][:, 0].float()[None], ego[:, 0],
                )
                risk = interaction_features(ego, prediction.pred_actor_futures)
                sample = {
                    "step": step,
                    "timestamp_seconds": snapshot["timestamp_seconds"],
                    "snapshot_sha256": digest(path),
                    "ego_future": future.tolist(),
                    "ego_score_horizon_seconds": config.world_step_seconds,
                    "interaction_horizon_seconds": config.world_num_steps
                    * config.world_step_seconds,
                    "vehicle_collision_in_future_horizon": any(
                        snapshot["timestamp_seconds"]
                        < t
                        <= snapshot["timestamp_seconds"]
                        + config.world_num_steps * config.world_step_seconds
                        for t in collision_times
                    ),
                    **{
                        key: statistics[key].item()
                        for key in (
                            "score",
                            "aleatoric",
                            "epistemic",
                            "surprisal",
                            "quantization_residual",
                        )
                    },
                    "world_conflict_count": risk[
                        "expected_conflicting_actor_count"
                    ].item(),
                    "detected_vehicle_confidence_sum": risk[
                        "total_detected_vehicle_confidence"
                    ].item(),
                    "channels": {},
                }
                for channel, calibration in calibrations.items():
                    evaluated = calibration.evaluate(
                        [sample[channel]],
                        [sample["aleatoric"]],
                        [sample["epistemic"]],
                        [str(directory.resolve())],
                    )
                    sample["channels"][channel] = {
                        "flag": bool(evaluated["flag"][0]),
                        "exceedance_probability": float(
                            evaluated["exceedance_probability"][0],
                        ),
                        "threshold": float(evaluated["threshold"][0]),
                    }
                case["samples"].append(sample)
                print(
                    f"Scored route {metadata['route_id']} snapshot {step}", flush=True,
                )
            case["sensor_contract_fields"] = list(sensor_keys)
            report["cases"].append(case)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
