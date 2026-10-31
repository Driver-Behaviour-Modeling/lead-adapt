"""Fit the reference on validation drives and rescore four frozen drive roles.

Consumes a capture made with --save-member-logits and a drive-role manifest.
Only validation logits/labels enter optimization. This performs no model update.
Threshold calibration remains a separate step using the calibration features.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from lead.scoring.distributions import MotionSmoother, ensemble_step_statistics
from lead.scoring.parameters import fit_parameters


def role_indices(routes, manifest):
    roles = manifest["roles"]
    required = {"validation", "calibration", "drive_reference", "test"}
    if set(roles) != required or any(not rows for rows in roles.values()):
        raise ValueError("Four nonempty normal-drive roles are required")
    seen = set()
    indices = {}
    available = set(routes)
    for role, rows in roles.items():
        if len(set(rows)) != len(rows) or seen.intersection(rows):
            raise ValueError("Drive overlap or duplicates in partition manifest")
        if not set(rows).issubset(available):
            raise ValueError(f"Some {role} drives are absent from the capture")
        seen.update(rows)
        indices[role] = np.flatnonzero(np.isin(routes, rows))
    inventory = manifest.get("inventory", {})
    group_roles = {}
    for role, rows in roles.items():
        for drive in rows:
            if drive in inventory:
                group = inventory[drive]["route_group"]
                if group in group_roles and group_roles[group] != role:
                    raise ValueError("Repeated physical route crosses drive roles")
                group_roles[group] = role
    return indices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--partitions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--temperature-bounds", type=float, nargs=2, default=(0.25, 8.0),
    )
    parser.add_argument(
        "--epsilons", type=float, nargs="+", default=(0.0, 0.001, 0.01, 0.05, 0.1),
    )
    parser.add_argument(
        "--sigmas", type=float, nargs="+", default=(0.025, 0.05, 0.1, 0.25, 0.5),
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.batch_size <= 0:
        raise ValueError("Batch size must be positive")
    torch.set_num_threads(4)
    metadata = json.loads(args.capture.with_suffix(".json").read_text())
    manifest = json.loads(args.partitions.read_text())
    if (
        manifest.get("data_root")
        and Path(metadata["data_root"]).resolve()
        != Path(manifest["data_root"]).resolve()
    ):
        raise ValueError("Capture and partition manifest use different data roots")
    with np.load(args.capture.with_suffix(".npz"), allow_pickle=False) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    routes = np.array([str(Path(route).resolve()) for route in data["route"]])
    indices = role_indices(routes, manifest)
    centroids = data["vocabulary_centroids"]
    vocab_hash = hashlib.sha256(centroids.tobytes()).hexdigest()
    if vocab_hash != metadata["vocabulary_sha256"]:
        raise ValueError("Stored vocabulary differs from the capture identity")
    logits = data["member_logits"]
    if logits.shape[:2] != (len(metadata["members"]), len(routes)):
        raise ValueError("Member logits disagree with capture identity/frames")
    validation = indices["validation"]
    print(
        f"Fitting on {len(validation)} frames from {len(manifest['roles']['validation'])} validation drives only",
        flush=True,
    )
    parameters = fit_parameters(
        logits[:, validation],
        data["token"][validation],
        centroids,
        metadata["heading_weight"],
        temperature_bounds=args.temperature_bounds,
        epsilons=args.epsilons,
        sigmas=args.sigmas,
    )
    parameters.update(
        members=metadata["members"],
        vocabulary_sha256=vocab_hash,
        heading_weight=metadata["heading_weight"],
        parameter_fit_drives=manifest["roles"]["validation"],
        partitions_sha256=hashlib.sha256(args.partitions.read_bytes()).hexdigest(),
        capture_metadata_sha256=hashlib.sha256(
            args.capture.with_suffix(".json").read_bytes(),
        ).hexdigest(),
    )
    args.output_dir.mkdir(parents=True)
    parameter_file = args.output_dir / "parameters.json"
    parameter_file.write_text(json.dumps(parameters, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: parameters[key]
                for key in (
                    "temperatures",
                    "epsilon",
                    "sigma",
                    "initial_nll",
                    "fitted_nll",
                )
            },
        ),
        flush=True,
    )
    smoother = MotionSmoother(
        centroids,
        metadata["heading_weight"],
        parameters["epsilon"],
        parameters["sigma"],
    ).to(args.device)
    temperatures = torch.tensor(
        parameters["temperatures"], device=args.device, dtype=torch.float64,
    )[:, None, None]
    with torch.inference_mode():
        for role, selection in indices.items():
            frames = {}
            for start in range(0, len(selection), args.batch_size):
                batch = selection[start : start + args.batch_size]
                values = torch.as_tensor(logits[:, batch], device=args.device)
                tokens = torch.as_tensor(data["token"][batch], device=args.device)
                result = ensemble_step_statistics(
                    smoother(values, temperatures), tokens,
                )
                for key, value in result.items():
                    if key != "log_probability":
                        frames.setdefault(key, []).append(value.cpu().numpy())
            output = {key: np.concatenate(value) for key, value in frames.items()}
            output.update(
                {
                    key: data[key][selection]
                    for key in (
                        "token",
                        "quantization_residual",
                        "world_conflict_count",
                        "frame",
                    )
                },
            )
            output["route"] = routes[selection]
            forecast = {}
            if "forecast_actor_count" in data:
                count = data["forecast_actor_count"][:, selection].sum(1)
                current = data["forecast_current_vehicle_count"][:, selection].sum(1)
                for member, denominator in enumerate(count):
                    forecast[str(member)] = {
                        "supervised_actors": int(denominator),
                        "current_vehicle_labels": int(current[member]),
                        **{
                            name: float(
                                data[f"forecast_{name}_sum"][member, selection].sum()
                                / denominator,
                            )
                            if denominator
                            else None
                            for name in ("min_ade", "mode_ade", "min_fde", "mode_fde")
                        },
                    }
            np.savez_compressed(args.output_dir / f"{role}.npz", **output)
            saved_metadata = {
                **metadata,
                **{
                    key: parameters[key]
                    for key in (
                        "temperatures",
                        "epsilon",
                        "sigma",
                        "parameter_fit_drives",
                    )
                },
                "scoring_parameters_sha256": hashlib.sha256(
                    parameter_file.read_bytes(),
                ).hexdigest(),
                "partitions_sha256": parameters["partitions_sha256"],
                "role": role,
                "frames": len(selection),
                "calibrated": False,
                "parameter_fit_only": role == "validation",
                "forecast_errors_metres": forecast,
            }
            (args.output_dir / f"{role}.json").write_text(
                json.dumps(saved_metadata, indent=2) + "\n",
            )
            print(f"Rescored {role}: {len(selection)} frames", flush=True)


if __name__ == "__main__":
    main()
