"""Fit/evaluate normal-drive calibration of ego or world-interaction evidence.

Fit on normal calibration drives only. Use a separate normal drive-reference
set to rank whole drives, and evaluate false flags on another held-out set.
This tool cannot establish that input drives are normal; the data split is an
experiment input and must be audited. Report each channel separately.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from lead.scoring.calibration import SceneCalibration, drive_percentiles


def identity(metadata):
    result = {
        key: metadata[key]
        for key in (
            "members",
            "temperatures",
            "epsilon",
            "sigma",
            "vocabulary_sha256",
            "heading_weight",
        )
    }
    result.update(
        {
            key: metadata.get(key)
            for key in ("scoring_parameters_sha256", "parameter_fit_drives")
        },
    )
    return result


def features(path):
    metadata = json.loads(path.with_suffix(".json").read_text())
    with np.load(path.with_suffix(".npz"), allow_pickle=False) as data:
        arrays = {
            key: data[key]
            for key in (
                "score",
                "aleatoric",
                "epistemic",
                "route",
                "frame",
                "world_conflict_count",
            )
        }
    if set(metadata.get("parameter_fit_drives", [])).intersection(arrays["route"]):
        raise ValueError(
            "Parameter-fitting drives overlap calibration/evaluation drives",
        )
    return metadata, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("fit")
    fit.add_argument("--features", type=Path, required=True)
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument(
        "--channel", choices=("score", "world_conflict_count"), default="score",
    )
    fit.add_argument("--alpha", type=float, default=0.05)
    fit.add_argument("--bins", type=int, default=2)
    fit.add_argument("--min-bin-size", type=int, default=1000)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--calibration", type=Path, required=True)
    evaluate.add_argument("--features", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument(
        "--normal-drive-reference",
        type=Path,
        help="Evaluation JSON for a separate set of normal reference drives",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.output.suffix == ".npz":
        raise ValueError("--output must name the JSON report, not its companion NPZ")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata, data = features(args.features)
    if args.command == "fit":
        calibration = SceneCalibration.fit(
            data[args.channel],
            data["aleatoric"],
            data["epistemic"],
            data["route"],
            alpha=args.alpha,
            bins=args.bins,
            min_bin_size=args.min_bin_size,
        )
        output = {
            "channel": args.channel,
            "identity": identity(metadata),
            "source_features": str(args.features.resolve()),
            "calibration": calibration.to_dict(),
            "interpretation": "Thresholds fitted on user-supplied normal calibration drives; held-out coverage is not established by fitting.",
        }
    else:
        saved = json.loads(args.calibration.read_text())
        if identity(metadata) != saved["identity"]:
            raise ValueError(
                "Scoring checkpoint, vocabulary, or scoring parameters differ from calibration",
            )
        calibration = SceneCalibration.from_dict(saved["calibration"])
        channel = saved["channel"]
        result = calibration.evaluate(
            data[channel], data["aleatoric"], data["epistemic"], data["route"],
        )
        output = {
            "channel": channel,
            "identity": saved["identity"],
            "calibration_drives": calibration.calibration_drives,
            "source_features": str(args.features.resolve()),
            "drives": result["drives"],
            "target_alpha": calibration.alpha,
            "step_flag_fraction": float(result["flag"].mean()),
            "bin_counts_and_flag_rates": {
                group: {
                    "n": int((result["bin"] == group).sum()),
                    "flag_fraction": float(
                        result["flag"][result["bin"] == group].mean(),
                    ),
                }
                for group in np.unique(result["bin"])
            },
            "interpretation": "Held-out empirical flags. Within-drive dependence prevents an unconditional conformal-coverage guarantee.",
        }
        scenarios = np.array(
            [Path(route).parent.name or "unlabeled" for route in data["route"]],
        )
        output["scenario_counts_and_flag_rates"] = {
            str(scenario): {
                "n": int((scenarios == scenario).sum()),
                "flag_fraction": float(result["flag"][scenarios == scenario].mean()),
            }
            for scenario in np.unique(scenarios)
        }
        drive_ids = np.unique(data["route"])
        totals = np.array([(data["route"] == drive).sum() for drive in drive_ids])
        flagged = np.array(
            [result["flag"][data["route"] == drive].sum() for drive in drive_ids],
        )
        draws = np.random.default_rng(20261024).integers(
            len(drive_ids), size=(1000, len(drive_ids)),
        )
        rates = flagged[draws].sum(1) / totals[draws].sum(1)
        output["drive_cluster_bootstrap_95_interval"] = np.quantile(
            rates, [0.025, 0.975],
        ).tolist()
        if args.normal_drive_reference:
            reference = json.loads(args.normal_drive_reference.read_text())
            if (
                reference["identity"] != output["identity"]
                or reference["channel"] != channel
                or reference["calibration_drives"] != output["calibration_drives"]
            ):
                raise ValueError(
                    "Drive reference uses another model, channel, or calibration set",
                )
            drive_ids = list(output["drives"])
            reference_ids = list(reference["drives"])
            ranks = drive_percentiles(
                [output["drives"][key]["U"] for key in drive_ids],
                [reference["drives"][key]["U"] for key in reference_ids],
                drive_ids,
                reference_ids,
            )
            for key, rank in zip(drive_ids, ranks, strict=True):
                output["drives"][key]["upper_tail_probability"] = float(rank)
        array_path = args.output.with_suffix(".npz")
        if array_path.exists() or array_path == args.output:
            raise FileExistsError(array_path)
        np.savez_compressed(
            array_path,
            **{key: value for key, value in result.items() if key != "drives"},
            route=data["route"],
            frame=data["frame"],
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Saved {args.command} output to {args.output}")


if __name__ == "__main__":
    main()
