"""Graded replay sensitivity of separately calibrated ego and interaction channels.

Only held-out test actions are changed. Initial scenes and other traffic stay
fixed. These perturbations are not labels of policy incorrectness or BADCAR
results. Sustained windows repeat perturbations in recorded scenes, not in a
reacting closed-loop rollout, and are not AR block-likelihood scores.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata

from lead.scoring.calibration import SceneCalibration, drive_percentiles
from lead.scoring.distributions import MotionSmoother
from lead.scoring.perturbations import (
    contiguous_window_means,
    perturb_ego_motion,
    severity_correlations,
    upper_tail_probability,
)
from lead.scoring.world_interactions import interaction_features


def load(prefix):
    with np.load(prefix.with_suffix(".npz"), allow_pickle=False) as values:
        return {key: values[key] for key in values.files}


def auroc(positive, negative):
    if not len(positive) or not len(negative):
        return None
    ranks = rankdata(np.r_[positive, negative])[: len(positive)]
    return float(
        (ranks.sum() - len(positive) * (len(positive) + 1) / 2)
        / (len(positive) * len(negative)),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--step-seconds", type=float, default=0.25)
    parser.add_argument("--frame-interval-seconds", type=float, default=0.25)
    parser.add_argument("--window-frames", type=int, nargs="+", default=(1, 2, 4, 8))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if any(width < 1 for width in args.window_frames):
        raise ValueError("Window lengths must be positive")
    torch.set_num_threads(4)
    root = args.experiment_dir
    parameters_path = root / "fitted/parameters.json"
    parameters = json.loads(parameters_path.read_text())
    parameter_hash = hashlib.sha256(parameters_path.read_bytes()).hexdigest()
    raw_meta = json.loads((root / "raw.json").read_text())
    if raw_meta["members"] != parameters["members"]:
        raise ValueError("Raw capture and scoring parameters use different members")
    raw = load(root / "raw")
    roles = json.loads((root / "partitions.json").read_text())["roles"]
    routes = np.array([str(Path(value).resolve()) for value in raw["route"]])
    test_rows = np.flatnonzero(np.isin(routes, roles["test"]))
    ref_rows = np.flatnonzero(np.isin(routes, roles["drive_reference"]))
    test, reference = load(root / "fitted/test"), load(root / "fitted/drive_reference")
    if not np.array_equal(routes[test_rows], test["route"]) or not np.array_equal(
        raw["frame"][test_rows], test["frame"],
    ):
        raise ValueError("Raw and fitted test captures are not aligned")
    if not np.array_equal(routes[ref_rows], reference["route"]) or not np.array_equal(
        raw["frame"][ref_rows], reference["frame"],
    ):
        raise ValueError("Raw and fitted reference captures are not aligned")
    codes = torch.as_tensor(raw["vocabulary_centroids"], dtype=torch.float64)
    logits = torch.as_tensor(raw["member_logits"][:, test_rows], dtype=torch.float64)
    temperatures = torch.tensor(parameters["temperatures"], dtype=torch.float64)[
        :, None, None,
    ]
    log_p = (logits / temperatures).log_softmax(-1)
    unsmoothed_aleatoric = -(log_p.exp() * log_p).sum(-1).mean(0).numpy()
    smoother = MotionSmoother(
        codes, parameters["heading_weight"], parameters["epsilon"], parameters["sigma"],
    )

    def ego_score(poses):
        observed = poses[:, 0]
        distance = torch.cdist(observed[:, :2], codes[:, :2])
        yaw = observed[:, 2, None] - codes[None, :, 2]
        distance += (
            parameters["heading_weight"] * torch.atan2(yaw.sin(), yaw.cos()).abs()
        )
        token = distance.argmin(-1)
        selected = log_p.gather(
            -1, token[None, :, None].expand(len(log_p), -1, -1),
        ).squeeze(-1)
        unsmoothed_surprisal = -(selected.logsumexp(0) - math.log(len(selected)))
        if parameters["epsilon"]:
            blurred = torch.logsumexp(log_p + smoother.log_kernel[token][None], dim=-1)
            selected = torch.logaddexp(
                selected + math.log1p(-parameters["epsilon"]),
                blurred + math.log(parameters["epsilon"]),
            )
        surprisal = -(selected.logsumexp(0) - math.log(len(selected)))
        return (
            surprisal.numpy() - test["aleatoric"],
            token.numpy(),
            unsmoothed_surprisal.numpy() - unsmoothed_aleatoric,
        )

    def world_score(poses):
        result = []
        for start in range(0, len(poses), 128):
            rows = test_rows[start : start + 128]
            member_risk = []
            for member in range(len(log_p)):
                prediction = {
                    key: torch.as_tensor(raw[f"world_{key}"][member, rows])
                    for key in (
                        "future_positions",
                        "future_velocities",
                        "memory_confidence",
                    )
                }
                member_risk.append(
                    interaction_features(poses[start : start + 128], prediction)[
                        "expected_conflicting_actor_count"
                    ],
                )
            result.append(torch.stack(member_risk).mean(0).numpy())
        return np.concatenate(result)

    poses = torch.as_tensor(raw["ego_future"][test_rows], dtype=torch.float64)
    baseline_ego, baseline_tokens, _baseline_unsmoothed = ego_score(poses)
    if not np.allclose(baseline_ego, test["score"], atol=1e-7, rtol=1e-7):
        raise ValueError("Reconstructed ego scores disagree with fitted capture")
    if not np.allclose(world_score(poses), test["world_conflict_count"], atol=1e-6):
        raise ValueError("Reconstructed interaction evidence disagrees with capture")
    calibrations, normal_test, normal_ref = {}, {}, {}
    for channel in ("score", "world_conflict_count"):
        saved = json.loads((root / f"{channel}_calibration.json").read_text())
        if saved["identity"]["scoring_parameters_sha256"] != parameter_hash:
            raise ValueError("Thresholds use different scoring parameters")
        calibration = SceneCalibration.from_dict(saved["calibration"])
        calibrations[channel] = calibration
        normal_test[channel] = calibration.evaluate(
            test[channel], test["aleatoric"], test["epistemic"], test["route"],
        )
        normal_ref[channel] = calibration.evaluate(
            reference[channel],
            reference["aleatoric"],
            reference["epistemic"],
            reference["route"],
        )
    report = {
        "experiment_dir": str(root.resolve()),
        "scoring_parameters_sha256": parameter_hash,
        "test_drives": len(np.unique(test["route"])),
        "test_frames": len(test_rows),
        "step_seconds": args.step_seconds,
        "assumed_recorded_frame_interval_seconds": args.frame_interval_seconds,
        "normal_flag_fractions": {
            key: float(value["flag"].mean()) for key, value in normal_test.items()
        },
        "interpretation": __doc__,
        "perturbations": {},
    }
    trials = [("accelerate", a, 0, 0) for a in (1, 2, 4)]
    trials += [("brake", -a, 0, 3) for a in (1, 3, 6)]
    trials += [
        (name, 0, sign * a, 3)
        for name, sign in (("swerve_positive_y", 1), ("swerve_negative_y", -1))
        for a in (1, 2, 4)
    ]
    tail_values = {}
    for name, longitudinal, lateral, min_speed in trials:
        eligible = raw["ego_speed"][test_rows] >= min_speed
        ref_eligible = raw["ego_speed"][ref_rows] >= min_speed
        changed = perturb_ego_motion(poses, longitudinal, lateral, dt=args.step_seconds)
        changed[~eligible] = poses[~eligible]
        ego, tokens, unsmoothed = ego_score(changed)
        tail = tail_values.setdefault(
            name, {"levels": [], "smoothed": [], "unsmoothed": []},
        )
        tail["levels"].append(max(abs(longitudinal), abs(lateral)))
        tail["smoothed"].append(ego[eligible])
        tail["unsmoothed"].append(unsmoothed[eligible])
        evidence = {"score": ego, "world_conflict_count": world_score(changed)}
        trial = {
            "longitudinal_residual_acceleration": longitudinal,
            "lateral_residual_acceleration": lateral,
            "eligible_frames": int(eligible.sum()),
            "minimum_initial_speed_metres_per_second": min_speed,
            "token_changed_fraction": float(
                (tokens[eligible] != baseline_tokens[eligible]).mean(),
            )
            if eligible.any()
            else None,
            "channels": {},
        }
        for channel, values in evidence.items():
            calibration = calibrations[channel]
            evaluated = calibration.evaluate(
                values, test["aleatoric"], test["epistemic"], test["route"],
            )
            normal, ref = normal_test[channel], normal_ref[channel]
            metrics = {
                "normal_eligible_flag_fraction": float(normal["flag"][eligible].mean())
                if eligible.any()
                else None,
                "perturbed_eligible_flag_fraction": float(
                    evaluated["flag"][eligible].mean(),
                )
                if eligible.any()
                else None,
                "step_sensitivity_auroc": auroc(
                    values[eligible], test[channel][eligible],
                ),
                "windows": {},
            }
            for width in args.window_frames:
                windows = contiguous_window_means(
                    evaluated["excess"], test["route"], test["frame"], width, eligible,
                )
                original = contiguous_window_means(
                    normal["excess"], test["route"], test["frame"], width, eligible,
                )
                reference_windows = contiguous_window_means(
                    ref["excess"],
                    reference["route"],
                    reference["frame"],
                    width,
                    ref_eligible,
                )
                if not len(windows) or not len(reference_windows):
                    continue
                metrics["windows"][str(width)] = {
                    "duration_seconds": width * args.frame_interval_seconds,
                    "windows": len(windows),
                    "reference_windows": len(reference_windows),
                    "normal_flag_fraction": float(
                        (
                            upper_tail_probability(original, reference_windows)
                            <= calibration.alpha
                        ).mean(),
                    ),
                    "perturbed_flag_fraction": float(
                        (
                            upper_tail_probability(windows, reference_windows)
                            <= calibration.alpha
                        ).mean(),
                    ),
                    "sensitivity_auroc": auroc(windows, original),
                }
            drive_ids, ref_ids = list(evaluated["drives"]), list(ref["drives"])
            ranks = drive_percentiles(
                [evaluated["drives"][key]["U"] for key in drive_ids],
                [ref["drives"][key]["U"] for key in ref_ids],
                drive_ids,
                ref_ids,
            )
            metrics["perturbed_drive_flag_fraction"] = float(
                (ranks <= calibration.alpha).mean(),
            )
            trial["channels"][channel] = metrics
        label = f"{name}_{max(abs(longitudinal), abs(lateral))}"
        report["perturbations"][label] = trial
        print(f"Evaluated {label}", flush=True)
    report["smoothing_tail_ordering"] = {
        "method": "Per-context Spearman correlation across three severities. Both references use the same validation-fitted temperatures; the unsmoothed reference sets epsilon=0 and recomputes aleatoric entropy. No parameter is tuned on these test actions.",
        "families": {},
    }
    for family, values in tail_values.items():
        smooth = severity_correlations(values["levels"], np.array(values["smoothed"]))
        raw = severity_correlations(values["levels"], np.array(values["unsmoothed"]))
        both = np.isfinite(smooth) & np.isfinite(raw)
        report["smoothing_tail_ordering"]["families"][family] = {
            "severity_levels": values["levels"],
            "contexts": len(smooth),
            "smoothed_rankable_contexts": int(np.isfinite(smooth).sum()),
            "unsmoothed_rankable_contexts": int(np.isfinite(raw).sum()),
            "smoothed_mean_spearman": float(smooth[np.isfinite(smooth)].mean())
            if np.isfinite(smooth).any()
            else None,
            "unsmoothed_mean_spearman": float(raw[np.isfinite(raw)].mean())
            if np.isfinite(raw).any()
            else None,
            "jointly_rankable_contexts": int(both.sum()),
            "mean_paired_improvement": float((smooth[both] - raw[both]).mean())
            if both.any()
            else None,
            "fraction_improved": float((smooth[both] > raw[both] + 1e-10).mean())
            if both.any()
            else None,
            "fraction_worsened": float((smooth[both] < raw[both] - 1e-10).mean())
            if both.any()
            else None,
            "smoothed_mean_score_by_level": [
                float(row.mean()) if len(row) else None for row in values["smoothed"]
            ],
            "unsmoothed_mean_score_by_level": [
                float(row.mean()) if len(row) else None for row in values["unsmoothed"]
            ],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
