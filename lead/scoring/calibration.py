"""Empirical uncertainty-bin calibration from normal drives only (PDF 3, §7).

Within-drive dependence means conformal coverage is an empirical target, not an
unconditional guarantee. Tiny bins are explicitly pooled; no test score chooses
its own bin. Calibrating thresholds cannot repair an inaccurate reference model.
"""

from dataclasses import asdict, dataclass

import numpy as np


def _array(value):
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 1 or not len(result) or not np.isfinite(result).all():
        raise ValueError("Expected a finite, nonempty one-dimensional array")
    return result


@dataclass
class SceneCalibration:
    alpha: float
    aleatoric_edges: list
    epistemic_edges: list
    bin_map: dict
    scores: dict
    thresholds: dict
    calibration_drives: list
    epistemic_limit: float

    @classmethod
    def fit(
        cls,
        score,
        aleatoric,
        epistemic,
        drive_ids,
        *,
        alpha=0.05,
        bins=2,
        min_bin_size=1000,
    ):
        if not 0 < alpha < 1 or type(bins) is not int or bins < 1:
            raise ValueError("alpha must be in (0, 1) and bins a positive integer")
        if type(min_bin_size) is not int or min_bin_size < 1:
            raise ValueError("min_bin_size must be positive")
        score, a, i = map(_array, (score, aleatoric, epistemic))
        drives = np.asarray(drive_ids, dtype=str)
        if not score.shape == a.shape == i.shape == drives.shape:
            raise ValueError(
                "Scores, uncertainty and drive IDs must have matching shapes",
            )
        if len(score) < min_bin_size:
            raise ValueError(
                "Not enough normal calibration steps, even for a pooled bin",
            )
        edges = [
            np.unique(np.quantile(x, np.arange(1, bins) / bins)).tolist()
            for x in (a, i)
        ]
        ai, ii = [
            np.searchsorted(e, x, side="right")
            for e, x in zip(edges, (a, i), strict=True)
        ]
        raw = np.array([f"{x}:{y}" for x, y in zip(ai, ii, strict=True)])
        mapping, distributions = {}, {"pooled": np.sort(score).tolist()}
        for x in range(len(edges[0]) + 1):
            for y in range(len(edges[1]) + 1):
                name = f"{x}:{y}"
                values = score[raw == name]
                mapping[name] = name if len(values) >= min_bin_size else "pooled"
                if mapping[name] == name:
                    distributions[name] = np.sort(values).tolist()
        return cls(
            alpha,
            *edges,
            mapping,
            distributions,
            {
                key: float(np.quantile(values, 1 - alpha, method="higher"))
                for key, values in distributions.items()
            },
            np.unique(drives).tolist(),
            float(np.quantile(i, 0.99, method="higher")),
        )

    def evaluate(self, score, aleatoric, epistemic, drive_ids):
        score, a, i = map(_array, (score, aleatoric, epistemic))
        drives = np.asarray(drive_ids, dtype=str)
        if not score.shape == a.shape == i.shape == drives.shape:
            raise ValueError(
                "Scores, uncertainty and drive IDs must have matching shapes",
            )
        if set(drives) & set(self.calibration_drives):
            raise ValueError("Evaluation drives overlap the threshold calibration set")
        ai = np.searchsorted(self.aleatoric_edges, a, side="right")
        ii = np.searchsorted(self.epistemic_edges, i, side="right")
        groups = np.array(
            [self.bin_map[f"{x}:{y}"] for x, y in zip(ai, ii, strict=True)],
        )
        exceedance, thresholds = np.empty_like(score), np.empty_like(score)
        for group in np.unique(groups):
            mask = groups == group
            reference = np.asarray(self.scores[group])
            # Count calibration values >= test value, including all ties.
            exceedance[mask] = (
                1
                + len(reference)
                - np.searchsorted(reference, score[mask], side="left")
            ) / (1 + len(reference))
            thresholds[mask] = self.thresholds[group]
        excess = np.maximum(score - thresholds, 0)
        low_confidence = i > self.epistemic_limit
        per_drive = {
            drive: {
                "U": float(excess[drives == drive].mean()),
                "total_excess": float(excess[drives == drive].sum()),
                "flag_fraction": float(
                    (exceedance[drives == drive] <= self.alpha).mean(),
                ),
                "low_confidence_fraction": float(
                    low_confidence[drives == drive].mean(),
                ),
            }
            for drive in np.unique(drives)
        }
        return {
            "bin": groups,
            "exceedance_probability": exceedance,
            "calibrated_score": -np.log(exceedance),
            "threshold": thresholds,
            "excess": excess,
            "low_confidence": low_confidence,
            "flag": exceedance <= self.alpha,
            "drives": per_drive,
        }

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(**data)


def drive_percentiles(
    test_drive_scores,
    reference_drive_scores,
    test_ids,
    reference_ids,
):
    """Upper-tail ranks against a *separate* set of normal reference drives."""
    test, reference = map(_array, (test_drive_scores, reference_drive_scores))
    if len(test) != len(test_ids) or len(reference) != len(reference_ids):
        raise ValueError("Drive IDs must match drive scores")
    if set(test_ids) & set(reference_ids):
        raise ValueError("Test drives overlap the drive-level reference set")
    return (1 + (reference[None] >= test[:, None]).sum(-1)) / (1 + len(reference))
