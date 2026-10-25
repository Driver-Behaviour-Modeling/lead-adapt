"""Drive-level null, sustained synthetic anomalies and residual scenario bias on Stage 1.

Reads the per-frame outputs of scripts/score_surprisal.py (outputs/surprisal/<run>.npz)
and splits the drives into two halves by route parity: everything that is fitted
(per-step bias, drive-level variance factor) comes from half A, everything that is
reported is measured on half B.

1. Drive-level null. Under a calibrated, memoryless score Z = S / sqrt(V) ~ N(0, 1).
   The score has memory, so Var S exceeds V. Two corrections are fitted on half A:
   - theory: Z / sqrt(c), c = (Var s / mean nu^2) * (1 + 2 sum_k w_k rho_k), the
     Newey-West long-run variance with Bartlett weights w_k = 1 - k / (K + 1);
   - empirical: Z / sqrt(c), c = variance of the bias-removed Z across half-A drives.
   Each is reported on half B as mean, std and share of drives with |Z| > 2 (target 4.6%).
2. Sustained anomalies. A synthetic anomaly lasting n steps (n x 0.25 s) is
   approximated in the same scene by summing the per-frame anomalous step scores
   over n consecutive frames (open loop: the scene and the ego state do not react).
   It is compared with the normal score summed over every n-frame stretch of half B:
   AUROC, and detection at the normal 99% / 95% quantile (1% / 5% false alarms).
3. Residual scenario bias: mean s per scenario type on half B.

Usage (repo root):
    python scripts/analyze_stage1_calibration.py --run stage1_full_histfree_e29_T1.5
"""

import argparse
import json

import numpy as np
from scipy.stats import rankdata

parser = argparse.ArgumentParser()
parser.add_argument("--run", required=True, help="name of an outputs/surprisal/<run>.npz")
parser.add_argument("--max_lag", type=int, default=16, help="autocorrelation lags for the long-run variance (x 0.25 s)")
parser.add_argument("--durations", type=int, nargs="+", default=[1, 2, 4, 8], help="anomaly lengths in steps")
args = parser.parse_args()

f = np.load(f"outputs/surprisal/{args.run}.npz")
route, s, nu2 = f["route"], f["l"] - f["H"], f["nu2"]
scenarios = np.array([str(p).split("/")[-2] for p in f["route_names"]])
anomalies = [k[2:] for k in f.files if k.startswith("l_") and k != "l_win"]
half = route % 2  # 0 = A (fit), 1 = B (report)
drives = {h: [r for r in np.unique(route) if r % 2 == h] for h in (0, 1)}


def acf(mask: np.ndarray, lag: int) -> float:
    same = (route[lag:] == route[:-lag]) & mask[lag:] & mask[:-lag]
    return float(np.corrcoef(s[lag:][same], s[:-lag][same])[0, 1])


def drive_z(h: int, bias: float) -> np.ndarray:
    return np.array([(s[route == r] - bias).sum() / np.sqrt(nu2[route == r].sum()) for r in drives[h]])


def describe(z: np.ndarray) -> dict:
    return {"mean": float(z.mean()), "std": float(z.std()), "frac_abs_gt_2": float((abs(z) > 2).mean()), "drives": len(z)}


def auroc(positive: np.ndarray, negative: np.ndarray) -> float:
    ranks = rankdata(np.concatenate([positive, negative]))[: len(positive)]
    return float((ranks.sum() - len(positive) * (len(positive) + 1) / 2) / (len(positive) * len(negative)))


result = {"run": args.run}

# ---------------------------------------------------------------------------
# 1. Drive-level null
# ---------------------------------------------------------------------------
A = half == 0
bias = float(s[A].mean())
rho = np.array([acf(A, k) for k in range(1, args.max_lag + 1)])
weights = 1 - np.arange(1, args.max_lag + 1) / (args.max_lag + 1)
c_theory = float(s[A].var() / nu2[A].mean() * (1 + 2 * (weights * rho).sum()))
c_empirical = float(drive_z(0, bias).var())
z_raw_B, z_B = drive_z(1, 0.0), drive_z(1, bias)
result["drive_null"] = {
    "fitted_on_A": {"per_step_bias": bias, "rho_1_4_8_16": rho[[0, 3, 7, 15]].tolist(), "c_theory": c_theory, "c_empirical": c_empirical},
    "B_uncorrected": describe(z_raw_B),
    "B_bias_removed": describe(z_B),
    "B_theory": describe(z_B / np.sqrt(c_theory)),
    "B_empirical": describe(z_B / np.sqrt(c_empirical)),
}

# ---------------------------------------------------------------------------
# 2. Sustained anomalies (open loop): summed step scores over n consecutive frames
# ---------------------------------------------------------------------------


def window_sums(values: np.ndarray, n: int) -> np.ndarray:
    """Sums over every n consecutive frames of each half-B drive; windows with a NaN are dropped."""
    out = []
    for r in drives[1]:
        v = values[route == r]
        if len(v) < n:
            continue
        c = np.concatenate([[0.0], np.cumsum(np.nan_to_num(v))])
        bad = np.concatenate([[0], np.cumsum(np.isnan(v))])
        sums = c[n:] - c[:-n]
        out.append(sums[(bad[n:] - bad[:-n]) == 0])
    return np.concatenate(out)


result["sustained"] = {}
for n in args.durations:
    normal = window_sums(s, n)
    q99, q95 = np.quantile(normal, [0.99, 0.95])
    result["sustained"][n] = {}
    for a in anomalies:
        anom = window_sums(f[f"l_{a}"] - f["H"], n)
        result["sustained"][n][a] = {
            "windows": len(anom),
            "auroc": auroc(anom, normal),
            "detected_at_1pct": float((anom > q99).mean()),
            "detected_at_5pct": float((anom > q95).mean()),
        }

# ---------------------------------------------------------------------------
# 3. Residual scenario bias on half B
# ---------------------------------------------------------------------------
B = half == 1
scen = scenarios[route]
result["scenarios_B"] = {
    name: {"frames": int((B & (scen == name)).sum()), "mean_s": float(s[B & (scen == name)].mean())}
    for name in np.unique(scen[B])
}
result["B_mean_s"] = float(s[B].mean())

with open(f"outputs/surprisal/{args.run}_analysis.json", "w") as out:
    json.dump(result, out, indent=2)

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
d = result["drive_null"]
print(f"{args.run}: half A {len(drives[0])} drives (fit), half B {len(drives[1])} drives (report)")
print(f"per-step bias on A {bias:+.3f}; autocorrelation lags 1/4/8/16: {np.round(rho[[0, 3, 7, 15]], 3)}")
print(f"variance factor c: theory {c_theory:.2f}, empirical {c_empirical:.2f}")
print(f"{'drive Z on B':24s}{'mean':>8s}{'std':>8s}{'|Z|>2':>8s}   (target 0 / 1 / 0.046)")
for label in ("B_uncorrected", "B_bias_removed", "B_theory", "B_empirical"):
    print(f"{label:24s}{d[label]['mean']:+8.2f}{d[label]['std']:8.2f}{d[label]['frac_abs_gt_2']:8.3f}")
print("\nsustained anomalies on B (open loop, summed step scores): AUROC / detected at 1% / at 5% false alarms")
print(f"{'':14s}" + "".join(f"{f'{n * 0.25:g} s':>22s}" for n in args.durations))
for a in anomalies:
    cells = [result["sustained"][n][a] for n in args.durations]
    print(f"{a:14s}" + "".join(f"{c['auroc']:8.3f} {c['detected_at_1pct']:5.2f} {c['detected_at_5pct']:5.2f}  " for c in cells))
print(f"\nmean s on B: {result['B_mean_s']:+.3f}; scenario types by mean s on B (frames):")
ranked = sorted(result["scenarios_B"].items(), key=lambda kv: -kv[1]["mean_s"])
for name, m in ranked[:8] + [("...", None)] + ranked[-4:]:
    print(f"  {name:45s}" + (f"{m['mean_s']:+.2f} ({m['frames']})" if m else ""))
