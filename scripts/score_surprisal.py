"""Excess-surprisal calibration of the ADAPT token head on normative drives.

Scores whole drives of a CARLA data root frame by frame (4 Hz, so one token =
one frame step) under a history-free checkpoint, conditioning on scene and
current state only. Augmentation is off, so the observed future starts at the
ego origin and its rolling tokens are what the reference model is asked to
explain.

Per frame:
- step score from the first token: l = -log p(u_1 | c), entropy H, varentropy
  nu^2, s = l - H, z = s / nu
- window score over the 2 s horizon (8 tokens): l_win from teacher forcing,
  H_win and nu^2_win estimated from --window_samples ancestral samples
- top-1 confidence/accuracy, coverage mass m = P(codes more likely than u_1),
  and the quantization residual of the first step

- synthetic step anomalies: the observed first step replaced by a hard brake
  (-6 m/s^2), a speed-up (+4 m/s^2) or a swerve (4 m/s^2 lateral, either side),
  snapped to its nearest code; their surprisal under the same scene, and how well
  their s separates from the normal s (AUROC, detection at 1% / 5% false alarms)

On normative drives a calibrated reference gives mean s ~ 0 overall and within
speed bins / entropy deciles, Var s ~ mean nu^2, drive-level Z = S / sqrt(V)
with mean 0 and std 1, coverage P(m + U p(u_1) < a) ~ a, and low ECE.

--temperature T divides the logits by T before every softmax (post-hoc
temperature scaling; T > 1 widens the distributions).

Usage (repo root, LEAD_TRAINING_CONFIG unset):
    CUDA_VISIBLE_DEVICES=GPU-4ec20303-3c97-fc1c-9499-e5539efe3717 \
    python scripts/score_surprisal.py \
        --checkpoint_dir outputs/checkpoints/histfree_e29 --num_routes 200
    # unseen drives: --data_root data/b2d_expert (scores every route)
"""

import argparse
import glob
import json
import os

import numpy as np
import torch
from scipy.stats import rankdata
from torch.utils.data import DataLoader, Subset

from lead.adapt.adapt_decoder import compute_kdisks_distances, generate_causal_mask
from lead.data_loader.carla_dataset import CARLAData
from lead.training.config_training import TrainingConfig
from lead.training.input_pipeline import prepared_training_batches
from lead.training.mixed_training_utils import mixed_data_collate_fn
from lead.training.training_utils import create_model

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_dir", required=True)
parser.add_argument("--data_root", default=None, help="carla_root to score; default: the training root")
parser.add_argument("--num_routes", type=int, default=0, help="random routes to score; 0 = all")
parser.add_argument("--window_samples", type=int, default=32)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--num_workers", type=int, default=16)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--temperature", type=float, default=1.0, help="divide the logits by T before every softmax")
parser.add_argument("--name", default=None, help="output name; default: <checkpoint>_<data root>")
args = parser.parse_args()

OVERRIDES = {"use_sensor_perburtation": False, "use_color_aug": False}
if args.data_root:
    OVERRIDES.update(carla_root=args.data_root, carla_data=os.path.join(args.data_root, "data"), use_persistent_cache=False)

with open(os.path.join(args.checkpoint_dir, "config.json")) as f:
    loaded = json.load(f)
loaded.update(OVERRIDES)
config = TrainingConfig(loaded, raise_error_on_missing_key=False)
(weights,) = glob.glob(os.path.join(args.checkpoint_dir, "model*.pth"))
model = create_model(config)
model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
model = model.cuda().eval()
decoder = model.adapt_decoder
assert decoder.history_free, "scoring conditions on scene + current state only"
centroids = decoder.kdisks_model.codebook.centroids.float()
heading_weight = decoder.kdisks_model.codebook.heading_weight
dt = config.waypoints_spacing / config.carla_fps



def step_offset(longitudinal: float, lateral: float):
    """Change of the body-frame step (dx, dy, dyaw) under an extra acceleration (m/s^2) at speed v."""
    return lambda v: torch.stack(
        [torch.full_like(v, 0.5 * longitudinal * dt**2), torch.full_like(v, 0.5 * lateral * dt**2), lateral / v.clamp_min(1.0) * dt],
        dim=-1,
    )


# Synthetic first-step anomalies: (offset added to the observed step, minimum speed for it to make sense).
ANOMALIES = {
    "brake": (step_offset(-6.0, 0.0), 3.0),
    "accelerate": (step_offset(4.0, 0.0), 0.0),
    "swerve_left": (step_offset(0.0, 4.0), 3.0),
    "swerve_right": (step_offset(0.0, -4.0), 3.0),
}

# The teacher-forced pass hands the decoder its cross-attention memory; keep it for sampling.
captured = {}
decoder.transformer_decoder.register_forward_pre_hook(
    lambda _, __, kwargs: captured.update(context=kwargs["encoder_output"]), with_kwargs=True,
)

# Whole drives in time order.
dataset = CARLAData(root=config.carla_data, config=config, training_session_cache=None, build_cache=False, random=False)
route_dirs = dataset.route_dirs.astype(str)
routes = np.unique(route_dirs)
if args.num_routes:
    routes = np.random.default_rng(args.seed).choice(routes, args.num_routes, replace=False)
indices = np.flatnonzero(np.isin(route_dirs, routes))
indices = indices[np.lexsort((dataset.route_indices[indices], route_dirs[indices]))]
loader = DataLoader(
    Subset(dataset, indices.tolist()),
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    collate_fn=mixed_data_collate_fn,
)
print(f"scoring {len(indices)} frames from {len(routes)} routes of {config.carla_data}")

frames: dict[str, list[np.ndarray]] = {}


def add(name: str, value: torch.Tensor) -> None:
    frames.setdefault(name, []).append(value.float().cpu().numpy())


def surprisal_stats(log_p: torch.Tensor, token: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Surprisal of ``token``, entropy and varentropy of each row of log-probabilities."""
    p = log_p.exp()
    entropy = -(p * log_p).sum(-1)
    varentropy = (p * log_p.square()).sum(-1) - entropy.square()
    return -log_p.gather(-1, token[:, None])[:, 0], entropy, varentropy.clamp_min(0)


with prepared_training_batches(loader, config) as batches, torch.inference_mode():
    for data in batches:
        with torch.amp.autocast("cuda", dtype=config.torch_float_type, enabled=config.use_mixed_precision_training):
            # Teacher forcing: the decoder takes its true-prefix path, submodules stay in eval mode.
            decoder.training = True
            model(data=data)
            decoder.training = False
            outputs = model._adapt_outputs
            tokens = outputs["future_token_ids"]
            log_p = torch.log_softmax(outputs["output_logits"].float() / args.temperature, dim=-1)  # [B, T, V]
            bs, steps = tokens.shape

            # Step score: the first token given scene + current state.
            surprisal, entropy, varentropy = surprisal_stats(log_p[:, 0], tokens[:, 0])
            add("l", surprisal)
            add("H", entropy)
            add("nu2", varentropy)
            add("top1_p", log_p[:, 0].max(-1).values.exp())
            add("top1_correct", log_p[:, 0].argmax(-1) == tokens[:, 0])
            p_u = (-surprisal).exp()
            add("coverage_mass", (log_p[:, 0].exp() * (log_p[:, 0].exp() > p_u[:, None])).sum(-1))
            residual = outputs["future_deltas"][:, 0].float() - centroids[tokens[:, 0]]
            residual[:, 2] = torch.atan2(torch.sin(residual[:, 2]), torch.cos(residual[:, 2]))
            add("residual", residual[:, :2].norm(dim=-1) + heading_weight * residual[:, 2].abs())
            add("token", tokens[:, 0])

            # Synthetic anomalies: surprisal of the perturbed first step in the same scene.
            first = outputs["future_deltas"][:, 0].float()
            speed = torch.as_tensor(data["speed"]).float().to(first.device).view(-1)
            for anomaly, (offset, min_speed) in ANOMALIES.items():
                code = compute_kdisks_distances(first + offset(speed), centroids, heading_weight).argmin(-1)
                valid = (speed > min_speed) & (code != tokens[:, 0])
                add(f"l_{anomaly}", torch.where(valid, -log_p[:, 0].gather(-1, code[:, None])[:, 0], torch.nan))

            # Window score: the whole 2 s token sequence, entropy by ancestral sampling.
            add("l_win", -log_p.gather(-1, tokens[..., None])[..., 0].sum(-1))
            samples = args.window_samples
            context = captured["context"].repeat_interleave(samples, dim=0)
            sequence = torch.full((bs * samples, 1), decoder._start_token, device=tokens.device)
            sample_log_p = torch.zeros(bs * samples, device=tokens.device)
            for _ in range(steps):
                hidden = decoder.transformer_decoder(
                    tgt_token_ids=sequence,
                    encoder_output=context,
                    self_attention_mask=generate_causal_mask(sequence.shape[1], sequence.device),
                )
                step_log_p = torch.log_softmax(decoder.output_projection(hidden[:, -1]).float() / args.temperature, dim=-1)
                token = torch.multinomial(step_log_p.exp(), 1)
                sample_log_p += step_log_p.gather(-1, token)[:, 0]
                sequence = torch.cat([sequence, token], dim=1)
            sample_nll = -sample_log_p.view(bs, samples)
            add("H_win", sample_nll.mean(-1))
            add("nu2_win", sample_nll.var(-1))

        add("speed", torch.as_tensor(data["speed"]).view(-1))
        add("yaw_rate", torch.as_tensor(data["future_yaws"])[:, 0] / dt)

per_frame = {name: np.concatenate(values) for name, values in frames.items()}
per_frame["route"] = np.unique(route_dirs[indices], return_inverse=True)[1]
per_frame["frame"] = dataset.route_indices[indices]
route_names = np.unique(route_dirs[indices])
scenarios = np.array([path.split("/")[-2] for path in route_names])

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
f = per_frame
s = f["l"] - f["H"]
nu = np.sqrt(np.maximum(f["nu2"], 1e-4))
z = s / nu
s_win = f["l_win"] - f["H_win"]
summary: dict = {"frames": len(s), "routes": len(route_names)}


def moments(mask: np.ndarray) -> dict:
    return {
        "n": int(mask.sum()),
        "mean_s": float(s[mask].mean()),
        "std_s": float(s[mask].std()),
        "mean_nu2": float(f["nu2"][mask].mean()),
        "mean_z": float(z[mask].mean()),
        "std_z": float(z[mask].std()),
        "mean_H": float(f["H"][mask].mean()),
        "mean_s_win": float(s_win[mask].mean()),
    }


everything = np.ones(len(s), bool)
summary["all"] = moments(everything)
summary["var_s_over_mean_nu2"] = float(s.var() / f["nu2"].mean())
summary["mean_l"] = float(f["l"].mean())
summary["perplexity"] = float(np.exp(f["l"].mean()))
summary["window"] = {
    "mean_l": float(f["l_win"].mean()),
    "mean_H": float(f["H_win"].mean()),
    "mean_s": float(s_win.mean()),
    "std_s": float(s_win.std()),
    "var_s_over_mean_nu2": float(s_win.var() / f["nu2_win"].mean()),
}

speed_bins = {"stopped <0.1": (-np.inf, 0.1), "slow 0.1-5": (0.1, 5), "medium 5-10": (5, 10), "fast >10": (10, np.inf)}
summary["speed_bins"] = {k: moments((f["speed"] >= lo) & (f["speed"] < hi)) for k, (lo, hi) in speed_bins.items()}
deciles = np.quantile(f["H"], np.linspace(0, 1, 11))
decile_of = np.clip(np.searchsorted(deciles, f["H"], side="right") - 1, 0, 9)
summary["entropy_deciles"] = [moments(decile_of == d) for d in range(10)]
summary["slope_s_on_H"] = float(np.polyfit(f["H"], s, 1)[0])
summary["slope_l_on_H"] = float(np.polyfit(f["H"], f["l"], 1)[0])
summary["scenarios"] = {name: moments(scenarios[f["route"]] == name) for name in np.unique(scenarios)}

# Calibration of the top-1 code and of the predictive sets.
bins = np.clip((f["top1_p"] * 10).astype(int), 0, 9)
summary["ece_top1"] = float(sum(abs(f["top1_p"][bins == b].sum() - f["top1_correct"][bins == b].sum()) for b in range(10)) / len(s))
summary["top1_accuracy"] = float(f["top1_correct"].mean())
# Randomized PIT of a discrete distribution: uniform on [0, 1] when calibrated.
pit = f["coverage_mass"] + np.random.default_rng(args.seed).uniform(size=len(s)) * np.exp(-f["l"])
summary["coverage"] = {str(a): float((pit < a).mean()) for a in (0.5, 0.8, 0.9, 0.95)}

# Drive level: S / sqrt(V) is ~N(0, 1) across normative drives if steps are a martingale.
drive_z = np.array([s[f["route"] == r].sum() / np.sqrt(f["nu2"][f["route"] == r].sum()) for r in range(len(route_names))])
summary["drive_Z"] = {"mean": float(drive_z.mean()), "std": float(drive_z.std()), "frac_abs_gt_2": float((abs(drive_z) > 2).mean())}
same_route = f["route"][1:] == f["route"][:-1]
summary["lag1_autocorr_s"] = float(np.corrcoef(s[1:][same_route], s[:-1][same_route])[0, 1])

# Is the entropy more than a function of the kinematics? R^2 of H on speed and |yaw rate| while moving.
moving = f["speed"] > 0.1
kinematics = np.stack([np.ones(moving.sum()), f["speed"][moving], f["speed"][moving] ** 2, abs(f["yaw_rate"][moving])], 1)
fit = np.linalg.lstsq(kinematics, f["H"][moving], rcond=None)[0]
summary["entropy_moving"] = {
    "var_H": float(f["H"][moving].var()),
    "r2_on_kinematics": float(1 - (f["H"][moving] - kinematics @ fit).var() / f["H"][moving].var()),
}
summary["residual"] = {"median": float(np.median(f["residual"])), "frac_gt_2tau": float((f["residual"] > 0.1).mean())}


def auroc(positive: np.ndarray, negative: np.ndarray) -> float:
    """P(score of an anomaly > score of a normal frame), ties counted half (Mann-Whitney)."""
    ranks = rankdata(np.concatenate([positive, negative]))[: len(positive)]
    return float((ranks.sum() - len(positive) * (len(positive) + 1) / 2) / (len(positive) * len(negative)))


# Synthetic anomalies against normal frames: separation of the step score s.
summary["anomalies"] = {}
for anomaly in ANOMALIES:
    s_anom = (f[f"l_{anomaly}"] - f["H"])[~np.isnan(f[f"l_{anomaly}"])]
    summary["anomalies"][anomaly] = {
        "n": len(s_anom),
        "mean_s": float(s_anom.mean()),
        "auroc": auroc(s_anom, s),
        "detected_at_1pct_false_alarms": float((s_anom > np.quantile(s, 0.99)).mean()),
        "detected_at_5pct_false_alarms": float((s_anom > np.quantile(s, 0.95)).mean()),
    }

name = args.name or f"{os.path.basename(os.path.normpath(args.checkpoint_dir))}_{os.path.basename(os.path.normpath(config.carla_root))}"
os.makedirs("outputs/surprisal", exist_ok=True)
np.savez_compressed(f"outputs/surprisal/{name}.npz", route_names=route_names, **per_frame)
with open(f"outputs/surprisal/{name}.json", "w") as out:
    json.dump({"args": vars(args), "summary": summary}, out, indent=2)

a = summary["all"]
print(f"step:   mean s {a['mean_s']:+.3f}  std s {a['std_s']:.3f}  Var s / mean nu2 {summary['var_s_over_mean_nu2']:.3f}  mean z {a['mean_z']:+.3f}  std z {a['std_z']:.3f}")
print(f"        mean l {summary['mean_l']:.3f}  mean H {a['mean_H']:.3f}  perplexity {summary['perplexity']:.1f}")
w = summary["window"]
print(f"window: mean s {w['mean_s']:+.3f}  std s {w['std_s']:.3f}  Var s / mean nu2 {w['var_s_over_mean_nu2']:.3f}  (l {w['mean_l']:.2f}, H {w['mean_H']:.2f})")
print(f"top-1 accuracy {summary['top1_accuracy']:.3f}  ECE {summary['ece_top1']:.4f}  coverage {summary['coverage']}")
d = summary["drive_Z"]
print(f"drive Z: mean {d['mean']:+.3f}  std {d['std']:.3f}  |Z|>2 {d['frac_abs_gt_2']:.3f}   lag-1 autocorr of s {summary['lag1_autocorr_s']:.3f}")
print(f"slope s~H {summary['slope_s_on_H']:+.3f}  l~H {summary['slope_l_on_H']:+.3f}")
e = summary["entropy_moving"]
print(f"moving: Var H {e['var_H']:.3f}  R^2 of H on speed/|yaw rate| {e['r2_on_kinematics']:.3f}")
print(f"residual: median {summary['residual']['median']:.3f}  frac > 2 tau {summary['residual']['frac_gt_2tau']:.3f}")
print(f"temperature {args.temperature}; synthetic anomalies vs normal s (threshold at the normal 99% / 95% quantile):")
for anomaly, m in summary["anomalies"].items():
    print(
        f"  {anomaly:13s} n {m['n']:6d}  mean s {m['mean_s']:+7.2f}  AUROC {m['auroc']:.3f}"
        f"  detected {m['detected_at_1pct_false_alarms']:.2f} / {m['detected_at_5pct_false_alarms']:.2f}",
    )
print(f"{'':16s}{'n':>7s}{'mean s':>8s}{'std s':>7s}{'nu2':>7s}{'mean z':>8s}{'H':>7s}{'s_win':>8s}")
rows = list(summary["speed_bins"].items()) + [(f"H decile {d}", m) for d, m in enumerate(summary["entropy_deciles"])]
for label, m in rows + list(summary["scenarios"].items()):
    print(f"{label[:16]:16s}{m['n']:7d}{m['mean_s']:+8.3f}{m['std_s']:7.3f}{m['mean_nu2']:7.3f}{m['mean_z']:+8.3f}{m['mean_H']:7.3f}{m['mean_s_win']:+8.3f}")
print("saved", f"outputs/surprisal/{name}.json")
