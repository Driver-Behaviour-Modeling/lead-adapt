"""Open-loop error of ADAPT token rollouts against teacher forcing.

Separates the sources of the token rollout's driving error on a fixed random
subset of post-train frames (augmentation off). Per horizon step (0.25 s ... 2 s)
it reports the position error in metres of:

- gt_tokens:   ground-truth future tokens integrated (codebook quantization floor)
- teacher:     argmax of each step given the true prefix (one-step model error)
- greedy/beam/expected: the autoregressive rollout under each decoding
- reference:   another checkpoint's own pred_future_waypoints (e.g. navcond's MLP)

plus token accuracy against the ground-truth tokens and the teacher-forced NLL.
Exposure bias shows as rollout error matching teacher forcing at step 1 and
diverging with the horizon; greedy myopia as beam closing that gap.

Usage (repo root, LEAD_TRAINING_CONFIG unset):
    python scripts/eval_open_loop_rollout.py \
        --checkpoint_dir outputs/checkpoints/adapt_token_full_v1 \
        --reference_dir outputs/checkpoints/adapt_radar_navcond_v1_retry1
"""

import argparse
import glob
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from lead.adapt.adapt_decoder import integrate_body_frame_deltas
from lead.data_loader.carla_dataset import CARLAData
from lead.training.config_training import TrainingConfig
from lead.training.input_pipeline import prepared_training_batches
from lead.training.mixed_training_utils import mixed_data_collate_fn
from lead.training.training_utils import create_model

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_dir", required=True)
parser.add_argument("--reference_dir", default=None)
parser.add_argument("--decodings", nargs="+", default=["greedy", "beam", "expected"])
parser.add_argument("--beam_width", type=int, default=8)
parser.add_argument("--num_samples", type=int, default=2048)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--num_workers", type=int, default=16)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

# Deterministic inputs: no sensor-perturbation draw, no colour augmentation.
OVERRIDES = {"use_sensor_perburtation": False, "use_color_aug": False}


def load(checkpoint_dir: str) -> tuple[TrainingConfig, torch.nn.Module]:
    with open(os.path.join(checkpoint_dir, "config.json")) as f:
        loaded = json.load(f)
    loaded.update(OVERRIDES)
    config = TrainingConfig(loaded, raise_error_on_missing_key=False)
    (weights,) = glob.glob(os.path.join(checkpoint_dir, "model*.pth"))
    model = create_model(config)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    return config, model.cuda().eval()


config, model = load(args.checkpoint_dir)
decoder = model.adapt_decoder
reference = load(args.reference_dir)[1] if args.reference_dir else None

dataset = CARLAData(root=config.carla_data, config=config, training_session_cache=None, build_cache=False)
indices = np.random.default_rng(args.seed).choice(len(dataset), args.num_samples, replace=False)
loader = DataLoader(
    Subset(dataset, indices.tolist()),
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    collate_fn=mixed_data_collate_fn,
)

errors: dict[str, list[torch.Tensor]] = {}
accuracy: dict[str, list[torch.Tensor]] = {}
nll: list[torch.Tensor] = []


def add(store: dict, name: str, value: torch.Tensor) -> None:
    store.setdefault(name, []).append(value.float().cpu())


def position_error(waypoints: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(waypoints[..., :2].float() - gt, dim=-1)


with prepared_training_batches(loader, config) as batches, torch.inference_mode():
    for data in batches:
        with torch.amp.autocast("cuda", dtype=config.torch_float_type, enabled=config.use_mixed_precision_training):
            gt = data["future_waypoints"][:, : config.num_way_points_prediction, :2].float().cuda()
            start = torch.cat([data["past_positions"][:, -1, :2], data["past_yaws"][:, -1:]], dim=-1).float().cuda()

            # Teacher forcing: the decoder takes its true-prefix path while all
            # submodules (dropout, BatchNorm) stay in eval mode.
            decoder.training = True
            model(data=data)
            outputs = model._adapt_outputs
            decoder.training = False
            gt_tokens = outputs["future_token_ids"]
            logits = outputs["output_logits"].float()
            nll.append(F.cross_entropy(logits.transpose(1, 2), gt_tokens, reduction="none").cpu())
            add(accuracy, "teacher", logits.argmax(-1) == gt_tokens)
            add(errors, "teacher", position_error(outputs["trajectory"], gt))
            floor = integrate_body_frame_deltas(decoder.kdisks_model.decode(gt_tokens).float(), start)
            add(errors, "gt_tokens", position_error(floor, gt))

            for decoding in args.decodings:
                decoder.decoding = decoding
                decoder.beam_width = args.beam_width
                prediction = model(data=data)
                add(accuracy, decoding, model._adapt_outputs["future_token_ids"] == gt_tokens)
                add(errors, decoding, position_error(prediction.pred_future_waypoints, gt))

            if reference is not None:
                add(errors, "reference", position_error(reference(data=data).pred_future_waypoints, gt))

steps = config.num_way_points_prediction
print(f"{len(indices)} frames from {args.checkpoint_dir}; error in metres at each horizon step (x 0.25 s)")
print(f"{'':12s}" + "".join(f"{f't+{k + 1}':>7s}" for k in range(steps)) + f"{'ADE':>7s}{'FDE':>7s}")
summary = {}
for name, values in errors.items():
    per_step = torch.cat(values).mean(0)
    summary[name] = {"per_step_m": per_step.tolist(), "ade_m": per_step.mean().item(), "fde_m": per_step[-1].item()}
    print(f"{name:12s}" + "".join(f"{v:7.3f}" for v in per_step) + f"{per_step.mean():7.3f}{per_step[-1]:7.3f}")
print("token accuracy per step")
for name, values in accuracy.items():
    per_step = torch.cat(values).mean(0)
    summary[name]["token_accuracy"] = per_step.tolist()
    print(f"{name:12s}" + "".join(f"{v:7.3f}" for v in per_step))
per_step_nll = torch.cat(nll).mean(0)
summary["teacher"]["nll"] = per_step_nll.tolist()
print(f"{'teacher NLL':12s}" + "".join(f"{v:7.3f}" for v in per_step_nll))

os.makedirs("outputs/open_loop", exist_ok=True)
out_path = f"outputs/open_loop/{os.path.basename(os.path.normpath(args.checkpoint_dir))}.json"
with open(out_path, "w") as f:
    json.dump({"args": vars(args), "summary": summary}, f, indent=2)
print("saved", out_path)
