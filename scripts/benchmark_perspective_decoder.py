"""Benchmark semantic + depth heads only; does not load data or train a policy.

Example:
    python scripts/benchmark_perspective_decoder.py --device cuda:1 --batch-size 4

Each measurement includes both forwards, CE/L1 losses and their backwards. The
optional paths change auxiliary predictions; timing is not an accuracy claim.
"""

import argparse
import contextlib
import json
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from lead.adapt.perspective_decoder import PerspectiveDecoder as AdaptDecoder
from lead.common.constants import SourceDataset
from lead.tfv6.perspective_decoder import PerspectiveDecoder as TransfuserDecoder
from lead.training.config_training import TrainingConfig


class BenchmarkConfig(TrainingConfig):
    """Fix CARLA panorama geometry without environment or CLI config overrides."""

    final_image_height = 384
    final_image_width = 1152
    upsample_perspective_logits = False
    upsample_mode = "bilinear"

    def __init__(self):
        pass


def measure(args, decoder_class, upsample_logits, mode, precision):
    device = torch.device(args.device)
    torch.cuda.empty_cache()
    config = BenchmarkConfig()
    config.upsample_perspective_logits = upsample_logits
    config.upsample_mode = mode
    torch.manual_seed(123)
    heads = (
        nn.ModuleList(
            [
                decoder_class(
                    config=config,
                    in_channels=512,
                    out_channels=out_channels,
                    perspective_upsample_factor=32,
                    modality=modality,
                    device=device,
                    source_data=int(SourceDataset.CARLA),
                )
                for modality, out_channels in (
                    ("semantic", config.num_semantic_classes),
                    ("depth", 1),
                )
            ],
        )
        .to(device)
        .train()
    )
    features = torch.randn(
        args.batch_size,
        512,
        12,
        36,
        device=device,
        requires_grad=True,
    )
    label_shape = (args.batch_size, 384, 1152)
    semantic_label = torch.randint(
        config.num_semantic_classes,
        label_shape,
        device=device,
    )
    depth_label = torch.rand(label_shape, device=device)

    def train_step():
        amp = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if precision == "bf16"
            else contextlib.nullcontext()
        )
        with amp:
            semantic = heads[0]({}, features, {})
            depth = heads[1]({}, features, {})
        loss = F.cross_entropy(semantic.float(), semantic_label)
        loss = loss + F.l1_loss(depth.float(), depth_label)
        loss.backward()

    for _ in range(args.warmup):
        heads.zero_grad(set_to_none=True)
        features.grad = None
        train_step()
    heads.zero_grad(set_to_none=True)
    features.grad = None
    torch.cuda.synchronize(device)
    baseline_bytes = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    elapsed_ms = []
    for _ in range(args.steps):
        heads.zero_grad(set_to_none=True)
        features.grad = None
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        train_step()
        end.record()
        end.synchronize()
        elapsed_ms.append(start.elapsed_time(end))
    peak_bytes = torch.cuda.max_memory_allocated(device)
    all_gradients_finite = bool(torch.isfinite(features.grad).all()) and all(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in heads.parameters()
    )
    result = {
        "upsample_perspective_logits": upsample_logits,
        "upsample_mode": mode,
        "precision": precision,
        "median_forward_loss_backward_ms": statistics.median(elapsed_ms),
        "min_forward_loss_backward_ms": min(elapsed_ms),
        "max_forward_loss_backward_ms": max(elapsed_ms),
        "peak_allocated_mib": peak_bytes / 2**20,
        "incremental_peak_allocated_mib": (peak_bytes - baseline_bytes) / 2**20,
        "all_gradients_finite": all_gradients_finite,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--implementation", choices=("adapt", "tfv6"), default="adapt")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.batch_size < 1 or args.steps < 1 or args.warmup < 1:
        parser.error("batch-size, steps and warmup must all be positive")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("a CUDA device is required for GPU timings")
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    decoder_class = (
        AdaptDecoder if args.implementation == "adapt" else TransfuserDecoder
    )
    report = {
        "scope": "isolated semantic and depth heads; forward + loss + backward",
        "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(device),
        "device": str(device),
        "implementation": args.implementation,
        "batch_size": args.batch_size,
        "input_shape": [args.batch_size, 512, 12, 36],
        "output_height_width": [384, 1152],
        "semantic_classes": BenchmarkConfig.num_semantic_classes,
        "steps": args.steps,
        "warmup": args.warmup,
        "cudnn_benchmark": True,
        "tf32_allowed": True,
        "results": [],
    }
    for precision in ("fp32", "bf16"):
        for upsample_logits in (False, True):
            for mode in ("bilinear", "nearest"):
                result = measure(args, decoder_class, upsample_logits, mode, precision)
                report["results"].append(result)
                print(json.dumps(result), flush=True)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Saved benchmark report: {args.output}")


if __name__ == "__main__":
    main()
