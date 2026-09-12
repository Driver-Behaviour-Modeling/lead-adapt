"""Short, isolated CARLA training benchmark; never saves model checkpoints.

Run from the repository root with ``python -m scripts.profile_training --help``.
Use a fresh process per configuration to isolate allocator and in-process cache
state. Persistent compiler caches can still shorten warmup across processes.
This is a single-GPU probe.
"""

import argparse
import json
import os
import random
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--epoch", type=int, default=39)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--compile-scope", choices=["backbone", "model"], default="backbone",
    )
    parser.add_argument("--compile-mode", default="default")
    parser.add_argument("--logits-upsample", action="store_true")
    parser.add_argument("--nearest", action="store_true")
    parser.add_argument("--prefetch", action="store_true")
    parser.add_argument("--gpu-augmentation", action="store_true")
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Export a separate diagnostic step after timing",
    )
    args = parser.parse_args()
    if os.environ.get("LEAD_TRAINING_CONFIG", "").strip():
        parser.error(
            "Unset LEAD_TRAINING_CONFIG so environment overrides cannot relabel this benchmark",
        )
    if min(args.batch_size, args.steps) < 1 or min(args.workers, args.warmup) < 0:
        parser.error("batch size/steps must be positive; workers/warmup nonnegative")
    if torch.cuda.device_count() != 1:
        parser.error("Select one GPU with CUDA_VISIBLE_DEVICES before running")
    return args


def timed_event():
    event = torch.cuda.Event(enable_timing=True)
    event.record()
    return event


def main():
    args = parse_args()
    # Imports after parsing keep --help quick and make the single-GPU contract clear.
    from lead.common.constants import SourceDataset
    from lead.data_loader.carla_dataset import CARLAData
    from lead.training import training_utils
    from lead.training.config_training import TrainingConfig
    from lead.training.input_pipeline import prepared_training_batches
    from lead.training.mixed_training_utils import mixed_data_collate_fn

    os.environ.setdefault("LEAD_PROJECT_ROOT", str(Path.cwd()))
    values = json.loads(args.config.read_text())
    values.update(
        compile=args.compile,
        compile_scope=args.compile_scope,
        compile_mode=args.compile_mode,
        upsample_perspective_logits=args.logits_upsample,
        upsample_mode="nearest" if args.nearest else "bilinear",
        gpu_color_augmentation=args.gpu_augmentation,
        cuda_prefetch=args.prefetch,
        batch_size=args.batch_size,
        prefetch_factor=2,
        carla_num_samples=args.batch_size * (args.warmup + args.steps + 2),
        use_training_session_cache=False,
        force_rebuild_data_cache=False,
        force_rebuild_bucket=False,
        use_color_aug=args.gpu_augmentation,
        use_sensor_perburtation=False,
        visualize_training=False,
        continue_failed_training=False,
        load_file=str(args.checkpoint.resolve()) if args.checkpoint else None,
    )
    config = TrainingConfig(dict(values))
    if not config.use_carla_data or config.use_navsim_data or config.use_waymo_e2e_data:
        raise ValueError("This benchmark currently supports CARLA-only runs")
    for key, expected in values.items():
        if key in {
            "compile",
            "upsample_mode",
            "gpu_color_augmentation",
            "cuda_prefetch",
        }:
            if getattr(config, key) != expected:
                raise ValueError(
                    f"Environment overrides the benchmark's {key}; unset LEAD_TRAINING_CONFIG",
                )
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    started = time.perf_counter()
    dataset = CARLAData(config.carla_data, config, training_session_cache=None)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        prefetch_factor=2 if args.workers else None,
        collate_fn=mixed_data_collate_fn,
        drop_last=True,
        worker_init_fn=training_utils.seed_worker,
    )
    model, _ = training_utils.initialize_model(config)
    model.train()
    if hasattr(model, "adapt_decoder"):
        decoder = model.adapt_decoder
        if config.use_scheduled_sampling:
            fraction = max(
                0.0,
                (args.epoch - config.scheduled_sampling_start_epoch)
                / max(1, config.scheduled_sampling_warmup_epochs),
            )
            decoder._ss_prob = max(
                min(
                    config.scheduled_sampling_max_prob,
                    config.scheduled_sampling_max_prob * fraction,
                ),
                min(
                    config.scheduled_sampling_min_prob,
                    config.scheduled_sampling_max_prob,
                ),
            )
    weights = config.detailed_loss_weights(SourceDataset.CARLA, args.epoch)
    weight_sum = sum(weights.values())
    weights = {key: value / weight_sum for key, value in weights.items()}
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        amsgrad=True,
        fused=True,
    )
    setup_seconds = time.perf_counter() - started

    def step(batch, index):
        batch["iteration"] = index
        batch["training_step"] = index
        events = [timed_event()]
        with (
            torch.profiler.record_function("forward"),
            torch.autocast(
                "cuda",
                dtype=config.torch_float_type,
                enabled=config.use_mixed_precision_training,
            ),
        ):
            prediction = model(data=batch)
        events.append(timed_event())
        with (
            torch.profiler.record_function("loss"),
            torch.autocast(
                "cuda",
                dtype=config.torch_float_type,
                enabled=config.use_mixed_precision_training,
            ),
        ):
            losses, _ = model.compute_loss(predictions=prediction, data=batch)
            loss = sum(weights[key] * value for key, value in losses.items())
        events.append(timed_event())
        with torch.profiler.record_function("backward"):
            loss.backward()
        events.append(timed_event())
        with torch.profiler.record_function("optimizer"):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        events.append(timed_event())
        return loss.detach(), events

    rows = []
    warmup_seconds = 0.0
    torch.cuda.reset_peak_memory_stats()
    with prepared_training_batches(loader, config) as batches:
        for index in range(args.warmup + args.steps):
            started = time.perf_counter()
            batch = next(batches)
            loaded = time.perf_counter()
            loss, events = step(batch, index)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite training loss at step {index}")
            if index < args.warmup:
                warmup_seconds += elapsed
                if index + 1 == args.warmup:
                    torch.cuda.reset_peak_memory_stats()
            else:
                rows.append(
                    dict(
                        wall_ms=elapsed * 1000,
                        loader_and_enqueue_ms=(loaded - started) * 1000,
                        forward_ms=events[0].elapsed_time(events[1]),
                        loss_ms=events[1].elapsed_time(events[2]),
                        backward_ms=events[2].elapsed_time(events[3]),
                        optimizer_ms=events[3].elapsed_time(events[4]),
                        loss=float(loss),
                    ),
                )
            print(
                f"step {index + 1}/{args.warmup + args.steps}: {elapsed:.3f}s loss={float(loss):.5f}",
                flush=True,
            )

        timed_peak_memory_gib = torch.cuda.max_memory_allocated() / 1024**3
        if args.trace:
            # Component scopes are installed only for this diagnostic step; they
            # do not perturb the timed runs above.
            handles, scopes = [], {}
            for name, module in model.named_children():

                def begin(_module, _inputs, label=name):
                    scope = torch.profiler.record_function(f"component/{label}")
                    scope.__enter__()
                    scopes[label] = scope

                def end(_module, _inputs, _output, label=name):
                    scopes.pop(label).__exit__(None, None, None)

                handles.extend(
                    [
                        module.register_forward_pre_hook(begin),
                        module.register_forward_hook(end, always_call=True),
                    ],
                )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            try:
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                ) as profiler:
                    batch = next(batches)
                    step(batch, args.warmup + args.steps)
                    torch.cuda.synchronize()
                profiler.export_chrome_trace(
                    str(args.output.with_suffix(".trace.json")),
                )
                args.output.with_suffix(".operators.txt").write_text(
                    profiler.key_averages().table(
                        sort_by="self_cuda_time_total", row_limit=40,
                    ),
                )
            finally:
                for handle in handles:
                    handle.remove()

    summary = {key: statistics.mean(row[key] for row in rows) for key in rows[0]}
    report = dict(
        gpu=torch.cuda.get_device_name(),
        torch_version=torch.__version__,
        config=str(args.config.resolve()),
        checkpoint=str(args.checkpoint.resolve()) if args.checkpoint else None,
        settings={
            key: value
            for key, value in vars(args).items()
            if not isinstance(value, Path)
        },
        scheduled_sampling_probability=getattr(
            getattr(model, "adapt_decoder", None), "_ss_prob", None,
        ),
        setup_seconds=setup_seconds,
        warmup_seconds=warmup_seconds,
        summary=summary,
        samples_per_second=args.batch_size * 1000 / summary["wall_ms"],
        peak_gpu_memory_gib=timed_peak_memory_gib,
        steps=rows,
        notes="One GPU, fixed seed, no checkpoint writes. Sensor-view augmentation disabled. Default image augmentation disabled; --gpu-augmentation enables the alternative recipe. Per-step synchronization makes these diagnostic timings rather than maximum throughput. Loader time includes host enqueue work; device-copy time may overlap other work.",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
