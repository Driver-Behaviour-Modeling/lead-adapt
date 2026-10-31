"""Controlled token-driving comparison, with optional frozen-model ablations.

Prepares a new isolated bundle by default; --run starts fresh CARLA processes.
Every selected route/variant gets one attempt. Five routes are selected by
default; --all-routes selects the full 220. No driving result is inferred from loss.
"""

import argparse
import json
import signal
from pathlib import Path

from scripts.diagnose_turns import (
    BASE_CONFIG,
    PROJECT_ROOT,
    check_prerequisites,
    evaluation_environment,
    file_hash,
    prepare_experiment,
    run_one,
    write_json,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--routes", nargs="+")
    selection.add_argument(
        "--all-routes",
        action="store_true",
        help="Select all 220 original Bench2Drive routes",
    )
    parser.add_argument(
        "--ablations",
        action="store_true",
        help="Also disable world fusion and token navigation separately",
    )
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--gpu", default="GPU-4ec20303-3c97-fc1c-9499-e5539efe3717")
    parser.add_argument("--graphics-adapter", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--boot-timeout", type=float, default=240)
    parser.add_argument("--route-timeout", type=float, default=900)
    parser.add_argument(
        "--python",
        default="/localstorage/home/f20221129/miniconda3/envs/lead/bin/python",
    )
    parser.add_argument(
        "--carla-root", type=Path, default=PROJECT_ROOT / "3rd_party/CARLA_0915",
    )
    args = parser.parse_args()
    if args.all_routes:
        args.routes = sorted(
            (
                path.stem
                for path in (PROJECT_ROOT / "data/benchmark_routes/bench2drive").glob(
                    "*.xml",
                )
            ),
            key=int,
        )
        if len(args.routes) != 220:
            parser.error("Full benchmark selection requires exactly 220 route XMLs")
    elif args.routes is None:
        args.routes = ["17655", "17569", "17563", "17635", "17598"]
    if args.seed < 0 or args.seed >= 2**32 or args.cpu_threads <= 0:
        parser.error("Use a nonnegative 32-bit seed and positive CPU thread count")
    if args.boot_timeout <= 0 or args.route_timeout <= 0:
        parser.error("Timeouts must be positive")
    checkpoint = args.checkpoint.resolve(strict=True)
    values = json.loads((checkpoint.parent / "config.json").read_text())
    if not values.get("adapt_history_free") or not values.get(
        "adapt_waypoints_from_tokens",
    ):
        raise ValueError("Comparison requires the history-free token trajectory model")
    if args.ablations and not (
        values.get("adapt_world_model")
        and values.get("adapt_token_navigation_conditioning")
    ):
        raise ValueError("Both combined-model branches are required for ablations")
    args.output = args.output.resolve()
    args.carla_root = args.carla_root.resolve()
    manifest = prepare_experiment(
        args.output,
        checkpoint,
        PROJECT_ROOT / "data/benchmark_routes/bench2drive",
        list(args.routes),
        ["baseline"],
        seed=args.seed,
    )
    manifest["source"]["files"]["scripts/evaluate_integrated_driving.py"] = file_hash(
        Path(__file__).resolve(),
    )
    # Only decoding changes. Keep the original config/hash and record the exact
    # inference config separately; architecture/weights remain strictly loaded.
    values.update(adapt_decoding="beam", adapt_beam_width=8)
    staged_config = args.output / "checkpoint/config.json"
    write_json(staged_config, values)
    manifest["effective_training_config"] = {
        "path": str(staged_config),
        "sha256": file_hash(staged_config),
        "overrides": {"adapt_decoding": "beam", "adapt_beam_width": 8},
    }
    common = BASE_CONFIG | {
        "steer_modality": "waypoints",
        "throttle_modality": "waypoints",
        "brake_modality": "waypoints",
        "adapt_disable_world_fusion": "false",
        "adapt_disable_token_navigation": "false",
    }
    variants = {"baseline": common}
    if args.ablations:
        variants.update(
            no_world=common | {"adapt_disable_world_fusion": "true"},
            no_token_navigation=common | {"adapt_disable_token_navigation": "true"},
        )
    manifest["variants"] = {
        name: " ".join(f"{key}={value}" for key, value in flags.items())
        for name, flags in variants.items()
    }
    manifest["interpretation"] = (
        "Paired route/seed token-control diagnostic. Frozen-model ablations do not establish matched-training architectural attribution. Missing attempts contribute zero to the planned-subset mean."
    )
    manifest["evaluation_scope"] = {
        "benchmark": "Bench2Drive",
        "planned_routes": len(args.routes),
        "full_benchmark": args.all_routes,
        "denominator": len(args.routes),
    }
    if args.all_routes:
        manifest["interpretation"] = (
            "All 220 original Bench2Drive routes with token control, one attempt each. "
            "Report missing/failed attempts and the fixed denominator of 220. "
            "Frozen-model ablations do not establish matched-training architectural attribution."
        )
    env = evaluation_environment(
        manifest, args.gpu, args.carla_root, args.python, args.cpu_threads,
    )
    manifest["prerequisites"] = check_prerequisites(
        args.python, args.carla_root, env, args.cpu_threads,
    )
    manifest["execution"] = {
        "cuda_device": args.gpu,
        "graphics_adapter": args.graphics_adapter,
        "cpu_threads": args.cpu_threads,
        "mode": "run" if args.run else "dry_run",
    }
    write_json(args.output / "manifest.json", manifest)
    if not manifest["prerequisites"]["ready"]:
        raise RuntimeError(
            "Prerequisites failed; see manifest.json. No simulator started.",
        )
    if not args.run:
        print(
            f"Prepared {len(args.routes)} routes × {len(variants)} variants at {args.output}; no simulator started.",
        )
        return 0
    manifest["status"] = "running"
    write_json(args.output / "manifest.json", manifest)
    results = []

    def terminate(_signal, _frame):
        raise KeyboardInterrupt

    previous = signal.signal(signal.SIGTERM, terminate)
    try:
        for route in args.routes:
            for variant in variants:
                print(f"Running {route}: {variant}, one attempt", flush=True)
                result = run_one(manifest, variant, route, args, env)
                results.append(result)
                write_json(args.output / "summary.json", results)
                if (
                    result["status"] != "evaluator_finished"
                    or not result["has_route_result"]
                    or result["diagnostics_step_records"] == 0
                ):
                    raise RuntimeError(
                        "Execution/output error; stopped without an automatic retry",
                    )
        manifest["status"] = "completed"
    except BaseException:
        manifest["status"] = "interrupted_or_failed"
        raise
    finally:
        signal.signal(signal.SIGTERM, previous)
        write_json(args.output / "manifest.json", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
