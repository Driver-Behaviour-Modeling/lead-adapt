"""Capture ego surprise and predicted-world interaction evidence on normal/evaluation drives.

Outputs are raw features, not calibrated anomaly verdicts. Fit temperatures and
smoothing on normal validation drives, then thresholds on separate normal
calibration drives. Independent models must share the body-frame vocabulary.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from lead.adapt.transfuser_utils import patch_norm_fp32
from lead.data_loader.carla_dataset import CARLAData
from lead.scoring.reference import NormativeTokenScorer
from lead.scoring.world_interactions import forecast_error_sums, interaction_features
from lead.training.config_training import TrainingConfig
from lead.training.input_pipeline import prepared_training_batches
from lead.training.mixed_training_utils import mixed_data_collate_fn
from lead.training.training_utils import create_model


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--temperatures", type=float, nargs="+")
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, required=True, help="New filename prefix (.npz/.json)",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--max-frames", type=int, default=0, help="Diagnostic cap; 0 scores all frames",
    )
    parser.add_argument(
        "--save-member-logits",
        action="store_true",
        help="Save raw logits for normal-validation temperature fitting",
    )
    parser.add_argument(
        "--save-world-futures",
        action="store_true",
        help="Save predicted traffic and observed ego poses for offline interaction experiments",
    )
    parser.add_argument(
        "--forecast-metrics",
        action="store_true",
        help="Measure held-out actor forecast errors using current-state matching",
    )
    args = parser.parse_args()
    if os.environ.get("LEAD_TRAINING_CONFIG"):
        raise ValueError("Unset LEAD_TRAINING_CONFIG to avoid changing model identity")
    if any(args.output.with_suffix(suffix).exists() for suffix in (".json", ".npz")):
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    models, configs, identities = [], [], []
    centroids = None
    for directory in args.checkpoint_dirs:
        weights = list(directory.glob("model*.pth"))
        if len(weights) != 1:
            raise ValueError(f"Expected exactly one model checkpoint in {directory}")
        values = json.loads((directory / "config.json").read_text())
        if not all(
            values.get(key)
            for key in (
                "adapt_history_free",
                "adapt_rolling_causal_first",
                "adapt_world_model",
            )
        ):
            raise ValueError(
                "Reference requires the trained history-free, causal-first, world-model configuration",
            )
        config = TrainingConfig(values)
        model = patch_norm_fp32(create_model(config)).cuda().eval()
        model.load_state_dict(
            torch.load(weights[0], map_location="cpu", weights_only=True), strict=True,
        )
        codes = model.adapt_decoder.kdisks_model.codebook.centroids
        if centroids is None:
            centroids = codes.detach().clone()
        elif not torch.equal(centroids, codes):
            raise ValueError("Ensemble members have different vocabulary centroids")
        if configs:
            for key in (
                "num_way_points_prediction",
                "world_num_steps",
                "world_step_seconds",
                "kdisks_heading_weight",
            ):
                if getattr(config, key) != getattr(configs[0], key):
                    raise ValueError(f"Ensemble members disagree on {key}")
        identities.append(
            {
                "weights": str(weights[0].resolve()),
                "sha256": digest(weights[0]),
                "seed": config.seed,
            },
        )
        configs.append(config)
        models.append(model)
    if len({row["sha256"] for row in identities}) != len(models):
        raise ValueError("Duplicate weights are not independent ensemble members")
    if len(models) > 1 and len({row["seed"] for row in identities}) != len(models):
        raise ValueError("Ensemble members must have distinct training seeds")
    temperatures = args.temperatures or [1.0] * len(models)
    if len(temperatures) != len(models):
        raise ValueError("Provide one temperature per independent ensemble member")
    scorer = NormativeTokenScorer(
        centroids,
        configs[0].kdisks_heading_weight,
        temperatures,
        epsilon=args.epsilon,
        sigma=args.sigma,
    ).cuda()
    values = json.loads((args.checkpoint_dirs[0] / "config.json").read_text())
    values.update(
        carla_root=str(args.data_root.resolve()),
        carla_data=str((args.data_root / "data").resolve()),
        use_sensor_perburtation=False,
        use_color_aug=False,
        use_persistent_cache=False,
        cuda_prefetch=True,
    )
    data_config = TrainingConfig(values)
    dataset = CARLAData(
        data_config.carla_data,
        data_config,
        training_session_cache=None,
        build_cache=False,
        random=False,
    )
    route_names = np.array(
        [str(Path(route).resolve()) for route in dataset.route_dirs.astype(str)],
    )
    expected_root = (args.data_root / "data").resolve()
    if any(
        not Path(route).is_relative_to(expected_root)
        for route in np.unique(route_names)
    ):
        raise ValueError("Cached dataset paths are outside the requested data root")
    indices = np.lexsort((dataset.route_indices, route_names))
    if args.max_frames:
        indices = indices[: args.max_frames]
    loader = DataLoader(
        Subset(dataset, indices.tolist()),
        batch_size=args.batch_size,
        num_workers=args.workers,
        collate_fn=mixed_data_collate_fn,
    )
    frames, logits_saved = {}, []
    world_saved = {
        key: []
        for key in ("future_positions", "future_velocities", "memory_confidence")
    }
    forecast_saved = {}

    def add(name, value):
        if value.dtype == torch.bfloat16:
            value = value.float()
        frames.setdefault(name, []).append(value.detach().cpu().numpy())

    with (
        prepared_training_batches(loader, data_config) as iterator,
        torch.inference_mode(),
    ):
        for data in iterator:
            member_logits, world_risk = [], []
            world_batch = {key: [] for key in world_saved}
            forecast_batch = {}
            ego = torch.cat(
                [data["future_waypoints"], data["future_yaws"][..., None]], dim=-1,
            )
            for model, config in zip(models, configs, strict=True):
                decoder = model.adapt_decoder
                # Use the fast teacher-forced path; causal masking keeps first
                # logits independent of the observed future label/prefix.
                decoder.training = True
                try:
                    with torch.autocast(
                        "cuda",
                        dtype=config.torch_float_type,
                        enabled=config.use_mixed_precision_training,
                    ):
                        prediction = model(data)
                finally:
                    decoder.training = False
                member_logits.append(
                    model._adapt_outputs["output_logits"][:, 0].float(),
                )
                world_risk.append(
                    interaction_features(ego, prediction.pred_actor_futures)[
                        "expected_conflicting_actor_count"
                    ],
                )
                if args.save_world_futures:
                    for key in world_batch:
                        world_batch[key].append(
                            prediction.pred_actor_futures[key].float().cpu().numpy(),
                        )
                if args.forecast_metrics:
                    # Match the detector's training precision context: its
                    # cost calculation promotes to fp32 under autocast before
                    # passing the current-state cost matrix to SciPy.
                    with torch.autocast(
                        "cuda",
                        dtype=config.torch_float_type,
                        enabled=config.use_mixed_precision_training,
                    ):
                        matching = model.radar_detector.compute_loss(
                            pred=prediction.pred_radar_predictions,
                            data=data,
                            loss={},
                            log={},
                            return_matching=True,
                        )
                    for key, value in forecast_error_sums(
                        prediction.pred_actor_futures, data, matching,
                    ).items():
                        forecast_batch.setdefault(key, []).append(value.cpu().numpy())
            logits = torch.stack(member_logits)
            score = scorer(logits, ego[:, 0])
            for key, value in score.items():
                if key != "log_probability":
                    add(key, value)
            add("world_conflict_count", torch.stack(world_risk).mean(0))
            if args.save_member_logits:
                logits_saved.append(logits.cpu().numpy())
            if args.save_world_futures:
                add("ego_future", ego.float())
                add("ego_speed", data["speed"].float().reshape(-1))
                for key, value in world_batch.items():
                    world_saved[key].append(np.stack(value))
            if args.forecast_metrics:
                for key, value in forecast_batch.items():
                    forecast_saved.setdefault(key, []).append(np.stack(value))
            print(
                f"Scored {sum(len(x) for x in frames['score'])}/{len(indices)} frames",
                flush=True,
            )
    output = {key: np.concatenate(value) for key, value in frames.items()}
    output["route"] = route_names[indices]
    output["frame"] = dataset.route_indices[indices]
    if logits_saved:
        output["member_logits"] = np.concatenate(logits_saved, axis=1)
        output["vocabulary_centroids"] = centroids.cpu().numpy()
    if args.save_world_futures:
        for key, value in world_saved.items():
            output[f"world_{key}"] = np.concatenate(value, axis=1)
    for key, value in forecast_saved.items():
        output[f"forecast_{key}"] = np.concatenate(value, axis=1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output.with_suffix(".npz"), **output)
    metadata = {
        "members": identities,
        "temperatures": temperatures,
        "epsilon": args.epsilon,
        "sigma": args.sigma,
        "data_root": str(args.data_root.resolve()),
        "vocabulary_sha256": hashlib.sha256(
            centroids.cpu().numpy().tobytes(),
        ).hexdigest(),
        "heading_weight": configs[0].kdisks_heading_weight,
        "single_member_has_no_epistemic_estimate": len(models) == 1,
        "frames": len(indices),
        "calibrated": False,
        "saved_world_futures": args.save_world_futures,
        "saved_forecast_metrics": args.forecast_metrics,
        "world_evidence": "Expected conflicting-actor count using assumed actor extents; a geometric proxy, not a likelihood or causal blame.",
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
