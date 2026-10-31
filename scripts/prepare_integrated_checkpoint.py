"""Create a reviewable combined checkpoint from history-free ego and vehicle weights.

Only the world predictor is transplanted from the older driving model. Its ego
decoder/codebook and learned fusion weights are not imported. The new navigation
and world-fusion residuals start at identity, preserving the ego model's outputs.
This is initialization for training, not a newly evaluated driving checkpoint.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from lead.training.config_training import TrainingConfig
from lead.training.training_utils import create_model


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ego-checkpoint", type=Path, required=True)
    parser.add_argument("--world-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if os.environ.get("LEAD_TRAINING_CONFIG"):
        raise ValueError(
            "Unset LEAD_TRAINING_CONFIG when preparing checkpoint metadata",
        )
    if args.output.exists():
        raise FileExistsError(args.output)
    ego_config = json.loads(args.ego_checkpoint.with_name("config.json").read_text())
    world_config = json.loads(
        args.world_checkpoint.with_name("config.json").read_text(),
    )
    if not all(
        ego_config.get(key)
        for key in (
            "adapt_history_free",
            "adapt_waypoints_from_tokens",
            "adapt_rolling_tokenization",
        )
    ):
        raise ValueError(
            "Ego checkpoint must be the history-free, rolling, token-driving model",
        )
    if not world_config.get("adapt_world_model"):
        raise ValueError("World checkpoint does not contain a trained world model")
    if world_config["world_num_steps"] != ego_config["num_way_points_prediction"]:
        raise ValueError("World and ego forecast horizons differ")
    if (
        abs(
            world_config["world_step_seconds"]
            - ego_config["waypoints_spacing"] / TrainingConfig(ego_config).carla_fps,
        )
        > 1e-8
    ):
        raise ValueError("World and ego forecast spacings differ")
    config = dict(ego_config)
    for key, value in world_config.items():
        if key.startswith("world_"):
            config[key] = value
    config.update(
        adapt_navigation_conditioning=True,
        adapt_token_navigation_conditioning=True,
        adapt_world_model=True,
        adapt_rolling_causal_first=True,
        adapt_train_token_decoder_only=False,
        use_scheduled_sampling=False,
        use_soft_ce=False,
        continue_failed_training=False,
        load_file=None,
        use_radars=True,
        radar_detection=True,
        use_radar_detection=True,
        detect_boxes=True,
        seed=args.seed,
    )
    torch.manual_seed(args.seed)
    model = create_model(TrainingConfig(config))
    ego_state = torch.load(args.ego_checkpoint, map_location="cpu", weights_only=True)
    result = model.load_state_dict(ego_state, strict=False)
    prefixes = (
        "world_model.",
        "adapt_decoder._world_fusion.",
        "adapt_decoder._token_navigation_modulation.",
    )
    if result.unexpected_keys or any(
        not key.startswith(prefixes) for key in result.missing_keys
    ):
        raise ValueError(f"Unexpected ego checkpoint incompatibility: {result}")
    world_state = torch.load(
        args.world_checkpoint, map_location="cpu", weights_only=True,
    )
    world_weights = {
        key.removeprefix("world_model."): value
        for key, value in world_state.items()
        if key.startswith("world_model.")
    }
    model.world_model.load_state_dict(world_weights, strict=True)
    current = model.state_dict()
    if any(
        not torch.equal(value.cpu(), current[key].cpu())
        for key, value in ego_state.items()
    ):
        raise RuntimeError("An existing ego parameter changed during initialization")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.save(
        {key: value.cpu() for key, value in current.items()},
        args.output / "model_init.pth",
    )
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    provenance = {
        "ego_checkpoint": str(args.ego_checkpoint.resolve()),
        "ego_sha256": sha256(args.ego_checkpoint),
        "world_checkpoint": str(args.world_checkpoint.resolve()),
        "world_sha256": sha256(args.world_checkpoint),
        "world_tensors_imported": len(world_weights),
        "ego_tensors_preserved": len(ego_state),
        "seed": args.seed,
        "new_keys": result.missing_keys,
        "limitations": [
            "World head was trained on a different backbone representation; fine-tune and validate forecasting.",
            "Causal first-token targets differ from the original rolling targets; retraining and recalibration required.",
            "Ensemble epistemic scoring requires independently trained members, not repeated checkpoints or world modes.",
        ],
    }
    (args.output / "initialization.json").write_text(
        json.dumps(provenance, indent=2) + "\n",
    )
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
