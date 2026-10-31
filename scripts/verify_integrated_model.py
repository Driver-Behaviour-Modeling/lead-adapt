"""Check full-model initialization and one real-data backward pass on the GPU."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from lead.adapt.transfuser_utils import patch_norm_fp32
from lead.data_loader.carla_dataset import CARLAData
from lead.scoring.world_interactions import interaction_features
from lead.training.config_training import TrainingConfig
from lead.training.input_pipeline import prepared_training_batches
from lead.training.mixed_training_utils import mixed_data_collate_fn
from lead.training.training_utils import create_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    config_values = json.loads((args.checkpoint_dir / "config.json").read_text())
    config_values.update(
        use_sensor_perburtation=False,
        use_color_aug=False,
        cuda_prefetch=True,
        compile=False,
    )
    config = TrainingConfig(config_values)
    dataset = CARLAData(
        config.carla_data,
        config,
        training_session_cache=None,
        build_cache=False,
        random=False,
    )
    loader = DataLoader(
        Subset(dataset, list(range(args.batch_size))),
        batch_size=args.batch_size,
        num_workers=0,
        collate_fn=mixed_data_collate_fn,
    )
    with prepared_training_batches(loader, config) as iterator:
        data = next(iterator)
    model = patch_norm_fp32(create_model(config)).cuda().eval()
    state = torch.load(
        args.checkpoint_dir / "model_init.pth", map_location="cpu", weights_only=True,
    )
    model.load_state_dict(state, strict=True)
    inference_data = {
        key: value
        for key, value in data.items()
        if key not in ("future_waypoints", "future_yaws")
    }
    with torch.no_grad(), torch.autocast("cuda", dtype=config.torch_float_type):
        prediction = model(inference_data)
        original = {
            key: model._adapt_outputs[key].clone()
            for key in (
                "output_logits",
                "pred_future_waypoints",
                "pred_route",
                "pred_target_speed_distribution",
            )
        }
        risk = interaction_features(
            torch.cat(
                [data["future_waypoints"], data["future_yaws"][..., None]], dim=-1,
            ),
            prediction.pred_actor_futures,
        )
    provenance = json.loads((args.checkpoint_dir / "initialization.json").read_text())
    base_path = Path(provenance["ego_checkpoint"])
    baseline_values = json.loads(base_path.with_name("config.json").read_text())
    baseline_values.update(
        use_sensor_perburtation=False, use_color_aug=False, compile=False,
    )
    baseline = (
        patch_norm_fp32(create_model(TrainingConfig(baseline_values))).cuda().eval()
    )
    baseline.load_state_dict(
        torch.load(base_path, map_location="cpu", weights_only=True), strict=True,
    )
    with torch.no_grad(), torch.autocast("cuda", dtype=config.torch_float_type):
        baseline(inference_data)
    deltas = {
        key: (original[key].float() - baseline._adapt_outputs[key].float())
        .abs()
        .max()
        .item()
        for key in original
    }
    if any(value != 0 for value in deltas.values()):
        raise RuntimeError(f"Identity initialization changed ego inference: {deltas}")
    del baseline
    model.train()
    with torch.autocast("cuda", dtype=config.torch_float_type):
        prediction = model(data)
        losses, _ = model.compute_loss(prediction, data)
        chosen = [
            "loss_kinematic_token",
            "world_loss_position",
            "world_loss_mode",
            "world_loss_vehicle",
        ]
        total = sum(losses[key] for key in chosen)
    total.backward()
    if not all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    ):
        raise RuntimeError("A real-data training gradient was nonfinite")
    nav = model.adapt_decoder._token_navigation_modulation.affine.weight.grad.norm().item()
    fusion = (
        model.adapt_decoder._world_fusion.attention.out_proj.weight.grad.norm().item()
    )
    if nav == 0 or fusion == 0:
        raise RuntimeError("Token training does not reach both new context branches")
    report = {
        "batch_size": args.batch_size,
        "identity_max_abs_delta": deltas,
        "real_data_losses": {key: losses[key].item() for key in chosen},
        "navigation_gradient_norm": nav,
        "world_fusion_gradient_norm": fusion,
        "supervised_actor_steps": int(data["world_future_mask"].sum().item()),
        "observed_interaction_proxy": risk["expected_conflicting_actor_count"].tolist(),
        "finite_gradients": True,
        "interpretation": "Initialization and backward smoke check; no optimizer step or driving-performance claim.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
