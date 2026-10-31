"""Object-centric vehicle forecasts used as predicted evidence for ego planning."""

import math

import torch
import torch.nn.functional as F
from torch import nn


class VehicleWorldModel(nn.Module):
    """Forecast multiple futures for each current radar query, without label inputs.

    Actor identities are local to a sample. Supervision uses the detector's current
    state assignment, not a second assignment based on future trajectories. All
    positions and interval velocities use the fixed current-ego coordinate frame.
    The planning memory is built from predicted states and probabilities only;
    there is no hidden-feature shortcut around forecasting.
    """

    def __init__(self, config):
        super().__init__()
        self.num_steps = config.world_num_steps
        self.num_modes = config.world_num_modes
        self.radar_dim = config.radar_token_dim
        hidden_dim = config.world_hidden_dim
        self.memory_dim = config.kinematic_embed_dim
        world_max_speed = getattr(config, "world_max_speed", config.max_speed)
        for name, value in (
            ("world_num_steps", self.num_steps),
            ("world_num_modes", self.num_modes),
            ("world_hidden_dim", hidden_dim),
            ("radar_token_dim", self.radar_dim),
            ("kinematic_embed_dim", self.memory_dim),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name, value in (
            ("world_step_seconds", config.world_step_seconds),
            ("world_position_scale", config.world_position_scale),
            ("max_speed", config.max_speed),
            ("world_max_speed", world_max_speed),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        self.register_buffer(
            "step_seconds",
            torch.tensor(config.world_step_seconds, dtype=torch.float32),
        )
        self.register_buffer(
            "position_scale",
            torch.tensor(config.world_position_scale, dtype=torch.float32),
        )
        self.register_buffer(
            "speed_scale",
            torch.tensor(world_max_speed, dtype=torch.float32),
        )
        self.register_buffer(
            "radar_speed_scale",
            torch.tensor(config.max_speed, dtype=torch.float32),
        )
        self.actor_encoder = nn.Sequential(
            nn.Linear(self.radar_dim + 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.displacement_head = nn.Linear(
            hidden_dim,
            self.num_modes * self.num_steps * 2,
        )
        self.mode_head = nn.Linear(hidden_dim, self.num_modes)
        self.vehicle_head = nn.Linear(hidden_dim, 1)
        # Independent mode outputs avoid identical initial hypotheses, while
        # small initial displacements avoid large arbitrary motion predictions.
        nn.init.normal_(self.displacement_head.weight, std=0.01)
        nn.init.zeros_(self.displacement_head.bias)
        self.memory_encoder = nn.Sequential(
            nn.Linear(self.num_steps * 4 + 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.memory_dim),
            nn.LayerNorm(self.memory_dim),
        )

    def encode_memory(
        self,
        future_positions: torch.Tensor,
        future_velocities: torch.Tensor,
        mode_logits: torch.Tensor,
        vehicle_logits: torch.Tensor,
        detection_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode separate predicted modes and their confidence for planning.

        Confidence stays differentiable; no threshold removes uncertain actors.
        Missing future annotations never enter this computation.
        """
        mode_probability = mode_logits.float().softmax(dim=-1)
        vehicle_probability = vehicle_logits.float().sigmoid().unsqueeze(-1)
        detection_probability = detection_logits.float().sigmoid().unsqueeze(-1)
        confidence = mode_probability * vehicle_probability * detection_probability
        states = torch.cat(
            [
                future_positions.float() / self.position_scale.float(),
                future_velocities.float() / self.speed_scale.float(),
            ],
            dim=-1,
        ).flatten(-2)
        features = torch.cat(
            [
                states,
                mode_probability.unsqueeze(-1),
                vehicle_probability.expand_as(mode_probability).unsqueeze(-1),
                detection_probability.expand_as(mode_probability).unsqueeze(-1),
            ],
            dim=-1,
        )
        memory = self.memory_encoder(
            features.to(dtype=self.memory_encoder[0].weight.dtype),
        )
        return memory.flatten(1, 2), confidence.flatten(1, 2)

    def forward(
        self,
        radar_features: torch.Tensor,
        radar_predictions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Predict future positions, interval velocities, modes and vehicle type."""
        if radar_features.ndim != 3 or radar_features.shape[-1] != self.radar_dim:
            raise ValueError(
                "radar_features must have shape [batch, queries, radar_dim]",
            )
        batch, queries, _ = radar_features.shape
        if radar_predictions.shape != (batch, queries, 4):
            raise ValueError("radar_predictions must have shape [batch, queries, 4]")
        device = self.position_scale.device
        state = radar_predictions.to(device=device, dtype=torch.float32)
        normalized_state = torch.cat(
            [
                state[..., :2] / self.position_scale.float(),
                state[..., 2:3] / self.radar_speed_scale.float(),
                state[..., 3:4].sigmoid(),
            ],
            dim=-1,
        )
        features = torch.cat(
            [radar_features.to(device=device, dtype=torch.float32), normalized_state],
            dim=-1,
        ).to(dtype=self.actor_encoder[0].weight.dtype)
        actor = self.actor_encoder(features)
        raw_displacements = (
            self.displacement_head(actor)
            .float()
            .reshape(batch, queries, self.num_modes, self.num_steps, 2)
        )
        # Bound vector speed, not each coordinate independently (which would
        # otherwise permit sqrt(2) times world_max_speed on diagonal trajectories).
        # Other traffic may exceed the ego/detector's configured speed range.
        unit_displacements = raw_displacements.tanh()
        unit_displacements = unit_displacements / torch.linalg.vector_norm(
            unit_displacements,
            dim=-1,
            keepdim=True,
        ).clamp_min(1.0)
        future_velocities = unit_displacements * self.speed_scale.float()
        displacements = future_velocities * self.step_seconds.float()
        future_positions = state[..., None, None, :2] + displacements.cumsum(dim=-2)
        mode_logits = self.mode_head(actor).float()
        vehicle_logits = self.vehicle_head(actor).squeeze(-1).float()
        memory, confidence = self.encode_memory(
            future_positions,
            future_velocities,
            mode_logits,
            vehicle_logits,
            state[..., 3],
        )
        return {
            "future_positions": future_positions,
            "future_velocities": future_velocities,
            "mode_logits": mode_logits,
            "vehicle_logits": vehicle_logits,
            "memory": memory,
            "memory_confidence": confidence,
        }

    def compute_loss(
        self,
        outputs: dict[str, torch.Tensor],
        data: dict[str, torch.Tensor],
        matching: tuple[torch.Tensor, torch.Tensor],
        loss: dict[str, torch.Tensor],
        log: dict[str, torch.Tensor],
    ) -> None:
        """Supervise futures using the *current detector* Hungarian assignment.

        Padding and unknown/truncated future frames contribute no regression or
        mode loss. Current vehicle classification is supervised independently of
        future availability, including negative padded/nonvehicle detector slots.
        """
        predicted = outputs["future_positions"]
        batch, queries, modes, steps, _ = predicted.shape
        if (modes, steps) != (self.num_modes, self.num_steps):
            raise ValueError("Future prediction dimensions do not match configuration")
        device = predicted.device
        pred_indices, gt_indices = (
            index.to(device=device, dtype=torch.long) for index in matching
        )
        if pred_indices.shape != (batch, queries) or gt_indices.shape != (
            batch,
            queries,
        ):
            raise ValueError("Detector matching must contain [batch, queries] indices")
        target = data["world_future_positions"].to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
        available = data["world_future_mask"].to(
            device=device,
            dtype=torch.bool,
            non_blocking=True,
        )
        vehicle = data["world_vehicle_mask"].to(
            device=device,
            dtype=torch.bool,
            non_blocking=True,
        )
        current_valid = (
            data["radar_detections"].to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )[..., 3]
            > 0.5
        )
        if target.shape != (batch, queries, steps, 2):
            raise ValueError(
                "world_future_positions must have shape [batch, queries, steps, 2]",
            )
        if available.shape != (batch, queries, steps):
            raise ValueError(
                "world_future_mask must have shape [batch, queries, steps]",
            )
        if vehicle.shape != (batch, queries) or current_valid.shape != (batch, queries):
            raise ValueError(
                "World vehicle and detection validity must match query dimensions",
            )
        batch_indices = torch.arange(batch, device=device)[:, None]
        target = target[batch_indices, gt_indices]
        is_vehicle = (vehicle & current_valid)[batch_indices, gt_indices]
        mask = available[batch_indices, gt_indices] & is_vehicle.unsqueeze(-1)
        predicted = predicted[batch_indices, pred_indices].float()
        mode_logits = outputs["mode_logits"][batch_indices, pred_indices].float()
        vehicle_logits = outputs["vehicle_logits"][batch_indices, pred_indices].float()

        with torch.autocast(device_type=device.type, enabled=False):
            # Replace unknown targets BEFORE subtraction: NaN * zero remains NaN.
            target = torch.where(mask.unsqueeze(-1), target, 0.0)
            difference = torch.where(
                mask[:, :, None, :, None],
                predicted - target.unsqueeze(2),
                0.0,
            )
            valid_steps = mask.sum(dim=-1).clamp_min(1).float()
            has_future = mask.any(dim=-1)
            valid_actors = has_future.sum().clamp_min(1).float()
            distances = torch.linalg.vector_norm(difference, dim=-1)
            ade = distances.sum(dim=-1) / valid_steps.unsqueeze(-1)
            winner = ade.detach().argmin(dim=-1)
            per_mode_position = difference.abs().sum(dim=(-1, -2)) / (
                2.0 * self.position_scale.float() * valid_steps.unsqueeze(-1)
            )
            winning_position = per_mode_position.gather(
                -1,
                winner.unsqueeze(-1),
            ).squeeze(-1)
            loss["world_loss_position"] = (
                winning_position * has_future
            ).sum() / valid_actors
            mode_ce = F.cross_entropy(
                mode_logits.reshape(-1, modes),
                winner.reshape(-1),
                reduction="none",
            ).reshape(batch, queries)
            loss["world_loss_mode"] = (mode_ce * has_future).sum() / valid_actors
            loss["world_loss_vehicle"] = F.binary_cross_entropy_with_logits(
                vehicle_logits,
                is_vehicle.float(),
            )

            min_ade = ade.min(dim=-1).values
            chosen_ade = ade.gather(
                -1,
                mode_logits.argmax(dim=-1, keepdim=True),
            ).squeeze(-1)
            # Last *available* frame, not necessarily the end of the horizon.
            time_indices = torch.arange(1, steps + 1, device=device)
            last_valid = (mask * time_indices).amax(dim=-1).sub(1).clamp_min(0)
            final_distances = distances.gather(
                -1,
                last_valid[:, :, None, None].expand(-1, -1, modes, 1),
            ).squeeze(-1)
            for name, value in (
                ("world_min_ade", min_ade),
                ("world_mode_ade", chosen_ade),
                ("world_min_fde", final_distances.min(dim=-1).values),
            ):
                log[f"metric/{name}"] = (
                    (value * has_future).sum() / valid_actors
                ).detach()
            log["metric/world_supervised_actors"] = has_future.sum().detach()
