"""Observed navigation geometry for conditioning the deployed planning queries."""

import torch
from torch import nn


class NavigationEncoder(nn.Module):
    """Jointly encode ordered targets, route segments and current/next commands.

    Inputs are the same ego-frame metre coordinates and one-hot commands used by
    the existing data loader and online agent. Directions are calculated before
    anisotropic coordinate normalization. Coincident targets have zero direction
    and an explicit invalid-segment bit (terminal/distant-target duplication).
    No future trajectory labels or sensor/AR hidden states enter this encoder.
    """

    TARGET_KEYS = ("target_point_previous", "target_point", "target_point_next")
    COMMAND_KEYS = ("command", "next_command")

    def __init__(self, d_model: int, command_dim: int, point_normalization):
        super().__init__()
        scales = torch.as_tensor(point_normalization, dtype=torch.float32).reshape(-1)
        if scales.shape != (2,) or not bool(torch.isfinite(scales).all()):
            raise ValueError(
                "Navigation point normalization must contain two finite values",
            )
        if not bool((scales > 0).all()):
            raise ValueError("Navigation point normalization must be positive")
        if d_model <= 0 or command_dim <= 0:
            raise ValueError(
                "Navigation embedding and command dimensions must be positive",
            )
        self.command_dim = command_dim
        # Persistent buffers make the feature scale part of new checkpoints.
        self.register_buffer("point_scale", scales)
        self.register_buffer("distance_scale", scales.min().clone())
        # Ordered points(6), segment offsets(4), lengths(2), directions(4),
        # segment validity(2), ego-to-target distances(3), two commands.
        self.feature_dim = 21 + 2 * command_dim
        self.encoder = nn.Sequential(
            nn.Linear(self.feature_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def geometry_features(self, data: dict[str, torch.Tensor]) -> torch.Tensor:
        reference = data["target_point"]
        if reference.ndim != 2 or reference.shape[-1] != 2:
            raise ValueError("Navigation targets must have shape [batch, 2]")
        batch_size = reference.shape[0]
        points = []
        for key in self.TARGET_KEYS:
            point = data[key]
            if point.shape != (batch_size, 2):
                raise ValueError(f"{key} must have shape [batch, 2]")
            points.append(
                point.to(
                    device=self.point_scale.device,
                    dtype=torch.float32,
                    non_blocking=True,
                ),
            )
        commands = []
        for key in self.COMMAND_KEYS:
            command = data[key]
            if command.shape != (batch_size, self.command_dim):
                raise ValueError(f"{key} must have shape [batch, {self.command_dim}]")
            commands.append(
                command.to(
                    device=self.point_scale.device,
                    dtype=torch.float32,
                    non_blocking=True,
                ),
            )

        # Geometry remains fp32 under autocast, particularly for short segments.
        points = torch.stack(points, dim=1)
        segments = points[:, 1:] - points[:, :-1]
        lengths = torch.linalg.vector_norm(segments, dim=-1, keepdim=True)
        valid = lengths > 1e-4
        directions = torch.where(valid, segments / lengths.clamp_min(1e-4), 0.0)
        distances = torch.linalg.vector_norm(points, dim=-1)
        features = torch.cat(
            [
                (points / self.point_scale).flatten(1),
                (segments / self.point_scale).flatten(1),
                (lengths / self.distance_scale).flatten(1),
                directions.flatten(1),
                valid.to(dtype=torch.float32).flatten(1),
                distances / self.distance_scale,
                *commands,
            ],
            dim=-1,
        )
        return features

    def forward(self, data: dict[str, torch.Tensor]) -> torch.Tensor:
        features = self.geometry_features(data)
        return self.encoder(features.to(dtype=self.encoder[0].weight.dtype))


class NavigationModulation(nn.Module):
    """Identity-initialized residual FiLM before a planning decoder layer."""

    def __init__(self, d_model: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.affine = nn.Linear(d_model, 2 * d_model)
        nn.init.zeros_(self.affine.weight)
        nn.init.zeros_(self.affine.bias)

    def forward(self, queries: torch.Tensor, navigation: torch.Tensor) -> torch.Tensor:
        scale, shift = self.affine(navigation).unsqueeze(1).chunk(2, dim=-1)
        return queries + scale * self.norm(queries) + shift
