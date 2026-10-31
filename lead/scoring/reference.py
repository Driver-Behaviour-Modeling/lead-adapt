"""Common next-step reference used by offline ego and interaction evaluation."""

import torch
from torch import nn

from lead.scoring.distributions import MotionSmoother, ensemble_step_statistics


class NormativeTokenScorer(nn.Module):
    """Temperature, metric smoothing, and independent-member entropy accounting."""

    def __init__(
        self,
        centroids,
        heading_weight,
        temperatures,
        *,
        epsilon=0.01,
        sigma=0.1,
    ):
        super().__init__()
        self.smoother = MotionSmoother(centroids, heading_weight, epsilon, sigma)
        self.register_buffer(
            "centroids",
            torch.as_tensor(centroids, dtype=torch.float64).clone(),
        )
        temperatures = torch.as_tensor(temperatures, dtype=torch.float64)
        if (
            temperatures.ndim != 1
            or not len(temperatures)
            or not torch.isfinite(temperatures).all()
            or not (temperatures > 0).all()
        ):
            raise ValueError("Provide one positive temperature per independent member")
        self.register_buffer("temperatures", temperatures)
        self.heading_weight = heading_weight

    def forward(self, member_logits, observed_delta):
        """Score the nearest physical motion token, without future-context labels."""
        if member_logits.ndim != 3 or member_logits.shape[0] != len(self.temperatures):
            raise ValueError(
                "Expected [members, samples, vocabulary] logits matching temperatures",
            )
        observed = observed_delta.to(device=self.centroids.device, dtype=torch.float64)
        if (
            observed.shape != (member_logits.shape[1], 3)
            or not torch.isfinite(observed).all()
        ):
            raise ValueError("Observed body-frame motions must be finite [samples, 3]")
        distance = torch.cdist(observed[:, :2], self.centroids[:, :2])
        yaw = observed[:, 2, None] - self.centroids[None, :, 2]
        distance += self.heading_weight * torch.atan2(yaw.sin(), yaw.cos()).abs()
        token = distance.argmin(-1)
        member_log_probs = self.smoother(
            member_logits,
            self.temperatures[:, None, None],
        )
        result = ensemble_step_statistics(member_log_probs, token)
        result.update(
            token=token,
            quantization_residual=distance.gather(-1, token[:, None]).squeeze(-1),
        )
        return result
