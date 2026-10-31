"""The distributions in DriverBehaviorModeling-3.pdf, equations 4–14 and 19.

Temperatures and smoothing parameters must be fitted on normal validation
drives, separately from the drives used for threshold calibration and testing.
An ensemble member is an independently trained model, never a forecast mode.
"""

import math

import torch
from torch import nn


class MotionSmoother(nn.Module):
    """Column-normalized metric kernel with log-space evaluation of tiny tails."""

    def __init__(self, centroids, heading_weight, epsilon, sigma, chunk_size=64):
        super().__init__()
        if not 0 <= epsilon < 1 or not math.isfinite(epsilon):
            raise ValueError("epsilon must be finite and in [0, 1)")
        if not math.isfinite(sigma) or sigma <= 0:
            raise ValueError("sigma must be finite and positive")
        if not math.isfinite(heading_weight) or heading_weight <= 0:
            raise ValueError("heading_weight must be finite and positive")
        if type(chunk_size) is not int or chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")
        codes = torch.as_tensor(centroids, dtype=torch.float64)
        if codes.ndim != 2 or codes.shape[1] != 3 or not torch.isfinite(codes).all():
            raise ValueError("centroids must be finite [vocabulary, 3] motions")
        if len(codes) < 2:
            raise ValueError("At least two vocabulary motions are required")
        distance = torch.cdist(codes[:, :2], codes[:, :2])
        yaw = codes[:, 2, None] - codes[None, :, 2]
        distance += heading_weight * torch.atan2(yaw.sin(), yaw.cos()).abs()
        # Entry [target, source]. Every source spreads a total mass of one.
        self.register_buffer("log_kernel", (-distance / sigma).log_softmax(dim=0))
        self.epsilon = float(epsilon)
        self.chunk_size = chunk_size

    def forward(self, logits, temperature=1.0):
        """Return normalized log probabilities with shape [..., vocabulary]."""
        if logits.shape[-1] != len(self.log_kernel) or not torch.isfinite(logits).all():
            raise ValueError("logits must be finite and match the motion vocabulary")
        values = logits.to(device=self.log_kernel.device, dtype=torch.float64)
        temperature = torch.as_tensor(
            temperature,
            device=values.device,
            dtype=values.dtype,
        )
        if not torch.isfinite(temperature).all() or not (temperature > 0).all():
            raise ValueError("Temperatures must be finite and positive")
        log_p = (values / temperature).log_softmax(-1)
        if self.epsilon == 0:
            return log_p
        flat = log_p.reshape(-1, log_p.shape[-1])
        blurred = []
        for start in range(0, len(self.log_kernel), self.chunk_size):
            kernel = self.log_kernel[start : start + self.chunk_size]
            blurred.append(torch.logsumexp(flat[:, None] + kernel[None], dim=-1))
        blurred = torch.cat(blurred, dim=-1).reshape_as(log_p)
        result = torch.logaddexp(
            log_p + math.log1p(-self.epsilon),
            blurred + math.log(self.epsilon),
        )
        return result - result.logsumexp(-1, keepdim=True)


def ensemble_step_statistics(member_log_probs, observed_tokens):
    """Score [members, samples, vocabulary] distributions in float64.

    M=1 is supported as a baseline and has I=0; it supplies no estimate of
    epistemic uncertainty. z_centred describes l-H; z_score describes l-A.
    Neither is a calibrated tail probability or a drive-level Gaussian claim.
    """
    p_log = member_log_probs.double()
    if p_log.ndim != 3 or p_log.shape[0] < 1:
        raise ValueError("member_log_probs must be [members, samples, vocabulary]")
    if not torch.isfinite(p_log).all():
        raise ValueError("Log probabilities must be finite")
    if not torch.allclose(
        p_log.logsumexp(-1),
        torch.zeros_like(p_log[..., 0]),
        atol=1e-8,
        rtol=0,
    ):
        raise ValueError("Member distributions must be normalized")
    tokens = torch.as_tensor(observed_tokens, device=p_log.device)
    if tokens.dtype != torch.long or tokens.shape != p_log.shape[1:2]:
        raise ValueError("observed_tokens must be int64 [samples]")
    if not ((tokens >= 0) & (tokens < p_log.shape[-1])).all():
        raise ValueError("Observed token is outside the vocabulary")
    mixture = p_log.logsumexp(0) - math.log(p_log.shape[0])
    probability = mixture.exp()
    entropy = -(probability * mixture).sum(-1)
    aleatoric = -(p_log.exp() * p_log).sum(-1).mean(0)
    epistemic = (entropy - aleatoric).clamp_min(0)
    surprisal = -mixture.gather(-1, tokens[:, None]).squeeze(-1)
    varentropy = (probability * (mixture + entropy[:, None]).square()).sum(-1)
    scale = varentropy.sqrt().clamp_min(1e-8)
    return {
        "log_probability": mixture,
        "surprisal": surprisal,
        "entropy": entropy,
        "aleatoric": aleatoric,
        "epistemic": epistemic,
        "varentropy": varentropy,
        "original_score": surprisal - entropy,
        "score": surprisal - aleatoric,
        "z_centred": (surprisal - entropy) / scale,
        "z_score": (surprisal - aleatoric) / scale,
    }


def ensemble_block_log_likelihood(member_observed_log_probs):
    """Mixture of member joint likelihoods, not product of stepwise mixtures.

    The input contains log probabilities of observed tokens along each member's
    teacher-forced prefix, after its temperature and smoothing transformation.
    Shape: [members, samples, steps]. Each member keeps the same index throughout
    a block. Block aleatoric entropy requires sampling each member separately.
    """
    if (
        member_observed_log_probs.ndim != 3
        or not torch.isfinite(member_observed_log_probs).all()
    ):
        raise ValueError(
            "Block log probabilities must be finite [members, samples, steps]",
        )
    members = member_observed_log_probs.shape[0]
    if members < 1 or member_observed_log_probs.shape[-1] < 1:
        raise ValueError("At least one member and step are required")
    return member_observed_log_probs.double().sum(-1).logsumexp(0) - math.log(members)
