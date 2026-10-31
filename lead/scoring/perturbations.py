"""Offline ego-motion sensitivity experiments; these are not closed-loop labels."""

import math

import numpy as np
import torch
from scipy.stats import rankdata


def perturb_ego_motion(
    poses,
    longitudinal_acceleration=0.0,
    lateral_acceleration=0.0,
    dt=0.25,
):
    """Integrate residual acceleration/curvature while preserving a zero perturbation.

    Poses are timed [B,T,3] in the current ego frame. Original local chord/slip
    angles and curvature are retained. Longitudinal residual acceleration changes
    interval arc lengths; speed is clamped at zero. A lateral residual adds
    curvature using the original interval speed, floored at 3 m/s. Callers must
    exclude slow scenes from lateral stress tests. Other traffic does not react.
    """
    if (
        not math.isfinite(dt)
        or dt <= 0
        or not all(
            math.isfinite(x) for x in (longitudinal_acceleration, lateral_acceleration)
        )
    ):
        raise ValueError("Finite accelerations and a positive time step are required")
    poses = torch.as_tensor(poses, dtype=torch.float64)
    if poses.ndim != 3 or poses.shape[-1] != 3 or not torch.isfinite(poses).all():
        raise ValueError("Expected finite timed ego poses [batch, steps, 3]")
    previous = torch.cat([torch.zeros_like(poses[:, :1]), poses[:, :-1]], dim=1)
    displacement = poses[..., :2] - previous[..., :2]
    dyaw = poses[..., 2] - previous[..., 2]
    dyaw = torch.atan2(dyaw.sin(), dyaw.cos())
    local_angle = (
        torch.atan2(displacement[..., 1], displacement[..., 0]) - previous[..., 2]
    )
    arc = displacement.norm(dim=-1) / torch.sinc(dyaw / (2 * math.pi)).clamp_min(1e-6)
    midpoint = (
        torch.arange(poses.shape[1], device=poses.device, dtype=poses.dtype) + 0.5
    ) * dt
    changed_arc = (arc + longitudinal_acceleration * midpoint * dt).clamp_min(0)
    curvature = torch.where(arc > 1e-8, dyaw / arc.clamp_min(1e-8), 0)
    changed_yaw_step = dyaw + curvature * (changed_arc - arc)
    changed_yaw_step += (
        lateral_acceleration / (arc / dt).square().clamp_min(9.0) * changed_arc
    )
    heading = changed_yaw_step.cumsum(-1)
    previous_heading = heading - changed_yaw_step
    direction = previous_heading + local_angle + 0.5 * (changed_yaw_step - dyaw)
    chord = changed_arc * torch.sinc(changed_yaw_step / (2 * math.pi))
    positions = (
        torch.stack([direction.cos(), direction.sin()], dim=-1) * chord[..., None]
    ).cumsum(1)
    return torch.cat(
        [positions, torch.atan2(heading.sin(), heading.cos())[..., None]],
        dim=-1,
    )


def contiguous_window_means(values, routes, frames, width, eligible=None):
    """Mean evidence over consecutive recorded frames, never across drive gaps."""
    values, routes, frames = map(np.asarray, (values, routes, frames))
    if (
        type(width) is not int
        or width < 1
        or not values.shape == routes.shape == frames.shape
    ):
        raise ValueError("Window width and array shapes are invalid")
    eligible = (
        np.ones(len(values), dtype=bool)
        if eligible is None
        else np.asarray(eligible, dtype=bool)
    )
    if eligible.shape != values.shape:
        raise ValueError("Eligibility must match evidence shape")
    windows = []
    for route in np.unique(routes):
        rows = np.flatnonzero(routes == route)
        rows = rows[np.argsort(frames[rows])]
        if len(rows) < width:
            continue
        index = np.lib.stride_tricks.sliding_window_view(rows, width)
        continuous = (np.diff(frames[index], axis=1) == 1).all(1)
        valid = continuous & eligible[index].all(1) & np.isfinite(values[index]).all(1)
        windows.append(values[index[valid]].mean(1))
    return np.concatenate(windows) if windows else np.empty(0)


def upper_tail_probability(values, reference):
    """Conservative empirical ranks; ties count toward the upper tail."""
    values, reference = np.asarray(values), np.sort(np.asarray(reference))
    if (
        not len(reference)
        or not np.isfinite(values).all()
        or not np.isfinite(reference).all()
    ):
        raise ValueError("Finite evidence and a nonempty reference are required")
    return (1 + len(reference) - np.searchsorted(reference, values, side="left")) / (
        len(reference) + 1
    )


def severity_correlations(levels, scores):
    """Per-context Spearman correlation; constant-score contexts are undefined.

    Columns are identical pre-action contexts; rows are graded perturbations.
    Never replace undefined correlations with zero or silently drop their count.
    """
    levels, scores = np.asarray(levels, dtype=float), np.asarray(scores, dtype=float)
    if levels.ndim != 1 or len(levels) < 3 or len(np.unique(levels)) != len(levels):
        raise ValueError("Provide at least three distinct severity levels")
    if (
        scores.ndim != 2
        or scores.shape[0] != len(levels)
        or not np.isfinite(scores).all()
        or not np.isfinite(levels).all()
    ):
        raise ValueError("Scores must be finite [severity levels, contexts]")
    rank = rankdata(scores, axis=0)
    rank -= rank.mean(axis=0, keepdims=True)
    severity_rank = rankdata(levels)
    severity_rank -= severity_rank.mean()
    denominator = np.sqrt(np.square(severity_rank).sum() * np.square(rank).sum(0))
    return np.divide(
        (rank * severity_rank[:, None]).sum(0),
        denominator,
        out=np.full(scores.shape[1], np.nan),
        where=denominator > 0,
    )
