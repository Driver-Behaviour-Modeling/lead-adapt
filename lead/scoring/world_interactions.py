"""Interaction evidence from timed ego motion and predicted vehicle futures.

These are geometric conflict proxies, not likelihoods or causal blame. The old
vehicle predictor estimates centres/velocities, not dimensions/headings, so actor
extents are explicit assumptions and heading comes from velocity. Stationary
actor orientation is unknown; an orientation-free enclosing circle is used.
Only predicted state enters the reference. Observed ego motion is the behaviour
being evaluated, supplied after its horizon has elapsed.
"""

import math

import torch


def forecast_error_sums(actor_futures, data, matching):
    """Per-scene forecast error sums using current-state detector assignment.

    Only annotated vehicle futures count. Report the denominator: missing future
    annotations and traffic beyond the detector's query limit are not evaluated.
    Future labels are used here after prediction, never to form scene context.
    """
    position = actor_futures["future_positions"].float()
    device = position.device
    batch, queries, modes, steps, _ = position.shape
    pred_indices, gt_indices = (x.to(device=device, dtype=torch.long) for x in matching)
    row = torch.arange(batch, device=device)[:, None]
    target = data["world_future_positions"].to(device=device).float()[row, gt_indices]
    available = data["world_future_mask"].to(device=device).bool()[row, gt_indices]
    vehicle = data["world_vehicle_mask"].to(device=device).bool()[row, gt_indices]
    current = (data["radar_detections"].to(device=device)[..., 3] > 0.5)[
        row,
        gt_indices,
    ]
    mask = available & (vehicle & current)[..., None]
    valid = mask.any(-1)
    difference = torch.where(
        mask[:, :, None, :, None],
        position[row, pred_indices]
        - torch.where(mask[..., None], target, 0).unsqueeze(2),
        0,
    )
    distance = difference.norm(dim=-1)
    ade = distance.sum(-1) / mask.sum(-1).clamp_min(1)[..., None]
    most_likely = actor_futures["mode_logits"].float()[row, pred_indices].argmax(-1)
    last = (
        (mask * torch.arange(1, steps + 1, device=device)).amax(-1).sub(1).clamp_min(0)
    )
    fde = distance.gather(-1, last[:, :, None, None].expand(-1, -1, modes, 1)).squeeze(
        -1,
    )
    return {
        "actor_count": valid.sum(-1),
        "current_vehicle_count": (vehicle & current).sum(-1),
        "min_ade_sum": (ade.min(-1).values * valid).sum(-1),
        "mode_ade_sum": (
            ade.gather(-1, most_likely[..., None]).squeeze(-1) * valid
        ).sum(-1),
        "min_fde_sum": (fde.min(-1).values * valid).sum(-1),
        "mode_fde_sum": (
            fde.gather(-1, most_likely[..., None]).squeeze(-1) * valid
        ).sum(-1),
    }


def interaction_features(
    ego_poses,
    actor_futures,
    *,
    ego_half_extent=(2.451, 1.064),
    actor_half_extent=(2.5, 1.1),
    margin=0.25,
    clearance_scale=0.5,
):
    """Return expected conflicting-actor count over forecast modes.

    ego_poses: [B,T,3] in the fixed current-ego frame at the predictor's times.
    Counts sum per-actor probabilities; they do not assume actors independent.
    The horizon is collapsed *within* each mode, then averaged across modes.
    A forecast with little existence confidence contributes correspondingly
    little evidence; missing traffic can still make the risk estimate wrong.
    """
    for name, value in (
        ("margin", margin),
        ("clearance_scale", clearance_scale),
        *[("half_extent", x) for x in (*ego_half_extent, *actor_half_extent)],
    ):
        if not math.isfinite(value) or value < 0 or (name != "margin" and value == 0):
            raise ValueError(f"{name} has an invalid value")
    position = actor_futures["future_positions"].float()
    velocity = actor_futures["future_velocities"].float()
    if (
        position.ndim != 5
        or position.shape[-1] != 2
        or velocity.shape != position.shape
    ):
        raise ValueError("Actor positions/velocities must be [B,Q,M,T,2]")
    batch, actors, modes, steps, _ = position.shape
    ego = ego_poses.to(device=position.device, dtype=torch.float32)
    if ego.shape != (batch, steps, 3):
        raise ValueError("Ego poses must match the forecast's batch and timestamps")
    probability = (
        actor_futures["memory_confidence"].float().reshape(batch, actors, modes)
    )
    if not all(
        torch.isfinite(value).all() for value in (position, velocity, ego, probability)
    ):
        raise ValueError("Interaction inputs must be finite")
    if not ((probability >= 0).all() and (probability.sum(-1) <= 1 + 1e-5).all()):
        raise ValueError("Mode confidence must represent sub-probabilities per actor")
    actor_yaw = torch.atan2(velocity[..., 1], velocity[..., 0])
    ego_yaw = ego[:, None, None, :, 2].expand_as(actor_yaw)
    ea = torch.stack([ego_yaw.cos(), ego_yaw.sin()], dim=-1)
    eb = torch.stack([-ego_yaw.sin(), ego_yaw.cos()], dim=-1)
    aa = torch.stack([actor_yaw.cos(), actor_yaw.sin()], dim=-1)
    ab = torch.stack([-actor_yaw.sin(), actor_yaw.cos()], dim=-1)
    delta = position - ego[:, None, None, :, :2]
    gaps = []
    for axis in (ea, eb, aa, ab):
        ego_radius = (
            ego_half_extent[0] * (ea * axis).sum(-1).abs()
            + ego_half_extent[1] * (eb * axis).sum(-1).abs()
        )
        actor_radius = (
            actor_half_extent[0] * (aa * axis).sum(-1).abs()
            + actor_half_extent[1] * (ab * axis).sum(-1).abs()
        )
        gaps.append((delta * axis).sum(-1).abs() - ego_radius - actor_radius - margin)
    # Positive = separated on an axis, negative = overlapping rectangles.
    # This is a separating-axis gap, not an exact Euclidean distance.
    clearance = torch.stack(gaps, dim=-1).max(-1).values
    stationary = velocity.norm(dim=-1) < 0.5
    circle_gap = (
        delta.norm(dim=-1)
        - math.hypot(*ego_half_extent)
        - math.hypot(*actor_half_extent)
        - margin
    )
    clearance = torch.where(stationary, circle_gap, clearance)
    mode_minimum = clearance.min(-1).values
    hard_conflict = (mode_minimum <= 0).float()
    soft_conflict = torch.sigmoid(-mode_minimum / clearance_scale)
    return {
        "mode_minimum_clearance": mode_minimum,
        "per_actor_conflict": (probability * hard_conflict).sum(-1),
        "expected_conflicting_actor_count": (probability * hard_conflict).sum((-1, -2)),
        "soft_expected_conflicting_actor_count": (probability * soft_conflict).sum(
            (-1, -2),
        ),
        "total_detected_vehicle_confidence": probability.sum((-1, -2)),
    }
