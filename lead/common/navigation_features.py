"""Shared navigation selection and navigation-frame-to-ego transforms."""

from collections.abc import Callable, Sequence

import numpy as np


def build_navigation_features(
    points: Sequence,
    commands: Sequence,
    ego_position: np.ndarray,
    ego_yaw: float,
    *,
    augment: Callable | None = None,
) -> dict:
    """Preserve the existing previous/current/next target and command convention.

    Points and ego position share a navigation frame in meters, yaw is in
    radians, and outputs use CARLA ego coordinates (x forward, y right).
    The GPS-derived frame's origin may differ from the simulator world origin.
    A two-point terminal route keeps
    both entries even when they coincide. Longer routes collapse consecutive
    duplicate XY positions, retaining the first associated command.
    """
    if len(points) != len(commands):
        raise ValueError("Navigation points and commands must have equal lengths")
    filtered_points, filtered_commands = [], []
    for point, command in zip(points, commands, strict=True):
        if (
            len(points) == 2
            or not filtered_points
            or not np.allclose(point[:2], filtered_points[-1][:2])
        ):
            filtered_points.append(point)
            filtered_commands.append(int(command))
    if len(filtered_points) < 2:
        raise ValueError("Navigation needs at least two distinct/terminal targets")
    rotation = np.array(
        [[np.cos(ego_yaw), -np.sin(ego_yaw)], [np.sin(ego_yaw), np.cos(ego_yaw)]],
    )

    def transform(index):
        local = rotation.T @ (
            np.array(filtered_points[index][:2]) - np.array(ego_position[:2])
        )
        return augment(local) if augment is not None else local

    return {
        "target_point_previous": transform(0),
        "target_point": transform(1),
        "target_point_next": transform(2 if len(filtered_points) > 2 else 1),
        "command_id": filtered_commands[0],
        "next_command_id": filtered_commands[1],
    }


def navigation_position_source(closed_loop, training) -> str:
    """Select the same estimator as route progression for the explicit ablation."""
    mode = closed_loop.navigation_position_source
    if mode == "legacy":
        use_filtered = closed_loop.use_kalman_filter
    elif mode == "planner":
        use_filtered = training.use_kalman_filter_for_gps
    else:
        raise ValueError(f"Unknown navigation_position_source: {mode!r}")
    return "filtered_state" if use_filtered else "noisy_state"


def navigation_pop_settings(closed_loop, training) -> tuple[float, bool]:
    """Return target transition distance and whether adaptive selection applies."""
    mode = closed_loop.navigation_pop_distance_mode
    if mode == "legacy":
        return (
            float(closed_loop.route_planner_min_distance),
            bool(closed_loop.sensor_agent_pop_distance_adaptive),
        )
    if mode == "training":
        return float(training.tp_pop_distance), False
    raise ValueError(f"Unknown navigation_pop_distance_mode: {mode!r}")
