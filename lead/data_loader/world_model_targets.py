"""Actor-future supervision derived from the existing CARLA sensor cache.

The recorder stores up to 41 actor poses at 20 Hz, starting with the current
pose. It omits observations when an actor is missing, without storing offsets.
Consequently only a complete 41-pose recording establishes the timestamps of
all eight targets at offsets 5, 10, ..., 40. Short recordings are entirely
masked; neither extrapolated suffixes nor possibly time-compressed prefixes
are used as ground truth. Missing observations do not establish nonexistence.
"""

from __future__ import annotations

import numpy as np

from lead.common.constants import TransfuserBoundingBoxClass, TransfuserBoundingBoxIndex
from lead.data_loader.carla_dataset_utils import select_radar_detection_boxes
from lead.data_loader.training_cache import SensorData
from lead.training.config_training import TrainingConfig


def validate_world_target_config(config: TrainingConfig) -> None:
    """Reject horizons for which this cache cannot establish dense timestamps."""
    if (
        not config.carla_leaderboard_mode
        or not config.use_radars
        or not config.detect_boxes
    ):
        raise ValueError(
            "World-model targets require CARLA radar and bounding-box labels",
        )
    steps = config.world_num_steps
    if steps != config.num_way_points_prediction:
        raise ValueError("world_num_steps must equal the cached ego waypoint horizon")
    if (
        steps * config.waypoints_spacing
        != config.other_vehicles_num_temporal_data_points_saved
    ):
        raise ValueError(
            "World-model targets must span the complete recorded actor horizon",
        )
    if not np.isclose(
        config.world_step_seconds,
        config.waypoints_spacing / config.carla_fps,
    ):
        raise ValueError("world_step_seconds must match the recorded waypoint spacing")


def build_world_model_targets(
    config: TrainingConfig,
    sensor_data: SensorData,
) -> dict[str, np.ndarray]:
    """Build targets in exactly the same row order as ``radar_detections``.

    Futures are already in the current ego frame, with the same sensor rotation
    and translation augmentation as their current boxes. This runs after cache
    loading so enabling the model does not change any stored sensor tensors.
    """
    validate_world_target_config(config)
    steps, queries = config.world_num_steps, config.num_radar_queries
    positions = np.zeros((queries, steps, 2), dtype=np.float32)
    future_mask = np.zeros((queries, steps), dtype=np.bool_)
    vehicle_mask = np.zeros(queries, dtype=np.bool_)
    boxes, indices = select_radar_detection_boxes(config, sensor_data)
    if sensor_data.boxes_waypoints is None or sensor_data.boxes_num_waypoints is None:
        raise ValueError(
            "World-model targets require cached actor futures and validity counts",
        )
    if sensor_data.boxes_waypoints.shape[1:] != (steps, 2):
        raise ValueError(
            "Cached actor future shape does not match the world-model horizon",
        )

    count = len(indices)
    vehicle_mask[:count] = np.isin(
        boxes[:, TransfuserBoundingBoxIndex.CLASS],
        [
            TransfuserBoundingBoxClass.VEHICLE,
            TransfuserBoundingBoxClass.SPECIAL,
            TransfuserBoundingBoxClass.PARKING,
        ],
    )
    selected_positions = sensor_data.boxes_waypoints[indices]
    complete = sensor_data.boxes_num_waypoints[indices] == steps
    finite = np.isfinite(selected_positions).all(axis=(1, 2))
    valid = complete & finite & vehicle_mask[:count]
    future_mask[:count] = valid[:, None]
    positions[:count] = np.where(valid[:, None, None], selected_positions, 0.0)
    return {
        "world_future_positions": positions,
        "world_future_mask": future_mask,
        "world_vehicle_mask": vehicle_mask,
    }
