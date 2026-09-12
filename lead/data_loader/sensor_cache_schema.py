"""Compatibility of the *stored* CARLA sensor tensors, before augmentation.

Keep this boundary in sync with CARLAData._load_sensor_data_and_build_cache and
its label builders. Bump SENSOR_CACHE_VERSION when their algorithms or stored
representation change. Inputs are assumed immutable within a dataset root;
replacing source files requires force_rebuild_data_cache or a new dataset root.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

from lead.common import constants

if TYPE_CHECKING:
    from lead.training.config_training import TrainingConfig


# Version 1 was the unversioned directory layout, which omitted radar and labels.
SENSOR_CACHE_VERSION = 2


def sensor_cache_spec(config: TrainingConfig) -> dict:
    """Return JSON-compatible preprocessing dependencies, excluding run settings.

    Cropping, camera selection, color augmentation, radar point padding and
    CenterNet target generation run after the cache boundary, so they deliberately
    do not appear here. Lossless PNG compression level also preserves the values.
    """
    spec = {
        "version": SENSOR_CACHE_VERSION,
        "modalities": {
            name: getattr(config, name)
            for name in (
                "use_rgb",
                "use_lidar",
                "use_radars",
                "use_semantic",
                "use_depth",
                "use_bev_semantic",
                "detect_boxes",
                "load_bev_3rd_person_images",
            )
        },
    }

    def settings(*names):
        return {name: getattr(config, name) for name in names}

    if config.use_lidar or config.detect_boxes:
        spec["bev_geometry"] = settings(
            "min_x_meter",
            "max_x_meter",
            "min_y_meter",
            "max_y_meter",
            "pixels_per_meter",
        )
    if config.use_lidar:
        spec["lidar"] = settings(
            "training_used_lidar_steps",
            "hist_max_per_pixel",
            "min_height_lidar",
            "max_height_lidar",
        )
    if config.use_semantic:
        spec["save_grouped_semantic"] = config.save_grouped_semantic
        if not config.save_grouped_semantic:
            # The loader builds its lookup array in insertion order.
            spec["semantic_converter"] = list(
                constants.SEMANTIC_SEGMENTATION_CONVERTER.values(),
            )
    if config.use_bev_semantic:
        spec["bev_semantic_converter"] = list(
            constants.CHAFFEURNET_TO_TRANSFUSER_BEV_SEMANTIC_CONVERTER.values(),
        )
        spec["occupancy"] = settings(
            "scale_pedestrian_bev_semantic_size",
            "pedestrian_bev_min_extent",
        )
        spec["occupancy_classes"] = {
            cls.name: cls.value for cls in constants.TransfuserBEVOccupancyClass
        }
    if config.detect_boxes or config.use_bev_semantic:
        spec["box_filtering"] = settings(
            "carla_leaderboard_mode",
            "vehicle_min_num_lidar_points",
            "vehicle_min_num_visible_pixels",
            "pedestrian_min_num_lidar_points",
            "pedestrian_min_num_visible_pixels",
            "parking_vehicle_min_num_lidar_points",
            "parking_vehicle_min_num_visible_pixels",
            "car_open_door_extra_width",
        )
        spec["static_types"] = sorted(config.data_bb_static_types_white_list)
        spec["label_constants"] = {
            "box_classes": {
                cls.name: cls.value for cls in constants.TransfuserBoundingBoxClass
            },
            "box_indices": {
                cls.name: cls.value for cls in constants.TransfuserBoundingBoxIndex
            },
            "semantic_classes": {
                cls.name: cls.value
                for cls in constants.TransfuserSemanticSegmentationClass
            },
            "box_dimensions": constants.LOOKUP_TABLE,
            "traffic_warning_size": constants.TRAFFIC_WARNING_BB_SIZE,
            "construction_cone_size": constants.CONSTRUCTION_CONE_BB_SIZE,
            "emergency_meshes": sorted(constants.EMERGENCY_MESHES),
            "biker_meshes": sorted(constants.BIKER_MESHES),
        }
    if config.detect_boxes:
        spec["boxes"] = settings(
            "min_z",
            "max_z",
            "max_num_bbs",
            "num_way_points_prediction",
            "waypoints_spacing",
            "max_distance_future_waypoint",
        )
    if config.use_radars:
        spec["radar_labels"] = {
            "num_radar_queries": config.num_radar_queries,
            "fields": {cls.name: cls.value for cls in constants.RadarLabels},
        }
    return spec


def sensor_cache_path(config: TrainingConfig) -> tuple[str, str]:
    """Versioned namespace shared by persistent and session caches."""
    encoded = json.dumps(
        sensor_cache_spec(config),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    fingerprint = hashlib.sha256(encoded).hexdigest()[:32]
    return f"sensor-v{SENSOR_CACHE_VERSION}", fingerprint
