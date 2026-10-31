"""Actor futures must match detection rows, timing, augmentation and cache use."""

from dataclasses import replace

import numpy as np
import pytest

from lead.common.constants import TransfuserBoundingBoxClass as BoxClass
from lead.data_loader.carla_dataset_utils import (
    bb_vehicle_to_image_system,
    bbox_json2array,
    parse_radar_detection_labels,
)
from lead.data_loader.sensor_cache_schema import sensor_cache_path
from lead.data_loader.training_cache import SensorData
from lead.data_loader.world_model_targets import (
    build_world_model_targets,
    validate_world_target_config,
)
from lead.training.config_training import TrainingConfig


@pytest.fixture
def config(monkeypatch):
    monkeypatch.delenv("LEAD_TRAINING_CONFIG", raising=False)
    monkeypatch.setattr("sys.argv", ["test"])
    config = TrainingConfig({"use_radars": True, "detect_boxes": True})
    config.adapt_world_model = True
    config.world_num_steps = 8
    config.world_step_seconds = 0.25
    return config


def sensor(config, boxes, positions=None, counts=None):
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 9)
    n = len(boxes)
    if positions is None:
        positions = np.repeat(boxes[:, None, :2], 8, axis=1)
        positions[:, :, 0] += np.arange(1, 9)
    if counts is None:
        counts = np.full(n, 8, dtype=np.int32)
    data = SensorData(
        image=None,
        rasterized_lidar=None,
        semantic=None,
        hdmap=None,
        depth=None,
        boxes=bb_vehicle_to_image_system(
            boxes,
            config.pixels_per_meter,
            config.min_x_meter,
            config.min_y_meter,
        ),
        boxes_waypoints=np.asarray(positions, dtype=np.float32).reshape(n, 8, 2),
        boxes_num_waypoints=np.asarray(counts, dtype=np.int32),
        bev_occupancy=None,
        bev_3rd_person_image=None,
        radars=None,
        radar_detections=None,
    )
    data.radar_detections = parse_radar_detection_labels(config, data)
    return data


def box(x, speed, cls=BoxClass.VEHICLE, radar_points=1):
    # x, y, extent_x, extent_y, yaw, speed, brake, class, radar points.
    return [x, 2, 2, 1, 0, speed, 0, cls, radar_points]


def test_targets_preserve_radar_sorting_classes_and_truncation(config):
    config.num_radar_queries = 5
    data = sensor(
        config,
        [
            box(11, 5, BoxClass.VEHICLE),
            box(12, 9, BoxClass.WALKER),
            box(13, 5, BoxClass.SPECIAL),
            box(14, 100, radar_points=0),
            box(15, 5, BoxClass.VEHICLE, radar_points=9),
            box(16, 0, BoxClass.PARKING),
            box(17, -1, BoxClass.OBSTACLE),
        ],
    )
    targets = build_world_model_targets(config, data)
    np.testing.assert_array_equal(data.radar_detections[:, 0], [12, 13, 15, 11, 16])
    np.testing.assert_array_equal(
        targets["world_vehicle_mask"],
        [False, True, True, True, True],
    )
    np.testing.assert_array_equal(
        targets["world_future_positions"][:, 0, 0],
        [0, 14, 16, 12, 17],
    )
    assert not targets["world_future_mask"][0].any()
    assert targets["world_future_mask"][1:].all()


def test_distinct_actor_targets_follow_box_permutations(config):
    data = sensor(config, [box(10, 1), box(20, 3), box(30, 2)])
    expected = build_world_model_targets(config, data)
    order = np.array([2, 0, 1])
    reordered = replace(
        data,
        boxes=data.boxes[order],
        boxes_waypoints=data.boxes_waypoints[order],
        boxes_num_waypoints=data.boxes_num_waypoints[order],
    )
    actual = build_world_model_targets(config, reordered)
    for key in expected:
        np.testing.assert_array_equal(expected[key], actual[key])


@pytest.mark.parametrize("count", [0, 1, 6, 7])
def test_incomplete_or_time_compressed_tracks_are_entirely_masked(config, count):
    # Even an apparently valid prefix may contain an omitted intermediate tick;
    # extrapolated suffixes from the legacy cache must never become supervision.
    data = sensor(config, [box(10, 3)], counts=[count])
    data.boxes_waypoints[0, count:] = 12345
    targets = build_world_model_targets(config, data)
    assert targets["world_vehicle_mask"][0]
    assert not targets["world_future_mask"].any()
    assert not targets["world_future_positions"].any()


def test_nonfinite_future_and_nonvehicle_are_never_supervised(config):
    data = sensor(config, [box(10, 3), box(20, 2, BoxClass.OBSTACLE)])
    data.boxes_waypoints[0, 4, 0] = np.nan
    targets = build_world_model_targets(config, data)
    assert not targets["world_future_mask"].any()
    assert np.isfinite(targets["world_future_positions"]).all()
    np.testing.assert_array_equal(targets["world_vehicle_mask"][:2], [True, False])


def test_empty_scene_produces_fixed_size_masked_targets(config):
    targets = build_world_model_targets(config, sensor(config, []))
    assert targets["world_future_positions"].shape == (config.num_radar_queries, 8, 2)
    assert targets["world_future_positions"].dtype == np.float32
    assert targets["world_future_mask"].dtype == np.bool_
    assert not any(values.any() for values in targets.values())


def test_augmented_future_positions_match_current_box_frame_and_time(config):
    raw = {
        "class": "car",
        "position": [4, 2, 0],
        "extent": [2, 1, 1],
        "yaw": 0.0,
        "speed": 4.0,
        "brake": 0.0,
        "num_radar_points": 2,
        "future_positions": np.column_stack([4 + 0.2 * np.arange(41), np.full(41, 2)]),
        "future_yaws": np.zeros(41),
    }
    current, positions, count = bbox_json2array(raw, 1.0, 90.0, config)
    data = sensor(config, current[None], positions[None], [count])
    targets = build_world_model_targets(config, data)
    # Move ego origin to y=1, rotate +90 degrees: (x, y) -> (y-1, -x).
    np.testing.assert_allclose(data.radar_detections[0, :2], [1, -4], atol=1e-5)
    expected = np.column_stack([np.ones(8), -(4 + np.arange(1, 9))])
    np.testing.assert_allclose(
        targets["world_future_positions"][0],
        expected,
        atol=1e-5,
    )
    assert targets["world_future_mask"][0].all()


def test_targets_survive_existing_sensor_cache_roundtrip(config):
    data = sensor(config, [box(10, 3), box(20, 1)], counts=[8, 6])
    before = build_world_model_targets(config, data)
    restored = data.compress(None, config, {}).decompress()
    after = build_world_model_targets(config, restored)
    for key in before:
        np.testing.assert_array_equal(before[key], after[key])
    np.testing.assert_array_equal(
        restored.radar_detections,
        parse_radar_detection_labels(config, restored),
    )
    enabled_key = sensor_cache_path(config)
    config.adapt_world_model = False
    assert sensor_cache_path(config) == enabled_key


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("world_num_steps", 7),
        ("world_step_seconds", 0.5),
        ("other_vehicles_num_temporal_data_points_saved", 80),
        ("use_radars", False),
        ("detect_boxes", False),
    ],
)
def test_incompatible_horizon_or_input_configuration_rejected(config, name, value):
    setattr(config, name, value)
    with pytest.raises(ValueError):
        validate_world_target_config(config)


def test_missing_future_cache_fields_raise_instead_of_disabling_loss(config):
    data = sensor(config, [box(10, 3)])
    data.boxes_num_waypoints = None
    with pytest.raises(ValueError, match="validity counts"):
        build_world_model_targets(config, data)
