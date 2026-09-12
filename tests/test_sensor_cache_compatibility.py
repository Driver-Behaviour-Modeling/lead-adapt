"""Sensor preprocessing changes must invalidate both cache tiers consistently."""

import json
import lzma
import pickle
from pathlib import Path

import diskcache
import numpy as np
import pytest

from lead.common import constants
from lead.data_loader import sensor_cache_schema, training_cache
from lead.data_loader.training_cache import CacheKey, PersistentCache, SensorData
from lead.training.config_training import TrainingConfig


@pytest.fixture
def config_factory(monkeypatch, tmp_path):
    monkeypatch.delenv("LEAD_TRAINING_CONFIG", raising=False)
    monkeypatch.setattr("sys.argv", ["test"])

    def make(**values):
        return TrainingConfig(
            {"carla_root": str(tmp_path / "carla_leaderboard2"), **values},
        )

    return make


def key(config, **values):
    return CacheKey(
        **{
            "scenario": "turn",
            "route": "route_1",
            "frame": "0020",
            "perturbated": False,
            **values,
        },
        config=config,
    )


@pytest.mark.parametrize(
    ("setting", "changed"),
    [
        ("use_rgb", False),
        ("use_lidar", False),
        ("use_radars", False),
        ("use_semantic", False),
        ("use_depth", False),
        ("use_bev_semantic", False),
        ("detect_boxes", False),
        ("load_bev_3rd_person_images", True),
        ("num_radar_queries", 5),
        ("pixels_per_meter", 2.0),
        ("min_x_meter", -16),
        ("training_used_lidar_steps", 2),
        ("hist_max_per_pixel", 8),
        ("min_height_lidar", -2.0),
        ("max_height_lidar", 5.0),
        ("save_grouped_semantic", False),
        ("vehicle_min_num_lidar_points", 2),
        ("vehicle_min_num_visible_pixels", 4),
        ("pedestrian_min_num_lidar_points", 2),
        ("pedestrian_min_num_visible_pixels", 4),
        ("parking_vehicle_min_num_lidar_points", 2),
        ("parking_vehicle_min_num_visible_pixels", 4),
        ("car_open_door_extra_width", 4.0),
        ("data_bb_static_types_white_list", ["new.obstacle"]),
        ("min_z", -2),
        ("max_z", 8),
        ("max_num_bbs", 12),
        ("num_way_points_prediction", 3),
        ("waypoints_spacing", 8),
        ("max_distance_future_waypoint", 100.0),
        ("scale_pedestrian_bev_semantic_size", 5.0),
        ("pedestrian_bev_min_extent", 2.0),
        ("carla_leaderboard_mode", False),
    ],
)
def test_preprocessing_changes_invalidate_persistent_and_session_keys(
    config_factory,
    monkeypatch,
    setting,
    changed,
):
    baseline = config_factory()
    assert getattr(baseline, setting) != changed
    original = key(baseline)
    if setting == "carla_leaderboard_mode":
        # A synthetic non-leaderboard CARLA config needs an explicit horizon.
        monkeypatch.setattr(
            TrainingConfig,
            "num_way_points_prediction",
            baseline.num_way_points_prediction,
        )
        monkeypatch.setattr(
            TrainingConfig,
            "waypoints_spacing",
            baseline.waypoints_spacing,
        )
    monkeypatch.setattr(TrainingConfig, setting, changed)
    updated = key(config_factory())
    assert original != updated
    assert str(original) != str(updated)
    assert original.persistent_cache_full_path != updated.persistent_cache_full_path
    assert updated not in {original: "cached"}


@pytest.mark.parametrize(
    ("setting", "changed"),
    [
        ("batch_size", 7),
        ("seed", 234),
        ("compile", False),
        ("compile_mode", "reduce-overhead"),
        ("gpu_color_augmentation", True),
        ("upsample_perspective_logits", True),
        ("cuda_prefetch", True),
        ("use_color_aug_prob", 0.7),
        ("use_sensor_perburtation_prob", 0.7),
        ("used_cameras", [True, False, True]),
        ("perspective_downsample_factor", 2),
        ("crop_height", 10),
        ("horizontal_fov_reduction", 10),
        ("num_radar_points_per_sensor", 50),
        ("training_png_compression_level", 1),
    ],
)
def test_run_and_post_cache_settings_keep_cache_identity(
    config_factory,
    monkeypatch,
    setting,
    changed,
):
    original = key(config_factory())
    monkeypatch.setattr(TrainingConfig, setting, changed)
    updated = key(config_factory())
    assert updated == original
    assert updated.persistent_cache_full_path == original.persistent_cache_full_path
    assert pickle.dumps(updated) == pickle.dumps(original)


def test_disabled_modalities_do_not_fingerprint_unused_label_settings(config_factory):
    first = config_factory(use_radars=False, use_lidar=False, num_radar_queries=2)
    second = config_factory(use_radars=False, use_lidar=False, num_radar_queries=5)
    second.hist_max_per_pixel = 33
    assert key(first) == key(second)


def test_dataset_and_view_identity_is_unambiguous(config_factory, tmp_path):
    original = key(config_factory())
    other_root = key(
        config_factory(carla_root=str(tmp_path / "other" / "carla_leaderboard2")),
    )
    assert other_root != original
    assert key(config_factory(), perturbated=True) != original
    assert key(config_factory(), frame="0021") != original
    assert key(config_factory(), scenario="turn_route", route="1") != original


def test_root_aliases_and_equivalent_configs_share_session_entry(
    config_factory,
    tmp_path,
):
    config = config_factory(batch_size=4, seed=5)
    Path(config.carla_root).mkdir()
    alias = tmp_path / "alias_carla_leaderboard2"
    alias.symlink_to(config.carla_root, target_is_directory=True)
    original = key(config)
    equivalent = key(config_factory(seed=99, batch_size=8, carla_root=str(alias)))
    assert original == equivalent
    with diskcache.Cache(str(tmp_path / "session")) as session:
        session[str(original)] = "legacy string entry"
        session[original] = {"radars": "present"}
        assert session[equivalent] == {"radars": "present"}
        assert key(config_factory(use_radars=False)) not in session


def test_key_snapshots_identity_and_serializes_without_full_run_config(config_factory):
    config = config_factory()
    original = key(config)
    original_path = original.persistent_cache_full_path
    config._unpickleable_run_state = lambda: None
    config.use_radars = False
    assert original.persistent_cache_full_path == original_path
    assert original != key(config)
    restored = pickle.loads(pickle.dumps(original))
    assert restored == original
    assert restored.persistent_cache_full_path == original_path
    assert restored.compatibility == tuple(json.loads(str(original))[2])


def test_preprocessing_schema_and_lookup_changes_invalidate(
    config_factory,
    monkeypatch,
):
    config = config_factory(save_grouped_semantic=False)
    original = key(config)
    converter = dict(constants.SEMANTIC_SEGMENTATION_CONVERTER)
    converter[next(iter(converter))] = 123
    monkeypatch.setattr(constants, "SEMANTIC_SEGMENTATION_CONVERTER", converter)
    assert key(config) != original
    changed_labels = key(config)
    monkeypatch.setattr(sensor_cache_schema, "SENSOR_CACHE_VERSION", 999)
    assert key(config) != changed_labels


def test_old_persistent_namespace_is_left_intact_and_not_reused(config_factory):
    config = config_factory()
    old_path = (
        Path(config.carla_root)
        / "cache/turn/route_1/1152/384/-32/64/-40/40/True/True/True/True/False/10/normal/0020.pkl"
    )
    old_path.parent.mkdir(parents=True)
    old_path.write_bytes(b"legacy cached data")
    cache = PersistentCache(config)
    assert key(config) not in cache
    cache[key(config)] = {"new": True}
    assert cache[key(config)] == {"new": True}
    assert old_path.read_bytes() == b"legacy cached data"


def test_compressed_sensor_roundtrip(config_factory):
    config = config_factory()
    sensor = SensorData(
        image=None,
        rasterized_lidar=np.full((3, 4), 0.3, np.float32),
        semantic=np.arange(12, dtype=np.uint8).reshape(3, 4),
        hdmap=np.ones((3, 3), np.uint8),
        depth=np.ones((3, 4), np.float32),
        boxes=np.ones((2, 9), np.float32),
        boxes_waypoints=np.ones((2, 3, 2), np.float32),
        boxes_num_waypoints=np.array([1, 2], np.int32),
        bev_occupancy=np.ones((3, 3), np.uint8),
        bev_3rd_person_image=None,
        radars=tuple(np.ones((i, 4), np.float16) for i in range(1, 5)),
        radar_detections=np.ones((config.num_radar_queries, 4), np.float32),
    )
    cache = PersistentCache(config)
    cache[key(config)] = sensor.compress(None, config, {})
    restored = cache[key(config)].decompress()
    for name in (
        "semantic",
        "hdmap",
        "boxes",
        "boxes_waypoints",
        "boxes_num_waypoints",
        "bev_occupancy",
        "radar_detections",
    ):
        np.testing.assert_array_equal(getattr(restored, name), getattr(sensor, name))
    for actual, expected in zip(restored.radars, sensor.radars, strict=True):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_allclose(
        restored.rasterized_lidar,
        sensor.rasterized_lidar,
        atol=1 / 65535,
    )


def test_atomic_replace_keeps_readers_on_complete_previous_entry(
    config_factory,
    monkeypatch,
):
    config = config_factory()
    cache = PersistentCache(config)
    identity = key(config)
    cache[identity] = {"version": 1}
    real_dump = pickle.dump

    def observe_during_write(value, stream, **kwargs):
        assert cache[identity] == {"version": 1}
        real_dump(value, stream, **kwargs)
        assert cache[identity] == {"version": 1}

    monkeypatch.setattr(training_cache.pickle, "dump", observe_during_write)
    cache[identity] = {"version": 2}
    assert cache[identity] == {"version": 2}
    assert not list(Path(identity.persistent_cache_full_path).parent.glob("*.tmp"))


def test_failed_write_preserves_previous_cache_and_cleans_temp(
    config_factory,
    monkeypatch,
):
    config = config_factory()
    cache = PersistentCache(config)
    identity = key(config)
    cache[identity] = {"version": 1}

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(training_cache.pickle, "dump", fail)
    with pytest.raises(OSError, match="disk full"):
        cache[identity] = {"version": 2}
    assert cache[identity] == {"version": 1}
    assert not list(Path(identity.persistent_cache_full_path).parent.glob("*.tmp"))


@pytest.mark.parametrize("payload", [b"", b"not lzma", lzma.compress(b"not pickle")])
def test_unreadable_cache_requests_loader_rebuild(config_factory, payload):
    config = config_factory()
    cache = PersistentCache(config)
    identity = key(config)
    path = Path(identity.persistent_cache_full_path)
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)
    assert identity in cache
    with pytest.raises(EOFError, match="Unreadable sensor cache"):
        cache[identity]
    cache[identity] = {"repaired": True}
    assert cache[identity] == {"repaired": True}


def test_file_removed_after_cached_existence_requests_rebuild(config_factory):
    config = config_factory()
    cache = PersistentCache(config)
    identity = key(config)
    cache[identity] = {"version": 1}
    Path(identity.persistent_cache_full_path).unlink()
    with pytest.raises(EOFError, match="Unreadable sensor cache"):
        cache[identity]
    assert identity not in cache
