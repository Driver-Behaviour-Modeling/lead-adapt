"""History timing regression tests using straight motion and a turn across pi."""

import json
from collections import deque
from pathlib import Path

import numpy as np
import pytest

from lead.common.base_agent import BaseAgent
from lead.common.history_features import (
    sample_runtime_history,
    sample_training_history,
)
from lead.data_loader.carla_dataset_utils import (
    perturbate_waypoints,
    perturbate_yaws,
)


def agent_history(turning: bool, count: int = 61):
    """Exercise BaseAgent's actual coordinate conversion without a simulator."""
    times = np.arange(count, dtype=np.float64) / 20
    if turning:
        headings = 2.9 + 0.6 * times
        positions = 8 * np.column_stack((np.sin(headings), -np.cos(headings)))
    else:
        headings = np.zeros(count)
        positions = np.column_stack((8 * times, np.zeros(count)))
    agent = BaseAgent()
    agent.step = count
    agent.smooth_history = np.column_stack((positions, headings, np.full(count, 8.0)))
    agent.compass = headings[-1]
    agent.yaws_queue = deque(headings)
    return agent.ego_past_positions, agent.ego_past_yaws


def legacy_runtime(positions, yaws, count=5, stride=5):
    """Original SensorAgent algorithm, retained independently as a regression oracle."""
    positions = np.asarray(positions, dtype=np.float32).reshape(-1, 2)
    yaws = np.asarray(yaws, dtype=np.float32)
    positions = positions[::-1][::stride][:count][::-1]
    yaws = yaws[::-1][::stride][:count][::-1]
    padding = count - len(positions)
    if padding:
        position_padding = (
            np.tile(positions[:1], (padding, 1))
            if len(positions)
            else np.zeros((padding, 2), dtype=np.float32)
        )
        yaw_padding = (
            np.tile(yaws[:1], padding)
            if len(yaws)
            else np.zeros(padding, dtype=np.float32)
        )
        positions = np.concatenate((position_padding, positions))
        yaws = np.concatenate((yaw_padding, yaws))
    return positions, yaws


@pytest.mark.parametrize("turning", [False, True])
def test_legacy_runtime_matches_metadata_with_actual_base_agent_geometry(turning):
    positions, yaws = agent_history(turning)
    # These are the exact history serialization operations in Expert.save_meta.
    saved_positions = np.array(positions, dtype=np.float32)[::-1]
    saved_yaws = np.array(yaws, dtype=np.float32)[::-1]
    training = sample_training_history(
        saved_positions,
        saved_yaws,
        num_history_poses=5,
        waypoints_spacing=5,
    )
    aligned = sample_runtime_history(
        positions,
        yaws,
        num_history_poses=5,
        waypoints_spacing=5,
        mode="training_aligned",
    )
    legacy = sample_runtime_history(
        positions,
        yaws,
        num_history_poses=5,
        waypoints_spacing=5,
    )
    np.testing.assert_array_equal(training.positions, legacy.positions)
    np.testing.assert_array_equal(training.yaws, legacy.yaws)
    np.testing.assert_array_equal(training.tick_ages, [20, 15, 10, 5, 0])
    np.testing.assert_array_equal(legacy.tick_ages, [20, 15, 10, 5, 0])
    np.testing.assert_array_equal(aligned.tick_ages, [25, 20, 15, 10, 5])
    np.testing.assert_array_equal(training.positions[-1], [0, 0])
    assert training.yaws[-1] == 0
    assert np.linalg.norm(aligned.positions[-1]) > 1
    if turning:
        assert abs(aligned.positions[-1, 1]) > 0.01
        assert aligned.yaws[-1] != 0
        assert np.all(np.abs(training.yaws) <= np.pi)
    else:
        np.testing.assert_allclose(aligned.positions[:, 0], [-10, -8, -6, -4, -2])
        np.testing.assert_array_equal(training.positions[:, 0], [-8, -6, -4, -2, 0])


@pytest.mark.parametrize("length", [0, 1, 4, 5, 6, 11, 20, 21, 25, 26, 61])
def test_legacy_runtime_values_and_padding_are_exactly_unchanged(length):
    positions = np.arange(length * 2, dtype=np.float64).reshape(-1, 2) / 7
    yaws = np.arange(length, dtype=np.float64) / 11
    expected_positions, expected_yaws = legacy_runtime(positions, yaws)
    result = sample_runtime_history(
        positions,
        yaws,
        num_history_poses=5,
        waypoints_spacing=5,
    )
    np.testing.assert_array_equal(result.positions, expected_positions)
    np.testing.assert_array_equal(result.yaws, expected_yaws)
    assert result.positions.dtype == np.float32
    assert result.yaws.dtype == np.float32
    np.testing.assert_array_equal(
        result.padding_mask,
        result.requested_tick_ages >= length,
    )


@pytest.mark.parametrize("length", [0, 5, 6, 20, 26, 61])
@pytest.mark.parametrize("perturbation", [(0.0, 0.0), (0.7, -14.0)])
def test_training_values_remain_exact_before_and_after_sensor_perturbation(
    length,
    perturbation,
):
    positions = np.arange(length * 3, dtype=np.float64).reshape(-1, 3) / 7
    yaws = np.arange(length, dtype=np.float64) / 11
    # CARLAData selection ending at the current pose. Third position coordinates
    # are deliberately ignored.
    indices = [20, 15, 10, 5, 0]
    expected_positions = np.array(
        [positions[i][:2] for i in indices if i < len(positions)],
        dtype=np.float32,
    ).reshape(-1, 2)
    expected_yaws = np.array(
        [yaws[i] for i in indices if i < len(yaws)],
        dtype=np.float32,
    ).reshape(-1)
    result = sample_training_history(
        positions,
        yaws,
        num_history_poses=5,
        waypoints_spacing=5,
    )
    np.testing.assert_array_equal(result.positions, expected_positions)
    np.testing.assert_array_equal(result.yaws, expected_yaws)
    assert not result.padding_mask.any()
    if not len(expected_positions):
        # Dataset startup filtering normally removes these frames. The existing
        # augmentation helper cannot accept empty waypoints; sampling remains empty.
        return
    translation, yaw = perturbation
    np.testing.assert_array_equal(
        perturbate_waypoints(result.positions, translation, yaw),
        perturbate_waypoints(expected_positions, translation, yaw),
    )
    np.testing.assert_array_equal(
        perturbate_yaws(result.yaws, yaw),
        perturbate_yaws(expected_yaws, yaw),
    )


def test_recorded_turn_matches_training_without_changing_its_coordinates():
    fixture_path = Path(__file__).with_name("fixtures") / "history_turn_metadata.json"
    fixture = json.loads(fixture_path.read_text())
    positions = np.array(fixture["past_positions"], dtype=np.float32)
    yaws = np.array(fixture["past_yaws"], dtype=np.float32)
    training = sample_training_history(
        positions,
        yaws,
        num_history_poses=5,
        waypoints_spacing=5,
    )
    aligned = sample_runtime_history(
        positions[::-1],
        yaws[::-1],
        num_history_poses=5,
        waypoints_spacing=5,
        mode="training_aligned",
    )
    legacy = sample_runtime_history(
        positions[::-1],
        yaws[::-1],
        num_history_poses=5,
        waypoints_spacing=5,
    )
    np.testing.assert_array_equal(training.positions, positions[[20, 15, 10, 5, 0]])
    np.testing.assert_array_equal(training.yaws, yaws[[20, 15, 10, 5, 0]])
    np.testing.assert_array_equal(legacy.positions, training.positions)
    np.testing.assert_array_equal(legacy.yaws, training.yaws)
    np.testing.assert_array_equal(aligned.positions, positions[[25, 20, 15, 10, 5]])
    assert np.linalg.norm(aligned.positions[-1] - training.positions[-1]) > 2.7
    assert abs(aligned.yaws[-1] - training.yaws[-1]) > 0.14


def test_sample_timestamps_report_actual_padding_ages():
    positions, yaws = agent_history(False, count=12)
    result = sample_runtime_history(
        positions,
        yaws,
        num_history_poses=5,
        waypoints_spacing=5,
        mode="training_aligned",
    )
    np.testing.assert_array_equal(result.requested_tick_ages, [25, 20, 15, 10, 5])
    np.testing.assert_array_equal(result.tick_ages, [10, 10, 10, 10, 5])
    np.testing.assert_array_equal(result.padding_mask, [True, True, True, False, False])
    np.testing.assert_array_equal(result.timestamps(12.0), [11.5] * 4 + [11.75])


def test_aligned_startup_uses_oldest_observation_until_first_requested_tick_exists():
    positions, yaws = agent_history(False, count=4)
    result = sample_runtime_history(
        positions,
        yaws,
        num_history_poses=5,
        waypoints_spacing=5,
        mode="training_aligned",
    )
    np.testing.assert_array_equal(
        result.positions,
        np.tile(np.float32(positions[0]), (5, 1)),
    )
    np.testing.assert_array_equal(result.tick_ages, [3] * 5)
    assert result.padding_mask.all()


def test_empty_queue_does_not_report_synthetic_zeros_as_observations():
    result = sample_runtime_history(
        [],
        [],
        num_history_poses=5,
        waypoints_spacing=5,
        mode="training_aligned",
    )
    np.testing.assert_array_equal(result.positions, np.zeros((5, 2), dtype=np.float32))
    np.testing.assert_array_equal(result.tick_ages, [-1] * 5)
    assert result.padding_mask.all()
    assert np.isnan(result.timestamps(12)).all()


@pytest.mark.parametrize("count,stride", [(1, 5), (3, 2), (8, 4)])
def test_sampling_contract_generalizes_beyond_checkpoint_defaults(count, stride):
    positions, yaws = agent_history(True)
    training = sample_training_history(
        positions[::-1],
        yaws[::-1],
        num_history_poses=count,
        waypoints_spacing=stride,
    )
    legacy = sample_runtime_history(
        positions,
        yaws,
        num_history_poses=count,
        waypoints_spacing=stride,
        tick_hz=10,
    )
    np.testing.assert_array_equal(legacy.positions, training.positions)
    np.testing.assert_array_equal(legacy.yaws, training.yaws)
    np.testing.assert_array_equal(legacy.time_offsets_seconds, -legacy.tick_ages / 10)


@pytest.mark.parametrize(
    "overrides",
    [
        {"num_history_poses": 0},
        {"waypoints_spacing": 0},
        {"tick_hz": 0},
        {"tick_hz": np.nan},
        {"mode": "unknown"},
    ],
)
def test_invalid_configuration_fails_explicitly(overrides):
    arguments = dict(num_history_poses=5, waypoints_spacing=5)
    arguments.update(overrides)
    with pytest.raises(ValueError):
        sample_runtime_history([[0, 0]], [0], **arguments)


def test_mismatched_pose_and_heading_queues_cannot_silently_desynchronize():
    with pytest.raises(ValueError, match="same ticks"):
        sample_runtime_history(
            [[0, 0], [1, 0]],
            [0],
            num_history_poses=5,
            waypoints_spacing=5,
        )
