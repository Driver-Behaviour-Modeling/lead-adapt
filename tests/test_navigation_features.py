from types import SimpleNamespace

import numpy as np
import pytest

from lead.common.navigation_features import (
    build_navigation_features,
    navigation_pop_settings,
    navigation_position_source,
)


def legacy_navigation(points, commands, position, yaw):
    filtered, codes = [], []
    for point, command in zip(points, commands, strict=True):
        if (
            len(points) == 2
            or not filtered
            or not np.allclose(point[:2], filtered[-1][:2])
        ):
            filtered.append(point)
            codes.append(command)
    rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    selected = [
        filtered[0],
        filtered[1],
        filtered[2] if len(filtered) > 2 else filtered[1],
    ]
    return [rotation.T @ (np.array(point[:2]) - position) for point in selected], codes[
        :2
    ]


@pytest.mark.parametrize(
    "points,commands",
    [
        ([[1.0, 2.0, 0.0], [3.0, 4.0, 0.0], [5.0, 6.0, 0.0]], [4, 1, 2]),
        (
            [[1.0, 2.0, 0.0], [1.0, 2.0, 2.0], [5.0, 6.0, 0.0], [7.0, 8.0, 0.0]],
            [4, 2, 1, 3],
        ),
        ([[1.0, 2.0, 0.0], [1.0, 2.0, 0.0]], [4, 1]),
    ],
)
@pytest.mark.parametrize("yaw", [0.0, 1.2, -3.0])
def test_shared_selection_matches_existing_paths(points, commands, yaw):
    position = np.array([9.0, -7.0])
    expected, codes = legacy_navigation(points, commands, position, yaw)
    actual = build_navigation_features(points, commands, position, yaw)
    for key, value in zip(
        ("target_point_previous", "target_point", "target_point_next"),
        expected,
        strict=True,
    ):
        np.testing.assert_array_equal(actual[key], value)
    assert [actual["command_id"], actual["next_command_id"]] == codes


def test_navigation_is_invariant_to_world_rigid_transform():
    points = np.array([[0.0, 0.0], [5.0, 0.0], [5.0, -5.0]])
    position = np.array([1.0, 0.0])
    yaw = 0.7
    angle = 1.1
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
    )
    translation = np.array([100.0, -230.0])
    first = build_navigation_features(points, [4, 1, 1], position, yaw)
    second = build_navigation_features(
        points @ rotation.T + translation,
        [4, 1, 1],
        position @ rotation.T + translation,
        yaw + angle,
    )
    for key in ("target_point_previous", "target_point", "target_point_next"):
        np.testing.assert_allclose(first[key], second[key], atol=1e-12)
    # CARLA y right: this supplied left-turn target has negative ego y at yaw=0.
    unrotated = build_navigation_features(points, [4, 1, 1], position, 0.0)
    assert unrotated["target_point_next"][1] < 0


def test_augmentation_applies_after_world_transform():
    plain = build_navigation_features(
        [[0.0, 0.0], [4.0, -2.0]],
        [4, 1],
        np.array([1.0, 3.0]),
        0.5,
    )
    augmented = build_navigation_features(
        [[0.0, 0.0], [4.0, -2.0]],
        [4, 1],
        np.array([1.0, 3.0]),
        0.5,
        augment=lambda value: value + [0.0, 2.0],
    )
    for key in ("target_point_previous", "target_point", "target_point_next"):
        np.testing.assert_array_equal(augmented[key], plain[key] + [0.0, 2.0])


def test_ablation_switches_change_only_the_requested_convention():
    training = SimpleNamespace(use_kalman_filter_for_gps=True, tp_pop_distance=3.25)
    online = SimpleNamespace(
        navigation_position_source="legacy",
        navigation_pop_distance_mode="legacy",
        use_kalman_filter=False,
        route_planner_min_distance=5.0,
        sensor_agent_pop_distance_adaptive=True,
    )
    assert navigation_position_source(online, training) == "noisy_state"
    assert navigation_pop_settings(online, training) == (5.0, True)
    online.navigation_position_source = "planner"
    assert navigation_position_source(online, training) == "filtered_state"
    assert navigation_pop_settings(online, training) == (5.0, True)
    training.use_kalman_filter_for_gps = False
    assert navigation_position_source(online, training) == "noisy_state"
    online.navigation_pop_distance_mode = "training"
    assert navigation_pop_settings(online, training) == (3.25, False)


def test_bad_navigation_input_is_rejected():
    with pytest.raises(ValueError, match="equal lengths"):
        build_navigation_features([[1.0, 2.0], [3.0, 4.0]], [1], np.zeros(2), 0.0)
    with pytest.raises(ValueError, match="two"):
        build_navigation_features([[1.0, 2.0]], [1], np.zeros(2), 0.0)
