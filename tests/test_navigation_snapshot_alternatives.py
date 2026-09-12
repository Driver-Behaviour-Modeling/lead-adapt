"""Navigation snapshots must use live planner state without affecting rollout."""

import importlib
from collections import deque
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


@pytest.fixture
def sensor_agent_module(monkeypatch):
    project = Path(__file__).resolve().parents[1]
    for relative in (
        "3rd_party/Bench2Drive/leaderboard",
        "3rd_party/Bench2Drive/scenario_runner",
        "3rd_party/CARLA_0915/PythonAPI/carla",
    ):
        monkeypatch.syspath_prepend(str(project / relative))
    return importlib.import_module("lead.inference.sensor_agent")


def navigation_agent(module):
    agent = module.SensorAgent.__new__(module.SensorAgent)
    agent.config_closed_loop = SimpleNamespace(
        navigation_position_source="legacy",
        navigation_pop_distance_mode="legacy",
        use_kalman_filter=False,
        route_planner_min_distance=5.0,
        sensor_agent_pop_distance_adaptive=True,
        sensor_agent_skip_distant_target_point=True,
        sensor_agent_skip_distant_target_point_threshold=50.0,
    )
    agent.training_config = SimpleNamespace(
        use_kalman_filter_for_gps=True,
        tp_pop_distance=3.25,
    )
    agent.compass = 0.0

    def planner(points, commands):
        def forbidden_step(*args):
            raise AssertionError("Snapshot collection must not advance a planner")

        return SimpleNamespace(
            route=deque(zip(np.asarray(points, dtype=float), commands, strict=True)),
            run_step=forbidden_step,
        )

    agent.gps_waypoint_planners_dict = {
        5.0: planner([[0, 0], [6, 0], [12, -4]], [4, 1, 4]),
        4.0: planner([[1, 0], [8, -2], [80, 0]], [1, 4, 4]),
        3.25: planner([[0, 0], [4, 0], [12, -8]], [4, 1, 4]),
    }
    data = {
        "noisy_state": np.array([0.0, 0.0, 0.0]),
        "filtered_state": np.array([1.0, 2.0, 0.0, 5.0]),
        "radar": np.array([[1.0, 2.0, 3.0, 4.0]]),
    }
    return agent, data


def test_snapshots_use_real_pop_routes_and_preserve_rollout(sensor_agent_module):
    agent, data = navigation_agent(sensor_agent_module)
    agent.update_navigation(data)
    np.testing.assert_array_equal(data["target_point"], [8.0, -2.0])
    np.testing.assert_array_equal(data["target_point_next"], [8.0, -2.0])
    assert agent._navigation_trace["pop_distance_m"] == 4.0
    assert agent._navigation_trace["distant_next_skipped"] is True
    original_data = deepcopy(data)
    original_config = vars(agent.config_closed_loop).copy()
    trace = agent._navigation_trace
    routes = {
        distance: deepcopy(list(planner.route))
        for distance, planner in agent.gps_waypoint_planners_dict.items()
    }

    variants = agent.snapshot_navigation_alternatives(data)
    assert set(variants) == {
        "baseline",
        "localized_navigation",
        "training_pop_distance",
    }
    for key, tensor in variants["baseline"]["inputs"].items():
        torch.testing.assert_close(
            tensor,
            torch.tensor(data[key], dtype=torch.float32).reshape(1, -1),
            rtol=0,
            atol=0,
        )
    localized = variants["localized_navigation"]
    torch.testing.assert_close(
        localized["inputs"]["target_point"],
        torch.tensor([[7.0, -4.0]]),
    )
    assert localized["navigation"]["target_position_source"] == "filtered_state"
    assert localized["navigation"]["pop_distance_m"] == 4.0
    training = variants["training_pop_distance"]
    assert training["navigation"]["pop_distance_m"] == 3.25
    assert training["navigation"]["target_position_source"] == "noisy_state"
    assert training["navigation"]["distant_next_skipped"] is False
    torch.testing.assert_close(
        training["inputs"]["target_point"],
        torch.tensor([[4.0, 0.0]]),
    )
    torch.testing.assert_close(
        training["inputs"]["target_point_next"],
        torch.tensor([[12.0, -8.0]]),
    )
    assert training["inputs"]["command"].argmax().item() == 3
    assert training["inputs"]["next_command"].argmax().item() == 0
    assert agent._navigation_trace is trace
    assert vars(agent.config_closed_loop) == original_config
    for key in data:
        np.testing.assert_array_equal(data[key], original_data[key])
    for distance, planner in agent.gps_waypoint_planners_dict.items():
        for (point, command), (old_point, old_command) in zip(
            planner.route,
            routes[distance],
            strict=True,
        ):
            np.testing.assert_array_equal(point, old_point)
            assert command == old_command


def test_localization_alternative_recomputes_adaptive_transition(sensor_agent_module):
    agent, data = navigation_agent(sensor_agent_module)
    agent.gps_waypoint_planners_dict[5.0].route = deque(
        zip(np.array([[9.5, 0.0], [15.0, 0.0], [40.0, 20.0]]), [4, 1, 4], strict=True),
    )
    data["filtered_state"] = np.array([-1.0, 0.0, 0.0, 5.0])
    agent.update_navigation(data)
    variants = agent.snapshot_navigation_alternatives(data)
    assert variants["baseline"]["navigation"]["pop_distance_m"] == 4.0
    localized = variants["localized_navigation"]
    assert localized["navigation"]["pop_distance_m"] == 5.0
    torch.testing.assert_close(
        localized["inputs"]["target_point"],
        torch.tensor([[16.0, 0.0]]),
    )
    assert agent._navigation_trace["pop_distance_m"] == 4.0


def test_terminal_duplicate_targets_and_skip_disabled(sensor_agent_module):
    agent, data = navigation_agent(sensor_agent_module)
    agent.config_closed_loop.sensor_agent_pop_distance_adaptive = False
    agent.config_closed_loop.sensor_agent_skip_distant_target_point = False
    agent.gps_waypoint_planners_dict[5.0].route = deque(
        [(np.array([60.0, -2.0]), 4), (np.array([60.0, -2.0]), 4)],
    )
    trace = agent.update_navigation(data)
    assert trace["pop_distance_m"] == 5.0
    assert trace["distant_next_skipped"] is False
    np.testing.assert_array_equal(data["target_point"], [60.0, -2.0])
    np.testing.assert_array_equal(data["target_point_next"], [60.0, -2.0])


def test_alternatives_do_not_modify_actual_config_overrides(
    sensor_agent_module,
    monkeypatch,
):
    monkeypatch.setattr("sys.argv", ["test"])
    monkeypatch.delenv("LEAD_OPEN_LOOP_CONFIG", raising=False)
    monkeypatch.setenv(
        "LEAD_CLOSED_LOOP_CONFIG",
        "sensor_agent_pop_distance_adaptive=false",
    )
    agent, data = navigation_agent(sensor_agent_module)
    agent.config_closed_loop = sensor_agent_module.ClosedLoopConfig()
    before = deepcopy(vars(agent.config_closed_loop))
    agent.update_navigation(data)
    variants = agent.snapshot_navigation_alternatives(data)
    assert variants["baseline"]["navigation"]["pop_distance_m"] == 5.0
    assert variants["localized_navigation"]["navigation"]["pop_distance_m"] == 5.0
    assert variants["training_pop_distance"]["navigation"]["pop_distance_m"] == 3.25
    assert vars(agent.config_closed_loop) == before
    assert agent.config_closed_loop.navigation_position_source == "legacy"
    assert agent.config_closed_loop.navigation_pop_distance_mode == "legacy"
