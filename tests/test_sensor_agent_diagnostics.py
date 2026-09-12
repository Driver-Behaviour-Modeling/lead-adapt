"""Exercise real SensorAgent entry points with CPU inputs and simulator stubs."""

import importlib
import json
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import carla
import numpy as np
import pytest
import torch

from lead.inference.driving_diagnostics import DrivingDiagnostics
from lead.inference.model_snapshots import NAVIGATION_INPUT_KEYS, ModelSnapshots


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


def make_agent(module, recorder, *, initial_braking=False):
    agent = module.SensorAgent.__new__(module.SensorAgent)
    agent.step = 0
    agent.initialized = True
    agent.driving_diagnostics = recorder
    agent.device = torch.device("cpu")
    agent.training_config = SimpleNamespace(
        use_radars=True,
        use_adapt_decoder=True,
        num_history_poses=5,
        waypoints_spacing=5,
        carla_fps=20,
        inital_frames_delay=3 if initial_braking else 0,
    )
    agent.config_closed_loop = SimpleNamespace(
        adapt_history_mode="legacy",
        save_path=None,
        carla_frame_rate=0.05,
        is_bench2drive=False,
    )
    agent._world = SimpleNamespace(
        get_map=lambda: SimpleNamespace(name="/Game/Carla/Maps/Town03"),
    )
    agent._vehicle = SimpleNamespace(
        get_transform=lambda: carla.Transform(
            carla.Location(x=23, y=-17, z=1),
            carla.Rotation(yaw=43),
        ),
    )
    agent.compass = 0.0
    agent.smooth_history = np.column_stack(
        (np.arange(31) * 0.4, np.zeros(31), np.zeros(31), np.full(31, 8.0)),
    )
    agent.yaws_queue = deque(np.zeros(31))
    agent._navigation_trace = {"pop_distance_m": 5.0, "command_id": 4}
    agent.bb_buffer = deque(maxlen=1)
    agent.force_move_post_processor = SimpleNamespace(
        adjust=lambda speed, throttle, brake: (throttle + 0.1, brake),
        stuck_detector=0,
        force_move=0,
    )
    agent.stop_sign_post_processor = SimpleNamespace(
        update_stop_box=lambda *args: None,
        adjust=lambda speed, throttle, brake: (throttle - 0.05, brake),
        stop_sign_buffer=[],
    )
    agent.meters_travelled = 0.0
    agent.check_infractions = lambda: None
    tick_data = {
        "rgb": np.zeros((3, 4, 4), dtype=np.uint8),
        "rasterized_lidar": np.zeros((1, 1, 4, 4), dtype=np.float32),
        "target_point_previous": np.array([-3.0, 0.0]),
        "target_point": np.array([8.0, -1.0]),
        "target_point_next": np.array([18.0, -5.0]),
        "command": np.array([0, 0, 0, 1, 0, 0], dtype=np.float32),
        "next_command": np.array([1, 0, 0, 0, 0, 0], dtype=np.float32),
        "speed": np.float64(8.0),
        "radar": np.zeros((2, 6), dtype=np.float32),
        "noisy_state": np.array([23.1, -17.2, 1.0]),
        "filtered_state": np.array([23.05, -17.1, 0.0, 8.0]),
    }
    agent.tick = lambda incoming: tick_data
    forward_inputs = []

    def forward(data):
        forward_inputs.append(
            {
                key: value.clone()
                for key, value in data.items()
                if torch.is_tensor(value)
            },
        )
        return SimpleNamespace(
            steer=0.25,
            throttle=0.4,
            brake=0.0,
            pred_bounding_box_vehicle_system=None,
            pred_route=torch.tensor([[[2.0, 0.0], [4.0, -0.2], [6.0, -1.0]]]),
            pred_future_waypoints=None,
            pred_target_speed_scalar=torch.tensor([9.0]),
        )

    agent.closed_loop_inference = SimpleNamespace(
        forward=forward,
        predictions=[
            SimpleNamespace(
                pred_radar_predictions={"motion": torch.ones(2)},
                pred_route=torch.tensor([[[2.0, 0.0], [4.0, -0.2], [6.0, -1.0]]]),
                pred_future_waypoints=None,
                pred_headings=None,
                pred_target_speed_distribution=torch.tensor([[-5.0, 2.0, 1.0]]),
                pred_target_speed_scalar=None,
            ),
        ],
        nets=[SimpleNamespace()],
    )
    return agent, forward_inputs


@pytest.mark.parametrize("initial_braking", [False, True])
def test_selected_snapshot_preserves_inputs_and_executed_controls(
    sensor_agent_module,
    tmp_path,
    initial_braking,
):
    diagnostics = DrivingDiagnostics(tmp_path, {})
    agent, captured_inputs = make_agent(
        sensor_agent_module,
        diagnostics,
        initial_braking=initial_braking,
    )
    plain, plain_inputs = make_agent(
        sensor_agent_module,
        None,
        initial_braking=initial_braking,
    )
    recorder = ModelSnapshots(
        tmp_path,
        [1],
        metadata={},
        device="cpu",
        autocast_enabled=False,
        autocast_dtype=torch.bfloat16,
    )
    agent.model_snapshots = recorder
    alternative_calls = []

    def navigation_alternatives(data):
        alternative_calls.append(data)
        return {
            "baseline": {
                "inputs": {
                    key: torch.tensor(data[key], dtype=torch.float32).reshape(1, -1)
                    for key in NAVIGATION_INPUT_KEYS
                },
                "navigation": {"command_id": 4},
            },
        }

    agent.snapshot_navigation_alternatives = navigation_alternatives
    sensors = {"rgb_1": (4321, None), "gps": (4321, None)}
    captured_control = agent.run_step(sensors, 7.25)
    plain_control = plain.run_step(sensors, 7.25)
    recorder.close()
    diagnostics.close()
    assert len(alternative_calls) == 1
    for key in ("steer", "throttle", "brake"):
        assert getattr(captured_control, key) == getattr(plain_control, key)
    snapshot = torch.load(recorder.directory / "00001.pth", weights_only=True)
    assert snapshot["timestamp_seconds"] == 7.25
    assert snapshot["inputs"]["town"] == ["Town03"]
    for key, value in captured_inputs[0].items():
        torch.testing.assert_close(value, plain_inputs[0][key], rtol=0, atol=0)
        torch.testing.assert_close(value, snapshot["inputs"][key], rtol=0, atol=0)
    assert "steer" not in snapshot["inputs"]
    assert "brake" not in snapshot["inputs"]
    assert "offline_ground_truth" not in snapshot["inputs"]
    assert snapshot["model_predictions"][0]["pred_target_speed_scalar"] is None
    torch.testing.assert_close(
        snapshot["model_predictions"][0]["pred_target_speed_distribution"],
        torch.tensor([[-5.0, 2.0, 1.0]]),
        rtol=0,
        atol=0,
    )
    assert snapshot["decoder_sensor_features"] == [None]


@pytest.mark.parametrize("selected", [False, True])
def test_unselected_snapshot_skips_alternative_computation(
    sensor_agent_module,
    tmp_path,
    selected,
):
    agent, _ = make_agent(sensor_agent_module, None)
    recorder = None
    if selected:
        recorder = ModelSnapshots(
            tmp_path,
            [5],
            metadata={},
            device="cpu",
            autocast_enabled=False,
            autocast_dtype=torch.bfloat16,
        )
        agent.model_snapshots = recorder
    agent.snapshot_navigation_alternatives = lambda data: pytest.fail(
        "unselected snapshot computed navigation",
    )
    agent.run_step({}, 0.05)
    if recorder is not None:
        recorder.close()
        assert not list(recorder.directory.glob("*.pth"))


@pytest.mark.parametrize("initial_braking", [False, True])
def test_diagnostics_preserve_actual_policy_inputs_and_executed_controls(
    sensor_agent_module,
    tmp_path,
    initial_braking,
):
    recorder = DrivingDiagnostics(tmp_path, {})
    recorded, recorded_inputs = make_agent(
        sensor_agent_module,
        recorder,
        initial_braking=initial_braking,
    )
    plain, plain_inputs = make_agent(
        sensor_agent_module,
        None,
        initial_braking=initial_braking,
    )
    raw_sensors = {"rgb_1": (4321, None), "gps": (4321, None), "unused": "value"}
    recorded_control = recorded.run_step(raw_sensors, 7.25)
    plain_control = plain.run_step(raw_sensors, 7.25)
    recorder.close()
    for key in ("steer", "throttle", "brake"):
        assert getattr(recorded_control, key) == getattr(plain_control, key)
    assert recorded_inputs[0].keys() == plain_inputs[0].keys()
    for key in recorded_inputs[0]:
        torch.testing.assert_close(
            recorded_inputs[0][key],
            plain_inputs[0][key],
            rtol=0,
            atol=0,
        )
    rows = [json.loads(line) for line in recorder.path.read_text().splitlines()]
    record = rows[1]
    assert record["step"] == 1
    assert record["timestamp_seconds"] == 7.25
    assert record["sensor_frames"] == {"rgb_1": 4321, "gps": 4321}
    assert record["history"]["actual_tick_ages"] == [20, 15, 10, 5, 0]
    assert record["history"]["timestamps_seconds"] == [6.25, 6.5, 6.75, 7.0, 7.25]
    assert record["controller_control"] == {
        "steer": 0.25,
        "throttle": 0.4,
        "brake": 0.0,
    }
    assert record["executed_control"]["brake"] == (1.0 if initial_braking else 0.0)
    assert record["executed_control"]["throttle"] == pytest.approx(
        0.0 if initial_braking else 0.45,
    )
    assert record["interventions"]["initial_braking"] is initial_braking
    assert record["offline_ground_truth"]["position_world_m"] == [23.0, -17.0, 1.0]
    assert "offline_ground_truth" not in recorded_inputs[0]
    assert record["radar_predictions_per_model"] == [{"motion": [1.0, 1.0]}]


@pytest.mark.parametrize("filtered", [False, True])
def test_set_target_points_uses_the_original_selected_localization(
    sensor_agent_module,
    filtered,
):
    agent = sensor_agent_module.SensorAgent.__new__(sensor_agent_module.SensorAgent)
    agent.config_closed_loop = SimpleNamespace(
        navigation_position_source="legacy",
        use_kalman_filter=filtered,
    )
    agent.training_config = SimpleNamespace(use_kalman_filter_for_gps=True)
    agent.compass = 0.8
    agent.filtered_state = np.array([4.0, 7.0, 0.8, 5.0])
    input_data = {
        "filtered_state": agent.filtered_state,
        "noisy_state": np.array([4.2, 6.7, 0.0]),
    }
    points = [np.array([2.0, 3.0]), np.array([6.0, 8.0]), np.array([9.0, 10.0])]
    agent.gps_waypoint_planners_dict = {
        5.0: SimpleNamespace(route=list(zip(points, [4, 1, 2], strict=True))),
    }
    agent.set_target_points(input_data, 5.0)
    position = agent.filtered_state[:2] if filtered else input_data["noisy_state"][:2]
    for key, point in zip(
        ("target_point_previous", "target_point", "target_point_next"),
        points,
        strict=True,
    ):
        expected = sensor_agent_module.common_utils.inverse_conversion_2d(
            point,
            position,
            agent.compass,
        )
        np.testing.assert_array_equal(input_data[key], expected)
    assert agent._navigation_trace["target_position_source"] == (
        "filtered_state" if filtered else "noisy_state"
    )
    assert agent._navigation_trace["planner_position_source"] == "filtered_state"


def test_history_ablation_changes_only_the_history_model_inputs(sensor_agent_module):
    legacy, legacy_inputs = make_agent(sensor_agent_module, None)
    aligned, aligned_inputs = make_agent(sensor_agent_module, None)
    aligned.config_closed_loop.adapt_history_mode = "training_aligned"
    legacy.run_step({}, 7.25)
    aligned.run_step({}, 7.25)
    for key in legacy_inputs[0]:
        if key not in {"past_positions", "past_yaws"}:
            torch.testing.assert_close(
                legacy_inputs[0][key],
                aligned_inputs[0][key],
                rtol=0,
                atol=0,
            )
    torch.testing.assert_close(
        legacy_inputs[0]["past_positions"][0, :, 0],
        torch.tensor([-8.0, -6.0, -4.0, -2.0, 0.0]),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        aligned_inputs[0]["past_positions"][0, :, 0],
        torch.tensor([-10.0, -8.0, -6.0, -4.0, -2.0]),
        rtol=0,
        atol=0,
    )
