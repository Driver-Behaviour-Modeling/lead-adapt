"""Visualization accepts the CPU and CUDA inputs the batch pipeline can yield."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lead.visualization import visualizer as viz_module
from lead.visualization.visualizer import Visualizer


@pytest.fixture(params=["cpu", "cuda"])
def tensor_device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device(request.param)


def _visualizer(image_dtype=np.uint8):
    # Exercise the rendering boundaries without constructing unrelated sensor
    # panels or importing a checkpoint.
    visualizer = object.__new__(Visualizer)
    visualizer.config = SimpleNamespace(
        use_radars=True,
        num_radar_sensors=1,
        ego_extent_x=2.5,
        ego_extent_y=1.0,
        pixels_per_meter=2.0,
        min_x_meter=-32.0,
        min_y_meter=-32.0,
    )
    visualizer.origin = (64, 64)
    visualizer.loc_pixels_per_meter = 2.0
    visualizer.scale_factor = 1
    visualizer.bev_image = np.full((128, 128, 3), 255, dtype=image_dtype)
    visualizer.predictions = None
    return visualizer


def test_raw_radar_tensor_rendering_matches_numpy(tensor_device):
    points = np.array(
        [
            [2.0, 3.0, 0.0, 4.0, 0.0],
            [-3.0, 4.0, 1.0, -8.0, 0.0],
            [5.0, -2.0, 0.0, np.nan, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )[None]
    expected = _visualizer()
    expected.data = {"radar1": points.copy()}
    expected._radars(plot_detection_label=False, plot_detection_prediction=False)

    actual = _visualizer()
    tensor = torch.from_numpy(points.copy()).to(tensor_device)
    actual.data = {"radar1": tensor}
    actual._radars(plot_detection_label=False, plot_detection_prediction=False)

    np.testing.assert_array_equal(actual.bev_image, expected.bev_image)
    assert np.any(actual.bev_image != 255)
    np.testing.assert_array_equal(tensor.cpu().numpy(), points)


def test_waypoint_box_tensor_scalars_match_python_floats(tensor_device):
    # BEV semantic blending supplies the floating image draw_box expects.
    expected = _visualizer(np.float32)
    expected._draw_bounding_box_from_waypoint(
        4.0,
        -3.0,
        0.35,
        2.5,
        1.0,
        color=(0, 128, 255),
    )

    actual = _visualizer(np.float32)
    pose = torch.tensor([4.0, -3.0, 0.35], dtype=torch.float64, device=tensor_device)
    actual._draw_bounding_box_from_waypoint(
        pose[0],
        pose[1],
        pose[2],
        2.5,
        1.0,
        color=(0, 128, 255),
    )

    np.testing.assert_array_equal(actual.bev_image, expected.bev_image)
    assert np.any(actual.bev_image != 255)


@pytest.mark.parametrize("prediction_dtype", [torch.float32, torch.bfloat16])
def test_feature_map_logging_with_prefetched_radar(
    tensor_device,
    prediction_dtype,
    monkeypatch,
):
    """Render the actual Matplotlib/W&B path, including valid radar labels."""
    data = {
        key: torch.zeros(1, 1, 4, 4, device=tensor_device)
        for key in (
            "rasterized_lidar",
            "center_net_heatmap",
            "center_net_wh",
            "center_net_yaw_res",
            "center_net_offset",
            "center_net_velocity",
        )
    }
    for key in ("center_net_yaw_class", "bev_semantic"):
        data[key] = torch.zeros(1, 4, 4, dtype=torch.long, device=tensor_device)
    data["radar"] = torch.tensor(
        [[[2.0, 3.0, 0.0, 4.0], [-3.0, 4.0, 0.0, -8.0]]],
        device=tensor_device,
        dtype=prediction_dtype,
    )
    data["radar_detections"] = torch.tensor(
        [[[2.0, 3.0, 4.0, 1.0], [-3.0, 4.0, -8.0, 0.0]]],
        device=tensor_device,
    )
    predictions = SimpleNamespace(
        pred_bounding_box=SimpleNamespace(
            center_heatmap_pred=torch.zeros(1, 2, 4, 4, device=tensor_device),
        ),
        pred_bev_semantic=torch.zeros(1, 2, 4, 4, device=tensor_device),
        pred_future_waypoints=torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0]]],
            device=tensor_device,
            dtype=prediction_dtype,
            requires_grad=True,
        ),
        pred_route=torch.tensor(
            [[[2.0, 1.0], [4.0, 3.0]]],
            device=tensor_device,
            dtype=prediction_dtype,
            requires_grad=True,
        ),
        pred_radar_predictions=torch.tensor(
            [[[5.0, 6.0, -2.0, 1.0], [7.0, 8.0, 3.0, -1.0]]],
            device=tensor_device,
            dtype=prediction_dtype,
            requires_grad=True,
        ),
    )
    config = SimpleNamespace(
        min_x_meter=-32,
        max_x_meter=64,
        min_y_meter=-40,
        max_y_meter=40,
        log_wandb=True,
    )
    logged = []

    def record_log(payload, commit):
        assert not commit
        assert payload["train_viz/feature_maps"].size[0] > 0
        axes = viz_module.plt.gcf().axes
        assert len(axes[12].collections) == 2  # Raw radar returns.
        assert len(axes[13].collections) == 1  # Invalid label filtered out.
        assert len(axes[14].collections) == 1  # Negative logit filtered out.
        np.testing.assert_array_equal(
            axes[13].collections[0].get_offsets(),
            [[2.0, 3.0]],
        )
        np.testing.assert_array_equal(axes[13].collections[0].get_sizes(), [14.0])
        np.testing.assert_array_equal(
            axes[14].collections[0].get_offsets(),
            [[5.0, 6.0]],
        )
        logged.append(payload)

    # Keep real plotting and PNG encoding; replace only the remote W&B boundary.
    monkeypatch.setattr(viz_module.wandb, "Image", lambda img: img.copy())
    monkeypatch.setattr(viz_module.wandb, "log", record_log)
    figures_before = viz_module.plt.get_fignums()
    try:
        viz_module.visualize_feature_maps(config, predictions, data, log_wandb=True)
        assert len(logged) == 1
        assert viz_module.plt.get_fignums() == figures_before
        assert data["radar_detections"].device.type == tensor_device.type
        assert predictions.pred_route.requires_grad
    finally:
        for figure in set(viz_module.plt.get_fignums()) - set(figures_before):
            viz_module.plt.close(figure)
