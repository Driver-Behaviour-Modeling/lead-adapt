"""Align actual logged motion to the exact sensor snapshot frame and horizon."""

import numpy as np
import pytest

from scripts.score_integrated_snapshots import observed_future


def trace():
    return [
        {
            "step": step,
            "timestamp_seconds": step * 0.05,
            "offline_ground_truth": {
                "position_world_m": [100, 200 + step * 0.05, 3],
                "yaw_degrees": 90,
            },
        }
        for step in range(81)
    ]


def test_world_pose_motion_is_rotated_into_current_forward_frame():
    result = observed_future(trace(), 10)
    np.testing.assert_allclose(result[:, 0], np.arange(1, 9) * 0.25)
    np.testing.assert_allclose(result[:, 1:], 0, atol=1e-12)


def test_wrapped_headings_interpolate_continuously():
    rows = trace()
    for row in rows:
        yaw = 175 + 10 * row["timestamp_seconds"]
        row["offline_ground_truth"]["yaw_degrees"] = (yaw + 180) % 360 - 180
    result = observed_future(rows, 0)
    np.testing.assert_allclose(result[:, 2], np.deg2rad(np.arange(1, 9) * 2.5))


def test_incomplete_horizons_and_pose_logging_gaps_are_rejected():
    with pytest.raises(ValueError, match="Complete observed future"):
        observed_future(trace(), 70)
    rows = [row for row in trace() if row["step"] != 15]
    with pytest.raises(ValueError, match="logging gap"):
        observed_future(rows, 10)
