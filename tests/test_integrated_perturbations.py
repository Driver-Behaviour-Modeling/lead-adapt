"""Zero-change invariance, non-reversing braking, and replay boundary handling."""

import numpy as np
import pytest
import torch

from lead.scoring.perturbations import (
    contiguous_window_means,
    perturb_ego_motion,
    severity_correlations,
    upper_tail_probability,
)


def test_zero_perturbation_preserves_turns_slip_and_wrapped_headings():
    poses = torch.tensor(
        [[[1.0, 0.2, 1], [0.6, 1.4, 3.13], [-0.4, 1.6, -2.9], [-0.4, 1.6, -2.8]]],
        dtype=torch.float64,
    )
    torch.testing.assert_close(perturb_ego_motion(poses), poses)


def test_residual_acceleration_has_expected_first_step_and_no_reverse_braking():
    poses = torch.zeros(1, 8, 3, dtype=torch.float64)
    poses[..., 0] = torch.arange(1, 9) * 1.0
    acceleration = perturb_ego_motion(poses, longitudinal_acceleration=4)
    assert acceleration[0, 0, 0].item() == pytest.approx(1 + 0.5 * 4 * 0.25**2)
    braking = perturb_ego_motion(poses, longitudinal_acceleration=-6)
    assert (torch.diff(braking[..., 0]) >= 0).all()
    assert braking[0, -1, 0] < poses[0, -1, 0]
    left = perturb_ego_motion(poses, lateral_acceleration=2)
    right = perturb_ego_motion(poses, lateral_acceleration=-2)
    torch.testing.assert_close(left[..., 0], right[..., 0])
    torch.testing.assert_close(left[..., 1:], -right[..., 1:])
    assert left[0, 0, 1] > 0


def test_windows_never_cross_drive_gaps_or_ineligible_frames():
    values = np.arange(7, dtype=float)
    route = np.array(["a"] * 5 + ["b"] * 2)
    frames = np.array([0, 1, 3, 4, 5, 0, 1])
    np.testing.assert_allclose(
        contiguous_window_means(values, route, frames, 2),
        [0.5, 2.5, 3.5, 5.5],
    )
    eligible = np.ones(7, dtype=bool)
    eligible[3] = False
    np.testing.assert_allclose(
        contiguous_window_means(values, route, frames, 2, eligible),
        [0.5, 5.5],
    )
    assert not len(contiguous_window_means(values, route, frames, 8))


def test_empirical_window_ranks_count_ties_conservatively():
    np.testing.assert_allclose(
        upper_tail_probability([0, 1, 2], [0, 0, 1]),
        [1, 0.5, 0.25],
    )


def test_severity_ranking_handles_ties_and_reports_constant_contexts():
    scores = np.array([[0, 2, 7, 0], [1, 1, 7, 0], [2, 0, 7, 2]])
    actual = severity_correlations([1, 2, 4], scores)
    np.testing.assert_allclose(actual[[0, 1, 3]], [1, -1, np.sqrt(3) / 2])
    assert np.isnan(actual[2])
