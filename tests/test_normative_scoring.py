"""Probability algebra, sparse-bin policy, leakage rejection, and world geometry."""

import math

import numpy as np
import pytest
import torch

from lead.scoring.calibration import SceneCalibration, drive_percentiles
from lead.scoring.distributions import (
    MotionSmoother,
    ensemble_block_log_likelihood,
    ensemble_step_statistics,
)
from lead.scoring.reference import NormativeTokenScorer
from lead.scoring.world_interactions import forecast_error_sums, interaction_features


def test_smoothing_kernel_is_column_stochastic_and_not_row_stochastic():
    codes = torch.tensor([[0, 0, 0], [0.1, 0, 0], [10, 0, 0]])
    smoother = MotionSmoother(codes, 2.67, 0.1, 0.5)
    kernel = smoother.log_kernel.exp()
    torch.testing.assert_close(kernel.sum(0), torch.ones(3, dtype=torch.float64))
    p = torch.tensor([[0.2, 0.3, 0.5]], dtype=torch.float64)
    actual = smoother(p.log()).exp()
    torch.testing.assert_close(actual, 0.9 * p + 0.1 * (kernel @ p.T).T)
    torch.testing.assert_close(actual.sum(-1), torch.ones(1, dtype=torch.float64))


def test_tail_smoothing_survives_extreme_distance_without_uniform_floor():
    smoother = MotionSmoother(torch.tensor([[0, 0, 0], [100, 0, 0]]), 1, 0.01, 0.01)
    result = smoother(torch.tensor([[100000, -100000.0]]))
    assert torch.isfinite(result).all()
    assert result[0, 1] < -9000
    assert result[0, 1] > -11000


def test_reference_scores_nearest_motion_and_distance_ordered_tail():
    codes = torch.tensor([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]])
    scorer = NormativeTokenScorer(codes, 2.67, [1.0], epsilon=0.01, sigma=0.5)
    logits = torch.tensor([[[100.0, -100, -100, -100]]]).expand(1, 4, 4)
    result = scorer(logits, codes)
    torch.testing.assert_close(result["token"], torch.arange(4))
    assert (result["surprisal"][1:] > result["surprisal"][:-1]).all()
    assert (result["quantization_residual"] == 0).all()


def test_uncertainty_bins_have_reasonable_flags_on_independent_normal_sample():
    random = np.random.default_rng(32)
    score = random.normal(size=12000)
    a, i = random.uniform(size=(2, 12000))
    fit = SceneCalibration.fit(score, a, i, ["cal"] * len(score), min_bin_size=1000)
    test = fit.evaluate(
        random.normal(size=12000),
        *random.uniform(size=(2, 12000)),
        ["test"] * 12000,
    )
    assert 0.035 < test["flag"].mean() < 0.065


def test_entropy_decomposition_matches_revised_pdf_example():
    members = torch.tensor([[[0.9, 0.1]], [[0.1, 0.9]]], dtype=torch.float64)
    result = ensemble_step_statistics(members.log(), torch.tensor([0]))
    assert result["original_score"].item() == pytest.approx(0)
    assert result["entropy"].item() == pytest.approx(math.log(2))
    assert result["aleatoric"].item() == pytest.approx(0.325082973)
    assert result["epistemic"].item() == pytest.approx(0.368064207)
    assert result["score"].item() == pytest.approx(result["epistemic"].item())


def test_single_member_reduces_to_original_score():
    p = torch.tensor([[[0.9, 0.1]]], dtype=torch.float64)
    result = ensemble_step_statistics(p.log(), torch.tensor([1]))
    assert result["epistemic"].item() == pytest.approx(0)
    assert result["score"].item() == pytest.approx(-math.log(0.1) - 0.325082973)


def test_block_mixture_keeps_member_identity_throughout_sequence():
    observed = torch.tensor([[[0.9, 0.1]], [[0.1, 0.9]]], dtype=torch.float64).log()
    result = ensemble_block_log_likelihood(observed)
    assert result.item() == pytest.approx(math.log(0.09))
    assert result.item() != pytest.approx(math.log(0.25))


@pytest.mark.parametrize("temperature", [0, -1, float("nan")])
def test_invalid_temperature_is_rejected(temperature):
    smoother = MotionSmoother(torch.zeros(2, 3), 1, 0.1, 0.5)
    with pytest.raises(ValueError, match="Temperature"):
        smoother(torch.zeros(1, 2), temperature)


def test_scene_calibration_uses_uncertainty_not_observed_score_for_bins():
    score = np.arange(1000) / 100
    a = np.r_[np.zeros(500), np.ones(500)]
    i = np.zeros_like(a)
    fit = SceneCalibration.fit(score, a, i, ["cal"] * 1000, bins=2, min_bin_size=400)
    result = fit.evaluate([1, 1000], [0, 0], [0, 0], ["test", "test"])
    assert result["bin"][0] == result["bin"][1]
    assert result["exceedance_probability"][1] == pytest.approx(1 / 501)
    assert result["flag"][1]
    restored = SceneCalibration.from_dict(fit.to_dict())
    assert restored.to_dict() == fit.to_dict()
    with pytest.raises(ValueError, match="overlap"):
        fit.evaluate([1], [0], [0], ["cal"])


def test_sparse_bins_are_explicitly_pooled_and_ties_conservative():
    score = np.ones(20)
    fit = SceneCalibration.fit(
        score,
        np.arange(20),
        np.arange(20),
        ["cal"] * 20,
        bins=4,
        min_bin_size=10,
    )
    result = fit.evaluate([1], [0], [0], ["test"])
    assert result["bin"].item() == "pooled"
    assert result["exceedance_probability"].item() == 1
    assert not result["flag"].item()


def test_drive_reference_must_be_separate():
    assert drive_percentiles([10], [0, 1, 2], ["test"], ["a", "b", "c"]).item() == 0.25
    with pytest.raises(ValueError, match="overlap"):
        drive_percentiles([10], [0], ["test"], ["test"])


def futures(y):
    position = torch.zeros(1, 1, 2, 8, 2)
    position[..., 0] = torch.arange(1, 9).float()
    position[..., 1] = y
    velocity = torch.zeros_like(position)
    velocity[..., 0] = 4
    return {
        "future_positions": position,
        "future_velocities": velocity,
        "memory_confidence": torch.tensor([[0.5, 0.5]]),
    }


def test_world_interaction_respects_lane_clearance_and_mode_probabilities():
    ego = torch.zeros(1, 8, 3)
    ego[..., 0] = torch.arange(1, 9).float()
    safe = interaction_features(ego, futures(3.5))
    conflict = interaction_features(ego, futures(0))
    assert safe["expected_conflicting_actor_count"].item() == 0
    assert conflict["expected_conflicting_actor_count"].item() == 1
    uncertain = futures(0)
    uncertain["future_positions"][:, :, 1, :, 1] = 10
    assert (
        interaction_features(ego, uncertain)["expected_conflicting_actor_count"].item()
        == 0.5
    )
    assert (
        safe["soft_expected_conflicting_actor_count"].item()
        < conflict["soft_expected_conflicting_actor_count"].item()
    )


def test_forecast_metrics_use_current_assignment_and_mask_unknown_futures():
    target = torch.tensor(
        [[[[1.0, 0], [2, 0]], [[float("nan"), 0], [float("nan"), 0]]]],
    )
    position = torch.zeros(1, 2, 2, 2, 2)
    position[:, 1, 0] = torch.tensor([[1.0, 1], [2, 1]])
    position[:, 1, 1] = torch.tensor([[1.0, 2], [2, 2]])
    data = {
        "world_future_positions": target,
        "world_future_mask": torch.tensor([[[True, True], [False, False]]]),
        "world_vehicle_mask": torch.ones(1, 2, dtype=torch.bool),
        "radar_detections": torch.ones(1, 2, 4),
    }
    prediction = {
        "future_positions": position,
        "mode_logits": torch.tensor([[[0.0, 1], [0, 1]]]),
    }
    result = forecast_error_sums(
        prediction,
        data,
        (torch.tensor([[1, 0]]), torch.tensor([[0, 1]])),
    )
    assert result["actor_count"].item() == 1
    assert result["current_vehicle_count"].item() == 2
    for name in ("min_ade_sum", "min_fde_sum"):
        assert result[name].item() == 1
    for name in ("mode_ade_sum", "mode_fde_sum"):
        assert result[name].item() == 2
