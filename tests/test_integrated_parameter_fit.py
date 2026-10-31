"""Check physical smoothing likelihood, temperature recovery, and fit isolation."""

import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from lead.scoring.distributions import MotionSmoother
from lead.scoring.parameters import ObservedLikelihood, fit_parameters
from scripts.fit_integrated_reference import role_indices
from scripts.partition_integrated_drives import partition

ROOT = Path(__file__).resolve().parents[1]


def test_observed_likelihood_matches_full_column_normalized_smoothing():
    codes = torch.tensor([[0, 0, 0], [1, 0, 3.13], [1, 0, -3.13], [3, 1, 0]])
    logits = torch.randn(2, 7, 4, generator=torch.Generator().manual_seed(91))
    tokens = torch.tensor([0, 1, 2, 3, 3, 2, 0])
    likelihood = ObservedLikelihood(logits, tokens, codes, 2.67)
    temperatures = [1.3, 0.8]
    observed = likelihood.evaluate(temperatures, 0.1, 0.5)
    full = MotionSmoother(codes, 2.67, 0.1, 0.5)(
        logits,
        torch.tensor(temperatures, dtype=torch.float64)[:, None, None],
    )
    expected = full.gather(-1, tokens[None, :, None].expand(2, -1, -1)).squeeze(-1)
    torch.testing.assert_close(observed, expected)


def test_temperature_fit_recovers_known_binary_frequency():
    logits = torch.tensor([[[math.log(9.0), 0]]] * 100).transpose(0, 1)
    tokens = torch.tensor([0] * 80 + [1] * 20)
    result = fit_parameters(
        logits,
        tokens,
        torch.tensor([[0, 0, 0], [1, 0, 0]]),
        2.67,
        epsilons=(0.0,),
        sigmas=(0.1,),
        rounds=1,
    )
    assert result["temperatures"][0] == pytest.approx(
        math.log(9) / math.log(4),
        rel=1e-3,
    )
    assert result["fitted_nll"] == pytest.approx(
        -0.8 * math.log(0.8) - 0.2 * math.log(0.2),
        rel=1e-6,
    )
    assert result["fitted_nll"] < result["initial_nll"]


def test_partitions_reject_repeated_route_in_different_roles():
    roles = {
        role: [role]
        for role in ("validation", "calibration", "drive_reference", "test")
    }
    manifest = {
        "roles": roles,
        "inventory": {
            "validation": {"route_group": "one"},
            "test": {"route_group": "one"},
        },
    }
    with pytest.raises(ValueError, match="Repeated physical route"):
        role_indices(np.array(list(roles)), manifest)


def test_scenario_balancing_and_repeated_recordings_stay_together():
    groups = {
        f"{scenario}-{i}": [{"scenario": scenario}] * (2 if i == 0 else 1)
        for scenario in ("a", "b", "c")
        for i in range(8)
    }
    assignment = partition(groups, 53)
    assert assignment == partition(groups, 53)
    for scenario in ("a", "b", "c"):
        assert (
            sorted(
                list(assignment.values()).count(role)
                for role in set(assignment.values())
            )
            == [6] * 4
        )
        assert {assignment[f"{scenario}-{i}"] for i in range(8)} == {
            "validation",
            "calibration",
            "drive_reference",
            "test",
        }


def test_cli_test_labels_cannot_change_validation_fit(tmp_path):
    codes = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
    roles = {
        role: [str(tmp_path / role)]
        for role in ("validation", "calibration", "drive_reference", "test")
    }
    manifest = tmp_path / "partitions.json"
    manifest.write_text(json.dumps({"roles": roles}))
    routes = np.repeat([paths[0] for paths in roles.values()], 20)
    tokens = np.tile(np.array([0] * 16 + [1] * 4, dtype=np.int64), 4)
    logits = np.tile([math.log(9), 0.0], (1, 80, 1))
    fitted = []
    for trial in range(2):
        prefix = tmp_path / f"capture{trial}"
        changed = tokens.copy()
        changed[60:] = trial
        np.savez_compressed(
            prefix.with_suffix(".npz"),
            vocabulary_centroids=codes,
            member_logits=logits,
            token=changed,
            route=routes,
            frame=np.tile(np.arange(20), 4),
            quantization_residual=np.zeros(80),
            world_conflict_count=np.zeros(80),
        )
        prefix.with_suffix(".json").write_text(
            json.dumps(
                {
                    "members": [{"sha256": "fixture", "seed": 0}],
                    "vocabulary_sha256": hashlib.sha256(codes.tobytes()).hexdigest(),
                    "heading_weight": 2.67,
                },
            ),
        )
        output = tmp_path / f"fit{trial}"
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/fit_integrated_reference.py"),
                "--capture",
                str(prefix),
                "--partitions",
                str(manifest),
                "--output-dir",
                str(output),
                "--epsilons",
                "0",
                "--sigmas",
                "0.1",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        fitted.append(json.loads((output / "parameters.json").read_text()))
        saved = json.loads((output / "test.json").read_text())
        assert saved["parameter_fit_drives"] == roles["validation"]
        assert saved["role"] == "test" and saved["frames"] == 20
    assert fitted[0]["temperatures"] == fitted[1]["temperatures"]
    assert fitted[0]["fitted_nll"] == fitted[1]["fitted_nll"]
