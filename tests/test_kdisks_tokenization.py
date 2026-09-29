"""The model tokenizes exactly the deltas the codebook was clustered from."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from lead.adapt.adapt_decoder import integrate_body_frame_deltas
from lead.kdisks import KDisksModel

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from extract_lead_carla_deltas import compute_deltas  # noqa: E402

CONFIG = SimpleNamespace(
    kdisks_vocab_path="lead/adapt/codebooks/kdisks_carla_body.pkl",
    kdisks_heading_weight=2.6719030517676834,
    normalize_kinematics=False,
)


def turning_poses(count: int = 14) -> np.ndarray:
    """A left turn through +-pi in a global frame, sampled at 4 Hz."""
    t = np.arange(count) * 0.25
    heading = 2.9 + 0.6 * t
    position = 8 * np.column_stack((np.sin(heading), -np.cos(heading))) + [120, -40]
    return np.column_stack((position, heading))


def to_frame(poses: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """Express global poses in the ego frame of ``origin``."""
    c, s = np.cos(origin[2]), np.sin(origin[2])
    d = poses[:, :2] - origin[:2]
    heading = np.arctan2(
        np.sin(poses[:, 2] - origin[2]),
        np.cos(poses[:, 2] - origin[2]),
    )
    return np.column_stack(
        (c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1], heading),
    )


def test_model_deltas_match_codebook_extraction():
    poses = turning_poses()
    expected = compute_deltas(poses)
    model = KDisksModel(CONFIG)
    # The dataloader hands the model poses in the current ego frame.
    local = torch.from_numpy(to_frame(poses, poses[5])).float()[None]
    np.testing.assert_allclose(
        model._compute_deltas(local)[0].numpy(),
        expected,
        atol=1e-5,
    )


def test_integration_inverts_deltas():
    local = torch.from_numpy(to_frame(turning_poses(), turning_poses()[5])).float()[
        None
    ]
    model = KDisksModel(CONFIG)
    deltas = model._compute_deltas(local)
    rebuilt = integrate_body_frame_deltas(deltas, local[:, 0])
    np.testing.assert_allclose(rebuilt.numpy(), local[:, 1:].numpy(), atol=1e-4)


def test_waypoints_from_tokens_integrate_the_argmax_rollout(tmp_path):
    import pickle

    from lead.adapt.adapt_decoder import AdaptDecoder

    sys.path.insert(0, str(Path(__file__).parent))
    from test_navigation_conditioning import SmallConfig, sample

    vocabulary = tmp_path / "vocabulary.pkl"
    centroids = np.zeros((8, 3), dtype=np.float32)
    centroids[:, 0] = np.arange(8) * 0.5
    centroids[:, 2] = np.linspace(-0.1, 0.1, 8)
    info = {"frame": "body", "heading_weight": CONFIG.kdisks_heading_weight}
    with vocabulary.open("wb") as stream:
        pickle.dump({"centroids": centroids, "info": info}, stream)
    config = SmallConfig()
    config.kdisks_vocab_path = str(vocabulary)
    config.adapt_waypoints_from_tokens = True
    model = AdaptDecoder(8, config, torch.device("cpu")).eval()
    assert not any(p.requires_grad for p in model._trajectory_head.parameters())

    data = sample()
    with torch.inference_mode():
        out = model(torch.randn(2, 8, 2, 3), None, None, data, {})
    tokens = out["output_logits"].argmax(-1)
    start = torch.cat(
        [data["past_positions"][:, -1], data["past_yaws"][:, -1:]],
        dim=-1,
    )
    expected = integrate_body_frame_deltas(torch.from_numpy(centroids)[tokens], start)
    torch.testing.assert_close(out["trajectory"], expected)
