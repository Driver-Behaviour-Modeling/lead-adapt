"""The model tokenizes exactly the deltas the codebook was clustered from."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
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


def _body_frame_decoder(
    tmp_path,
    decoding: str,
    beam_width: int = 8,
    history_free: bool = False,
    rolling: bool = False,
):
    import pickle

    from lead.adapt.adapt_decoder import AdaptDecoder

    sys.path.insert(0, str(Path(__file__).parent))
    from test_navigation_conditioning import SmallConfig

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
    config.adapt_decoding = decoding
    config.adapt_beam_width = beam_width
    config.adapt_history_free = history_free
    config.adapt_rolling_tokenization = rolling
    torch.manual_seed(3)
    return AdaptDecoder(8, config, torch.device("cpu")).eval()


def _sequence_log_prob(model, prefix, context, future):
    """Joint log-probability of future tokens under teacher forcing."""
    from lead.adapt.adapt_decoder import generate_causal_mask

    tokens = torch.cat([prefix, future], dim=1)
    out = model.transformer_decoder(
        tgt_token_ids=tokens[:, :-1],
        encoder_output=context,
        self_attention_mask=generate_causal_mask(tokens.shape[1] - 1, "cpu"),
    )
    log_probs = torch.log_softmax(model.output_projection(out).float(), dim=-1)
    steps = log_probs[:, prefix.shape[1] - 1 :]
    return steps.gather(-1, future.unsqueeze(-1)).squeeze(-1).sum(-1)


def test_beam_width_one_is_greedy(tmp_path):
    sys.path.insert(0, str(Path(__file__).parent))
    from test_navigation_conditioning import sample

    greedy = _body_frame_decoder(tmp_path, "greedy")
    beam = _body_frame_decoder(tmp_path, "beam", beam_width=1)
    bev, data = torch.randn(2, 8, 2, 3), sample()
    with torch.inference_mode():
        a = greedy(bev, None, None, data, {})
        b = beam(bev, None, None, data, {})
    torch.testing.assert_close(
        a["future_token_ids"],
        b["future_token_ids"],
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(a["trajectory"], b["trajectory"])


def test_exhaustive_beam_finds_the_most_likely_sequence(tmp_path):
    import itertools

    model = _body_frame_decoder(tmp_path, "beam", beam_width=8**3)  # V^(T-1): exact
    prefix = torch.tensor([[1, 2], [5, 0]])
    context = torch.randn(2, 6, 16)
    with torch.inference_mode():
        best, _ = model._beam_search(prefix, context)
        every = torch.tensor(list(itertools.product(range(8), repeat=4)))
        for row in range(2):
            scores = _sequence_log_prob(
                model,
                prefix[row : row + 1].expand(len(every), -1),
                context[row : row + 1].expand(len(every), -1, -1),
                every,
            )
            torch.testing.assert_close(
                best[row],
                every[scores.argmax()],
                atol=0,
                rtol=0,
            )


def test_rolling_tokenization_recovers_reachable_codes_and_corrects_offsets():
    from lead.adapt.adapt_decoder import AdaptDecoder

    model = SimpleNamespace(kdisks_model=KDisksModel(CONFIG), rolling_beam_width=32)
    centroids = model.kdisks_model.codebook.centroids
    origin = torch.zeros(1, 3)
    # A trajectory made exactly of codebook steps is tokenized back to its codes.
    codes = torch.tensor([[40, 41, 42, 43, 44, 45, 46, 47]])
    path = integrate_body_frame_deltas(centroids[codes], origin)
    tokens, _ = AdaptDecoder._rolling_tokens(model, origin, path)
    torch.testing.assert_close(tokens, codes, atol=0, rtol=0)
    # A straight 8 m/s path seen from a start shifted 0.4 m sideways (as under
    # sensor perturbation): per-step tokenization keeps the offset for the whole
    # horizon, sequential tokenization steers back towards the path.
    straight = torch.stack(
        [torch.tensor([2.0 * (k + 1), 0.0, 0.0]) for k in range(8)],
    ).unsqueeze(0)
    start = torch.tensor([[0.0, 0.4, 0.0]])
    rolled, _ = AdaptDecoder._rolling_tokens(model, start, straight)
    rolled_end = integrate_body_frame_deltas(centroids[rolled], start)[0, -1]
    independent = model.kdisks_model.encode(
        model.kdisks_model._compute_deltas(
            torch.cat([start[:, None], straight], dim=1),
        ),
    )
    independent_end = integrate_body_frame_deltas(centroids[independent], start)[0, -1]
    assert abs(rolled_end[1]) < 0.1 < abs(independent_end[1])


@pytest.mark.parametrize("decoding", ["greedy", "beam", "expected"])
def test_history_free_decoder_needs_no_history(tmp_path, decoding):
    sys.path.insert(0, str(Path(__file__).parent))
    from test_navigation_conditioning import sample

    model = _body_frame_decoder(tmp_path, decoding, history_free=True, rolling=True)
    assert not hasattr(model, "history_pose_proj")
    data = sample()
    del data["past_positions"], data["past_yaws"]
    bev = torch.randn(2, 8, 2, 3)
    with torch.inference_mode():
        out = model(bev, None, None, data, {})
    assert out["output_logits"].shape == (2, 4, 8)
    assert out["future_token_ids"].max() < 8  # the start token is never emitted
    start = torch.zeros(2, 3)
    if decoding != "expected":
        expected = integrate_body_frame_deltas(
            model.kdisks_model.codebook.centroids[out["future_token_ids"]],
            start,
        )
        torch.testing.assert_close(out["trajectory"], expected)
    model.train()
    out = model(bev, None, None, data, {})
    losses = {}
    model.compute_loss(data, out, losses, {})
    losses["loss_kinematic_token"].backward()
    assert torch.isfinite(losses["loss_kinematic_token"])
