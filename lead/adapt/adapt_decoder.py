import logging
import math
from typing import TypedDict

import jaxtyping as jt
import numpy as np
import torch
import torch.nn.functional as F
from beartype import beartype
from torch import nn

import lead.common.common_utils as common_utils
from lead.common.constants import RadarLabels
from lead.adapt import transfuser_utils as fn
from lead.kdisks import KDisksModel
from lead.training.config_training import TrainingConfig

logger = logging.getLogger(__name__)


def generate_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Build an upper-triangular causal mask compatible with ``nn.MultiheadAttention``.

    Args:
        seq_len: Length of the target sequence.
        device: Device the mask is materialised on.

    Returns:
        A ``[seq_len, seq_len]`` boolean mask where ``True`` marks positions
        that must not be attended to.
    """
    return torch.triu(
        torch.ones(seq_len, seq_len, device=device), diagonal=1,
    ).bool()


class AdaptDecoderOutput(TypedDict):
    """Outputs of :meth:`AdaptDecoder.forward`.

    The auxiliary tensors (logits, token IDs, deltas, centroids, codebook
    losses) are passed alongside the user-facing waypoint/heading/trajectory
    fields so that :meth:`AdaptDecoder.compute_loss` can compute token CE,
    soft CE and commitment terms without leaking them into the top-level
    ``Prediction`` dataclass.
    """

    pred_future_waypoints: jt.Float[torch.Tensor, "B n_waypoints 2"]
    pred_headings: jt.Float[torch.Tensor, "B n_waypoints"]
    trajectory: jt.Float[torch.Tensor, "B n_waypoints 3"]
    output_logits: jt.Float[torch.Tensor, "B n_waypoints V"]
    future_token_ids: jt.Int[torch.Tensor, "B n_waypoints"]
    future_deltas: jt.Float[torch.Tensor, "B n_waypoints 3"] | None
    centroids: jt.Float[torch.Tensor, "K 3"] | None
    kinematic_tokens: jt.Int[torch.Tensor, "B T_hist_minus_1"]
    commitment_loss: jt.Float[torch.Tensor, ""] | None
    dictionary_loss: jt.Float[torch.Tensor, ""] | None


def wrap_angle_torch(angle: torch.Tensor) -> torch.Tensor:
    """Wrap angle to [-π, π] range."""
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def compute_kdisks_distances(
    gt_deltas: torch.Tensor,
    centroids: torch.Tensor,
    heading_weight: float = 1.0,
) -> torch.Tensor:
    """Compute K-disks distances from GT deltas to all centroids.

    Position L2 + weighted (wrapped) heading L1 — same metric used by
    the K-disks codebook for clustering and lookup.

    Args:
        gt_deltas: Ground-truth deltas (Δx, Δy, Δheading).
        centroids: Codebook centroids.
        heading_weight: Multiplier on the heading distance term.

    Returns:
        Per-pair distance tensor.
    """
    pos_diff = gt_deltas[:, :2].unsqueeze(1) - centroids[:, :2].unsqueeze(0)
    pos_dist = torch.norm(pos_diff, dim=-1)

    heading_diff = gt_deltas[:, 2:3] - centroids[:, 2:3].T
    heading_diff = wrap_angle_torch(heading_diff)
    heading_dist = torch.abs(heading_diff) * heading_weight

    return pos_dist + heading_dist


def soft_cross_entropy_loss(
    predicted_logits: torch.Tensor,
    gt_deltas: torch.Tensor,
    centroids: torch.Tensor,
    sigma: float = 0.1,
    heading_weight: float = 1.0,
    min_prob: float = 1e-6,
) -> torch.Tensor:
    """Soft cross-entropy against Gaussian-weighted soft targets in delta space.

    Tokens that represent motions close to the GT delta receive partial credit,
    rather than hard one-hot supervision penalising near-correct neighbours equally.

    Args:
        predicted_logits: Per-step logits over the codebook.
        gt_deltas: Ground-truth (Δx, Δy, Δheading) per step.
        centroids: Codebook centroids in delta space.
        sigma: Gaussian std in delta space — smaller = sharper targets.
        heading_weight: Weight on heading in the K-disks distance.
        min_prob: Floor for numerical stability.

    Returns:
        Mean soft cross-entropy.
    """
    distances = compute_kdisks_distances(gt_deltas, centroids, heading_weight)
    neg_sq_distances = -distances.pow(2) / (2 * sigma**2)
    soft_targets = F.softmax(neg_sq_distances, dim=-1)
    soft_targets = torch.clamp(soft_targets, min=min_prob)
    soft_targets = soft_targets / soft_targets.sum(dim=-1, keepdim=True)
    log_probs = F.log_softmax(predicted_logits, dim=-1)
    return -(soft_targets * log_probs).sum(dim=-1).mean()


def _rotated_l1_loss(
    pred_traj: torch.Tensor,
    gt_traj: torch.Tensor,
    lat_weight: float = 5.0,
) -> torch.Tensor:
    """Rotated L1 loss decomposed into longitudinal/lateral/heading components.

    Args:
        pred_traj: Predicted trajectory (B, T, 3) — (x, y, heading).
        gt_traj: Ground-truth trajectory (B, T, 3).
        lat_weight: Multiplier on the lateral error term.

    Returns:
        Scalar mean loss.
    """
    pred_xy = pred_traj[..., :2]
    gt_xy = gt_traj[..., :2]
    gt_heading = gt_traj[..., 2]

    diff = pred_xy - gt_xy
    c = torch.cos(gt_heading)
    s = torch.sin(gt_heading)

    long_error = diff[..., 0] * c + diff[..., 1] * s
    lat_error = -diff[..., 0] * s + diff[..., 1] * c
    heading_error = pred_traj[..., 2] - gt_traj[..., 2]

    return (
        torch.abs(long_error)
        + lat_weight * torch.abs(lat_error)
        + torch.abs(heading_error)
    ).mean()


def _soft_frechet_loss(
    pred_traj: torch.Tensor,
    gt_traj: torch.Tensor,
    gamma: float = 0.1,
) -> torch.Tensor:
    """Soft discrete Fréchet distance over 2D trajectories.

    Args:
        pred_traj: Predicted xy coordinates (B, T, 2).
        gt_traj: Ground-truth xy coordinates (B, T, 2).
        gamma: Smoothing parameter — lower is closer to the hard min/max.

    Returns:
        Mean soft Fréchet distance over the batch.
    """
    pred_traj = pred_traj.float()
    gt_traj = gt_traj.float()
    B, T, _ = pred_traj.shape
    device = pred_traj.device

    dist_mat = torch.cdist(pred_traj, gt_traj, p=2)
    ca = torch.full((B, T, T), fill_value=1e6, device=device)

    def soft_max(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return gamma * torch.logsumexp(torch.stack([a, b], dim=-1) / gamma, dim=-1)

    def soft_min_three(
        a: torch.Tensor, b: torch.Tensor, c: torch.Tensor,
    ) -> torch.Tensor:
        return -gamma * torch.logsumexp(
            torch.stack([-a, -b, -c], dim=-1) / gamma, dim=-1,
        )

    ca[:, 0, 0] = dist_mat[:, 0, 0]
    for i in range(1, T):
        ca[:, i, 0] = soft_max(ca[:, i - 1, 0], dist_mat[:, i, 0])
        ca[:, 0, i] = soft_max(ca[:, 0, i - 1], dist_mat[:, 0, i])

    for i in range(1, T):
        for j in range(1, T):
            prev_min = soft_min_three(
                ca[:, i - 1, j], ca[:, i - 1, j - 1], ca[:, i, j - 1],
            )
            ca[:, i, j] = soft_max(prev_min, dist_mat[:, i, j])

    return ca[:, T - 1, T - 1].mean()


class AdaptDecoder(nn.Module):
    @beartype
    def __init__(
        self,
        input_bev_channels: int,
        config: TrainingConfig,
        device: torch.device,
    ):
        super().__init__()
        self.device = device
        self.config = config
        self.adapt_context_encoder = PlanningContextEncoder(
            config=self.config,
            input_bev_channels=input_bev_channels,
            device=self.device,
        )

        # K-disks tokenizer: loads the frozen codebook from
        # ``config.kdisks_vocab_path`` and exposes ``_compute_deltas`` /
        # ``encode`` / ``decode`` for kinematic tokenization.
        self.kdisks_model = KDisksModel(self.config)
        self._num_history_poses = self.config.num_history_poses
        self._num_future_poses = self.config.num_way_points_prediction

        # Autoregressive token decoder. Operates at ``kinematic_embed_dim``;
        # cross-attends to ``context_tokens`` from the context encoder.
        self.transformer_decoder = Decoder(
            kinematic_vocab_size=config.kinematic_vocab_size,
            d_model=config.kinematic_embed_dim,
            ffn_hidden=config.decoder_ffn_dim,
            num_heads=config.decoder_num_heads,
            drop_prob=config.decoder_dropout,
            num_layers=config.decoder_num_layers,
            max_sequence_length=config.max_sequence_length,
        )

        # The context encoder emits tokens at ``transfuser_token_dim`` while
        # the decoder operates at ``kinematic_embed_dim``. ``MultiheadAttention``
        # requires query/key dim parity, so we project when they differ.
        if self.config.kinematic_embed_dim != self.config.transfuser_token_dim:
            self.context_proj = nn.Linear(
                self.config.transfuser_token_dim,
                self.config.kinematic_embed_dim,
            )
        else:
            self.context_proj = nn.Identity()

        # Fold history poses (x, y, heading) into the cross-attention context.
        # Mirrors ADAPT's ``_history_pose_proj`` so the decoder sees both BEV
        # tokens and continuous past poses.
        self.history_pose_proj = nn.Linear(3, self.config.kinematic_embed_dim)

        # Learned positional embedding over the concatenated encoder context
        # (context_tokens from the encoder + projected history poses). The
        # exact BEV-token count is only known at forward time, so we size the
        # embedding to a generous upper bound; unused rows are inert.
        max_context_len = (
            self.adapt_context_encoder.num_status_tokens
            + 1024  # upper bound on flattened BEV grid + scratch
            + self._num_history_poses
        )
        self.encoder_context_pos_embedding = nn.Embedding(
            max_context_len,
            self.config.kinematic_embed_dim,
        )

        # Output projection: decoder hidden state → logits over codebook.
        self.output_projection = nn.Linear(
            config.kinematic_embed_dim, config.kinematic_vocab_size,
        )

        # Trajectory refinement head: concat decoder hidden states → (x, y, h)
        # trajectory. Concatenation preserves temporal ordering vs mean-pooling.
        self._trajectory_head = nn.Sequential(
            nn.Linear(
                config.kinematic_embed_dim * self._num_future_poses,
                config.decoder_ffn_dim,
            ),
            nn.ReLU(),
            nn.Linear(config.decoder_ffn_dim, self._num_future_poses * 3),
        )

        # Scheduled-sampling probability — bumped by the training loop if used.
        self._ss_prob = 0.0

    @beartype
    def forward(
        self,
        bev_features: jt.Float[torch.Tensor, "bs bev_dim height_bev width_bev"],
        data: dict,
        log: dict,
    ) -> AdaptDecoderOutput:
        """Run the autoregressive ADAPT decoder over BEV-derived context.

        Mirrors the NavSim ADAPT forward path (history pose extraction →
        K-disks tokenization → encoder context build → AR decode → trajectory
        head) but consumes the LEAD-style ``data`` dict instead of NavSim's
        feature dict.

        Args:
            bev_features: Raw BEV feature grid from the backbone.
            data: Per-batch data dict. Required keys: ``past_positions`` and
                ``past_yaws`` for history. During training, ``future_waypoints``
                and ``future_yaws`` are used to build teacher-forcing targets.
            log: Mutable log dict for metrics.

        Returns:
            Dict with ``pred_future_waypoints``, ``pred_headings``,
            ``trajectory``, ``output_logits``, ``future_token_ids``,
            ``future_deltas``, ``centroids``, ``kinematic_tokens``,
            ``commitment_loss`` and ``dictionary_loss``.
        """
        device = bev_features.device
        bs = bev_features.shape[0]

        # ----------------------------------------------------------------------
        # 1. BEV/status context tokens from the existing context encoder
        # ----------------------------------------------------------------------
        context_tokens = self.adapt_context_encoder(
            bev_features=bev_features,
            radar_logits=None,
            radar_predictions=None,
            data=data,
            log=log,
        )
        context_tokens = self.context_proj(context_tokens)

        # ----------------------------------------------------------------------
        # 2. History pose extraction and K-disks tokenization
        # ----------------------------------------------------------------------
        # Keep all K-disks math in float32 — the codebook centroids are
        # registered as float32 buffers and autocast can otherwise downcast
        # ``history_deltas`` to bf16/fp16, leaving ``argmin`` on a mixed-dtype
        # distance matrix.
        past_positions = data["past_positions"].to(
            device=device, non_blocking=True,
        )[:, : self._num_history_poses, :2].float()
        past_yaws = data["past_yaws"].to(
            device=device, non_blocking=True,
        )[:, : self._num_history_poses].float()
        history_poses = torch.cat(
            [past_positions, past_yaws.unsqueeze(-1)], dim=-1,
        )  # [B, T_hist, 3]
        T_hist = history_poses.shape[1]

        history_deltas = self.kdisks_model._compute_deltas(history_poses)
        kinematic_tokens = self.kdisks_model.encode(history_deltas).view(
            bs, T_hist - 1,
        )

        # ----------------------------------------------------------------------
        # 3. Future tokenization for teacher forcing (training only)
        # ----------------------------------------------------------------------
        future_token_ids: torch.Tensor | None = None
        future_deltas: torch.Tensor | None = None
        centroids: torch.Tensor | None = None
        commitment_loss = torch.tensor(0.0, device=device)
        dictionary_loss = torch.tensor(0.0, device=device)

        if self.training and "future_waypoints" in data:
            T_future_cfg = self._num_future_poses
            future_waypoints = data["future_waypoints"].to(
                device=device, non_blocking=True,
            )[:, :T_future_cfg, :2].float()
            future_yaws = data["future_yaws"].to(
                device=device, non_blocking=True,
            )[:, :T_future_cfg].float()
            target_trajectory = torch.cat(
                [future_waypoints, future_yaws.unsqueeze(-1)], dim=-1,
            )  # [B, T_future, 3]
            T_future = target_trajectory.shape[1]

            last_history_pose = history_poses[:, -1:, :]
            full_future_seq = torch.cat(
                [last_history_pose, target_trajectory], dim=1,
            )  # [B, T_future+1, 3]
            future_deltas = self.kdisks_model._compute_deltas(full_future_seq)
            future_token_ids = self.kdisks_model.encode(future_deltas).view(
                bs, T_future,
            )

            if self.config.use_kdisks:
                centroids = self.kdisks_model.codebook.centroids

        # ----------------------------------------------------------------------
        # 4. Build encoder context: BEV/status tokens + history pose embeddings
        # ----------------------------------------------------------------------
        history_pose_emb = self.history_pose_proj(history_poses)  # [B, T_hist, D]
        encoder_context = torch.cat([context_tokens, history_pose_emb], dim=1)
        context_len = encoder_context.shape[1]
        pos_ids = torch.arange(context_len, device=device).unsqueeze(0)
        encoder_context = encoder_context + self.encoder_context_pos_embedding(
            pos_ids,
        )

        # ----------------------------------------------------------------------
        # 5. Autoregressive decode
        # ----------------------------------------------------------------------
        if self.training and future_token_ids is not None:
            T_future = future_token_ids.shape[1]

            if self._ss_prob > 0.0:
                # ---- Scheduled sampling: step-by-step, coin-flip GT vs pred --
                generated_tokens = kinematic_tokens.clone()
                all_logits_list: list[torch.Tensor] = []
                all_hidden_list: list[torch.Tensor] = []

                for t in range(T_future):
                    seq_len = generated_tokens.shape[1]
                    causal_mask = generate_causal_mask(seq_len, device)

                    decoder_output = self.transformer_decoder(
                        tgt_token_ids=generated_tokens,
                        encoder_output=encoder_context,
                        self_attention_mask=causal_mask,
                    )

                    all_hidden_list.append(decoder_output[:, -1:, :])
                    last_logits = self.output_projection(
                        decoder_output[:, -1:, :],
                    )
                    all_logits_list.append(last_logits)

                    if t < T_future - 1:
                        predicted_token = last_logits.argmax(dim=-1)
                        gt_token = future_token_ids[:, t : t + 1]
                        use_pred = (
                            torch.rand(bs, 1, device=device) < self._ss_prob
                        ).long()
                        next_token = (
                            use_pred * predicted_token + (1 - use_pred) * gt_token
                        )
                    else:
                        next_token = last_logits.argmax(dim=-1)

                    generated_tokens = torch.cat(
                        [generated_tokens, next_token], dim=1,
                    )

                output_logits = torch.cat(all_logits_list, dim=1)
                future_hidden_states = torch.cat(all_hidden_list, dim=1)
            else:
                # ---- Pure teacher forcing (parallel) -------------------------
                decoder_input_tokens = torch.cat(
                    [kinematic_tokens, future_token_ids[:, :-1]], dim=1,
                )
                seq_len = decoder_input_tokens.shape[1]
                causal_mask = generate_causal_mask(seq_len, device)

                decoder_output = self.transformer_decoder(
                    tgt_token_ids=decoder_input_tokens,
                    encoder_output=encoder_context,
                    self_attention_mask=causal_mask,
                )
                all_logits = self.output_projection(decoder_output)
                num_history_tokens = kinematic_tokens.shape[1]
                output_logits = all_logits[:, num_history_tokens - 1 :, :]
                future_hidden_states = decoder_output[
                    :, num_history_tokens - 1 :, :
                ]
        else:
            # ---- AR inference (validation / test) ---------------------------
            T_future = self._num_future_poses
            generated_tokens = kinematic_tokens.clone()
            all_logits_list = []
            all_hidden_list = []

            for _ in range(T_future):
                seq_len = generated_tokens.shape[1]
                causal_mask = generate_causal_mask(seq_len, device)

                decoder_output = self.transformer_decoder(
                    tgt_token_ids=generated_tokens,
                    encoder_output=encoder_context,
                    self_attention_mask=causal_mask,
                )
                all_hidden_list.append(decoder_output[:, -1:, :])
                last_logits = self.output_projection(decoder_output[:, -1:, :])
                all_logits_list.append(last_logits)

                next_token = last_logits.argmax(dim=-1)
                generated_tokens = torch.cat(
                    [generated_tokens, next_token], dim=1,
                )

            output_logits = torch.cat(all_logits_list, dim=1)
            future_hidden_states = torch.cat(all_hidden_list, dim=1)

        # ----------------------------------------------------------------------
        # 6. Trajectory head — concat hidden states → (x, y, heading)
        # ----------------------------------------------------------------------
        concat_hidden = future_hidden_states.reshape(bs, -1)
        trajectory_flat = self._trajectory_head(concat_hidden)
        trajectory_raw = trajectory_flat.reshape(bs, self._num_future_poses, 3)
        heading = trajectory_raw[..., 2:3].tanh() * np.pi
        trajectory = torch.cat([trajectory_raw[..., :2], heading], dim=-1)

        # If we are not training (no GT future), populate ``future_token_ids``
        # with the AR samples so the dataclass contract is honoured. These are
        # not used for any loss in that mode.
        if future_token_ids is None:
            future_token_ids = output_logits.argmax(dim=-1)

        return AdaptDecoderOutput(
            pred_future_waypoints=trajectory[..., :2],
            pred_headings=trajectory[..., 2],
            trajectory=trajectory,
            output_logits=output_logits,
            future_token_ids=future_token_ids,
            future_deltas=future_deltas,
            centroids=centroids,
            kinematic_tokens=kinematic_tokens,
            commitment_loss=commitment_loss,
            dictionary_loss=dictionary_loss,
        )

    @beartype
    def compute_loss(
        self,
        data: dict,
        decoder_outputs: AdaptDecoderOutput,
        loss: dict,
        log: dict,
    ) -> None:
        """Populate ``loss`` (and metrics in ``log``) for the ADAPT decoder.

        Reads the dict returned by :meth:`forward` rather than the top-level
        ``Prediction`` dataclass, so the auxiliary tensors needed for token CE
        do not need to leak into the dataclass schema.

        Args:
            data: Per-batch data dict, must include ``future_waypoints`` and
                ``future_yaws`` aligned to ``num_way_points_prediction``.
            decoder_outputs: Dict returned by :meth:`forward`.
            loss: Mutable loss dict — each term added under its own key.
            log: Mutable log dict for metrics.
        """
        with torch.amp.autocast(device_type="cuda", enabled=False):
            num_steps = self.config.num_way_points_prediction

            waypoints_label = data["future_waypoints"].to(
                self.device,
                dtype=self.config.torch_float_type,
                non_blocking=True,
            )[:, :num_steps].float()
            heading_label = data["future_yaws"].to(
                self.device,
                dtype=self.config.torch_float_type,
                non_blocking=True,
            )[:, :num_steps].float()
            gt_traj = torch.cat(
                [waypoints_label, heading_label.unsqueeze(-1)], dim=-1,
            )  # [B, T, 3]

            pred_traj = decoder_outputs["trajectory"][:, :num_steps].float()

            loss["loss_trajectory"] = _rotated_l1_loss(
                pred_traj, gt_traj, lat_weight=5.0,
            )
            loss["loss_soft_frechet"] = _soft_frechet_loss(
                pred_traj[..., :2],
                gt_traj[..., :2],
                gamma=self.config.frechet_gamma,
            )

            output_logits = decoder_outputs["output_logits"].float()
            future_token_ids = decoder_outputs["future_token_ids"]
            logits_flat = output_logits.reshape(-1, output_logits.shape[-1])
            target_flat = future_token_ids.reshape(-1)

            use_soft_ce = (
                getattr(self.config, "use_soft_ce", False)
                and getattr(self.config, "use_kdisks", False)
                and decoder_outputs.get("centroids") is not None
                and decoder_outputs.get("future_deltas") is not None
            )
            if use_soft_ce:
                future_deltas_flat = decoder_outputs["future_deltas"].reshape(-1, 3).float()
                loss["loss_kinematic_token"] = soft_cross_entropy_loss(
                    predicted_logits=logits_flat,
                    gt_deltas=future_deltas_flat,
                    centroids=decoder_outputs["centroids"].float(),
                    sigma=getattr(self.config, "soft_ce_sigma", 0.1),
                    heading_weight=getattr(self.config, "kdisks_heading_weight", 1.0),
                    min_prob=getattr(self.config, "soft_ce_min_prob", 1e-6),
                )
            else:
                loss["loss_kinematic_token"] = F.cross_entropy(
                    logits_flat, target_flat, reduction="mean",
                )

            commitment_loss = decoder_outputs.get("commitment_loss")
            if commitment_loss is not None:
                loss["loss_commitment"] = commitment_loss.float()
            dictionary_loss = decoder_outputs.get("dictionary_loss")
            if dictionary_loss is not None:
                loss["loss_dictionary"] = dictionary_loss.float()

        if (
            "iteration" in data
            and ((data["iteration"] + 1) % self.config.log_scalars_frequency) == 0
        ):
            pred_waypoints = decoder_outputs["pred_future_waypoints"]
            pred_headings = decoder_outputs["pred_headings"]
            log["metric/waypoints_ade"] = common_utils.average_displacement_error(
                pred_waypoints, waypoints_label,
            )
            log["metric/waypoints_fde"] = common_utils.final_displacement_error(
                pred_waypoints, waypoints_label,
            )
            log["metric/heading_ade"] = common_utils.average_displacement_error(
                pred_headings, heading_label,
            )

            with torch.no_grad():
                pred_token_ids = output_logits.argmax(dim=-1)
                token_accuracy = (pred_token_ids == future_token_ids).float().mean()
            log["metric/token_accuracy"] = token_accuracy.item()

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al., 2017).

    Ported verbatim from ADAPT's :mod:`temporal_transformer` so the AR decoder
    drop-in matches the NavSim implementation.
    """

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        self.d_model = d_model
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model),
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class FeedForward(nn.Module):
    """GELU MLP used inside the decoder layers (ported from ADAPT)."""

    def __init__(self, d_model: int, ffn_hidden: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, ffn_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class DecoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob, use_separable_conv=False):
        super().__init__()
        self.masked_attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.encoder_decoder_attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)
        
        # self.separable_conv = SeparableConvolution(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.ffn = FeedForward(d_model, ffn_hidden, drop_prob)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(self, x, encoder_output, self_attention_mask=None, cross_attention_mask=None):
        residual = x
        x = self.norm1(x)
        x, _ = self.masked_attention(x, x, x, attn_mask=self_attention_mask, need_weights=False)
        x = residual + self.dropout1(x)
        
        residual = x
        x = self.norm2(x)
        x, _ = self.encoder_decoder_attention(x, encoder_output, encoder_output, 
                               attn_mask=cross_attention_mask, need_weights=False)
        x = residual + self.dropout2(x)
        
        # residual = x.clone()
        # x = self.separable_conv(x)
        # x = self.dropout3(x)
        # x = self.norm3(x + residual)
        residual = x
        x = self.norm3(x)
        x = residual + self.dropout3(self.ffn(x))
        return x
    

class Decoder(nn.Module):
    def __init__(self,
                 kinematic_vocab_size,
                 d_model,
                 ffn_hidden,
                 num_heads,
                 drop_prob,
                 num_layers,
                 max_sequence_length,
                 use_separable_conv=False):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(kinematic_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_sequence_length)
        self.dropout = nn.Dropout(p=drop_prob)
        
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, ffn_hidden, num_heads, drop_prob, use_separable_conv) 
            for _ in range(num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, tgt_token_ids, encoder_output, self_attention_mask=None, cross_attention_mask=None):
        """
        Args:
            tgt_token_ids: [batch, tgt_seq_len]
            encoder_output: [batch, src_seq_len, d_model]
            self_attention_mask: Optional causal mask for decoder self-attention
            cross_attention_mask: Optional mask for encoder-decoder attention
        """
        # Embed and add positional encoding
        x = self.embedding(tgt_token_ids) * math.sqrt(self.d_model)  # Scale embeddings
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Apply decoder layers
        for layer in self.layers:
            x = layer(x, encoder_output, self_attention_mask, cross_attention_mask)
        
        x = self.final_norm(x)
        return x

@beartype
def decode_two_hot(
    two_hot_label: jt.Float[torch.Tensor, "B C"],
    class_values: list[float],
    device: torch.device,
) -> jt.Float[torch.Tensor, " B"]:
    """Decode a two-hot encoded tensor into a scalar representation.

    Args:
        two_hot_label: The two-hot encoded tensor. Must be between 0 and 1 and sum to 1 along the last dimension.
        class_values: List of class values (e.g., target_speeds or throttle_classes).
        device: Device to place tensors on.

    Returns:
        The decoded scalar tensor.
    """
    classes = torch.tensor(
        class_values,
        device=device,
        dtype=two_hot_label.dtype,
    ).unsqueeze(0)
    decoded = (two_hot_label * classes).sum(axis=-1)
    return decoded


@beartype
def encode_two_hot(
    scalar_values: jt.Float[torch.Tensor, " B"],
    class_values: list[float],
    brake: jt.Bool[torch.Tensor, " B"],
) -> jt.Float[torch.Tensor, "B C"]:
    """Encode scalar values into two-hot representation with linear interpolation.

    Args:
        scalar_values: Scalar values to encode (e.g., speeds or throttle values).
        class_values: List of class bin values (e.g., [0.0, 4.0, 8.0, ...] for speeds).
        brake: Optional boolean mask. If provided, positions where True will be encoded as class 0.

    Returns:
        Two-hot encoded distribution.
    """
    assert all(scalar_values >= 0.0)
    target_speeds = torch.tensor(
        class_values,
        dtype=scalar_values.dtype,
        device=scalar_values.device,
    )
    labels = torch.zeros(
        len(scalar_values),
        len(target_speeds),
        dtype=scalar_values.dtype,
        device=scalar_values.device,
    )
    labels[brake, 0] = 1.0
    non_brake = ~brake
    scalars = scalar_values[non_brake]
    last_bin = scalars >= target_speeds[-1]
    labels[non_brake & (scalar_values >= target_speeds[-1]), -1] = 1.0

    # Interpolation between bins
    interp_mask = ~last_bin
    if interp_mask.any():
        interp_speeds = scalars[interp_mask]
        upper_idx = torch.searchsorted(target_speeds, interp_speeds, right=False)
        lower_idx = upper_idx - 1

        lower_val = target_speeds[lower_idx]
        upper_val = target_speeds[upper_idx]

        lower_weight = (upper_val - interp_speeds) / (upper_val - lower_val)
        upper_weight = (interp_speeds - lower_val) / (upper_val - lower_val)

        row_idx = torch.where(non_brake)[0][interp_mask]
        labels[row_idx, lower_idx] = lower_weight
        labels[row_idx, upper_idx] = upper_weight

    return labels


class PlanningContextEncoder(nn.Module):
    @beartype
    def __init__(
        self,
        config: TrainingConfig,
        input_bev_channels: int,
        device: torch.device,
    ):
        super().__init__()
        self.device = device
        self.config: TrainingConfig = config

        self.num_status_tokens = 0

        if self.config.use_velocity:
            self.num_status_tokens += 1
            self.velocity_encoder = nn.Sequential(
                nn.Linear(1, self.config.transfuser_token_dim),
            )
            logger.info("Using velocity encoder.")

        if self.config.use_acceleration:
            self.num_status_tokens += 1
            self.acceleration_encoder = nn.Sequential(
                nn.Linear(1, self.config.transfuser_token_dim),
            )
            logger.info("Using acceleration encoder.")

        if self.config.use_discrete_command:
            self.num_status_tokens += 1
            self.command_encoder = nn.Sequential(
                nn.Linear(
                    self.config.discrete_command_dim,
                    self.config.transfuser_token_dim,
                ),
            )
            logger.info("Using discrete command encoder.")

        if self.config.use_tp:
            self.num_status_tokens += 1
            self.tp_encoder = nn.Linear(2, config.transfuser_token_dim)
            logger.info("Using target point encoder.")

        if self.config.use_previous_tp:
            self.num_status_tokens += 1
            logger.info("Using previous target point encoder.")

        if self.config.use_next_tp:
            self.num_status_tokens += 1
            logger.info("Using next target point encoder.")

        if self.config.use_past_positions:
            self.num_status_tokens += self.config.num_past_samples_used
            logger.info("Using past positions encoder.")
            self.past_positions_encoder = nn.Linear(2, config.transfuser_token_dim)

        if self.config.use_past_speeds:
            self.num_status_tokens += self.config.num_past_samples_used
            logger.info("Using past speeds encoder.")
            self.past_speeds_encoder = nn.Linear(1, config.transfuser_token_dim)

        if (
            self.config.use_radars
            and self.config.radar_detection
            and self.config.use_radar_detection
        ):
            self.num_status_tokens += self.config.num_radar_queries
            self.radar_encoder = nn.Linear(
                self.config.radar_token_dim,
                config.transfuser_token_dim,
            )
            logger.info(
                f"Using radar encoder with {self.config.num_radar_queries} tokens.",
            )

        self.cosine_pos_embeding = PositionEmbeddingSine(
            config,
            self.config.transfuser_token_dim // 2,
            normalize=True,
        )
        self.status_pos_embedding = nn.Parameter(
            torch.zeros(1, self.num_status_tokens, self.config.transfuser_token_dim),
        )

        self.dimension_adapter = nn.Conv2d(
            input_bev_channels,
            self.config.transfuser_token_dim,
            kernel_size=1,
        )
        self.reset_parameters()

        self.target_points_normalization_constants = torch.tensor(
            self.config.target_points_normalization_constants,
            device=self.device,
            dtype=self.config.torch_float_type,
        )

    def reset_parameters(self):
        nn.init.uniform_(self.status_pos_embedding)

    @beartype
    def forward(
        self,
        bev_features: jt.Float[torch.Tensor, "B C H W"],
        radar_logits: jt.Float[torch.Tensor, "B Q C"] | None,
        radar_predictions: jt.Float[torch.Tensor, "B Q 4"] | None,
        data: dict,
        log: dict,
    ) -> jt.Float[torch.Tensor, "B N D"]:
        """
        Args:
            bev_features: Raw BEV features.
            radar_logits: Radar logits.
            radar_predictions: Radar predictions.
            data: dict
            log: dict
        Returns:
            context_tokens: Output tokens for planning transformer decoder.
        """
        # Load data
        if self.config.use_velocity:
            velocity = (
                data["speed"]
                .reshape(-1, 1)
                .to(self.device, dtype=self.config.torch_float_type)
            )
        if self.config.use_discrete_command:
            command = data["command"].to(
                self.device,
                dtype=self.config.torch_float_type,
            )

        status_tokens = []

        # Encode speed
        if self.config.use_velocity:
            velocity_token = self.velocity_encoder(
                velocity / self.config.max_speed,
            ).reshape(
                -1,
                1,
                self.config.transfuser_token_dim,
            )  # (bs, 1, transfuser_token_dim)
            status_tokens.append(velocity_token)

        # Encode acceleration
        if self.config.use_acceleration:
            acceleration = (
                data["acceleration"]
                .reshape(-1, 1)
                .to(self.device, dtype=self.config.torch_float_type)
            )
            acceleration_token = self.acceleration_encoder(
                acceleration / self.config.max_acceleration,
            ).reshape(
                -1,
                1,
                self.config.transfuser_token_dim,
            )  # (bs, 1, transfuser_token_dim)
            status_tokens.append(acceleration_token)

        # Encode command
        if self.config.use_discrete_command:
            command_token = self.command_encoder(command).reshape(
                -1,
                1,
                self.config.transfuser_token_dim,
            )  # (bs, 1, transfuser_token_dim)
            status_tokens.append(command_token)

        # Encode target point
        if self.config.use_tp:
            target_point = data["target_point"].to(
                self.device,
                dtype=self.config.torch_float_type,
                non_blocking=True,
            )
            target_point = target_point / self.target_points_normalization_constants
            tp_token = self.tp_encoder(target_point).reshape(
                -1,
                1,
                self.config.transfuser_token_dim,
            )  # (bs, 1, transfuser_token_dim)
            status_tokens.append(tp_token)

        if self.config.use_previous_tp:
            previous_tp = data["target_point_previous"].to(
                self.device,
                dtype=self.config.torch_float_type,
                non_blocking=True,
            )
            previous_tp = previous_tp / self.target_points_normalization_constants
            previous_tp_token = self.tp_encoder(previous_tp).reshape(
                -1,
                1,
                self.config.transfuser_token_dim,
            )  # (bs, 1, transfuser_token_dim)
            status_tokens.append(previous_tp_token)

        if self.config.use_next_tp:
            next_tp = data["target_point_next"].to(
                self.device,
                dtype=self.config.torch_float_type,
                non_blocking=True,
            )
            next_tp = next_tp / self.target_points_normalization_constants
            next_tp_token = self.tp_encoder(next_tp).reshape(
                -1,
                1,
                self.config.transfuser_token_dim,
            )  # (bs, 1, transfuser_token_dim)
            status_tokens.append(next_tp_token)

        # Encode radar
        if (
            self.config.use_radars
            and self.config.radar_detection
            and self.config.use_radar_detection
        ):
            radar_token = self.radar_encoder(radar_logits).reshape(
                -1,
                self.config.num_radar_queries,
                self.config.transfuser_token_dim,
            )  # (bs, num_radar_queries, transfuser_token_dim)
            radar_pos_embed = fn.gen_sineembed_for_position(
                fn.unit_normalize_bev_points(
                    radar_predictions[..., [RadarLabels.X, RadarLabels.Y]].reshape(
                        -1,
                        2,
                    ),
                    self.config,
                ),
                self.config.transfuser_token_dim,
            ).reshape(
                radar_token.shape,
            )  # (bs, num_radar_queries, transfuser_token_dim)
            radar_token = (
                radar_token + radar_pos_embed
            )  # (bs, num_radar_queries, transfuser_token_dim)
            status_tokens.append(radar_token)

        # Concatenate status tokens if any
        has_statuses = False
        if len(status_tokens) > 0:
            status_tokens = torch.cat(
                status_tokens,
                dim=1,
            )  # (bs, num_status_tokens, transfuser_token_dim)
            has_statuses = True

        # Process BEV features
        context_tokens = self.dimension_adapter(
            bev_features,
        )  # (bs, transfuser_token_dim, height, width)

        # Concatenate and add positional embeddings
        if has_statuses:
            context_tokens = context_tokens + self.cosine_pos_embeding(
                context_tokens,
            )  # (bs, transfuser_token_dim, height, width)
            context_tokens = torch.flatten(
                context_tokens,
                start_dim=2,
            )  # (bs, transfuser_token_dim, height * width)
            context_tokens = torch.permute(
                context_tokens,
                (0, 2, 1),
            )  # (bs, height * width, transfuser_token_dim)

            status_tokens = (
                status_tokens + self.status_pos_embedding
            )  # (bs, num_status_tokens, transfuser_token_dim)
            context_tokens = torch.cat(
                [context_tokens, status_tokens],
                dim=1,
            )  # (bs, height * width + num_status_tokens, transfuser_token_dim)

        return context_tokens


class PositionEmbeddingSine(nn.Module):
    def __init__(
        self,
        config: TrainingConfig,
        num_pos_feats=64,
        temperature=10000,
        normalize=False,
        scale=None,
    ):
        super().__init__()
        self.config = config
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor: torch.Tensor):
        x = tensor
        bs, _, h, w = x.shape
        not_mask = torch.ones((bs, h, w), device=x.device)
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (
            2 * (torch.div(dim_t, 2, rounding_mode="floor")) / self.num_pos_feats
        )

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()),
            dim=4,
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()),
            dim=4,
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos.to(self.config.torch_float_type).contiguous()
