"""Inject predicted actor futures into the existing ego planning context."""

import torch
from torch import nn


class FutureContextFusion(nn.Module):
    """Confidence-weighted residual attention with an exact identity warm start.

    The always-valid null token prevents empty/low-confidence scenes from
    normalizing attention over spurious vehicles. No ground-truth mask is used.
    """

    def __init__(self, dimension: int, heads: int):
        super().__init__()
        self.heads = heads
        self.query_norm = nn.LayerNorm(dimension)
        self.memory_norm = nn.LayerNorm(dimension)
        self.attention = nn.MultiheadAttention(dimension, heads, batch_first=True)
        nn.init.zeros_(self.attention.out_proj.weight)
        nn.init.zeros_(self.attention.out_proj.bias)

    def forward(self, context, memory, confidence):
        if memory.ndim != 3 or confidence.shape != memory.shape[:2]:
            raise ValueError("World memory must be [B,N,D] with confidence [B,N]")
        if memory.shape[0] != context.shape[0] or memory.shape[2] != context.shape[2]:
            raise ValueError("World memory and ego context must share batch and width")
        memory = self.memory_norm(memory)
        memory = torch.cat(
            [memory, memory.new_zeros(memory.shape[0], 1, memory.shape[2])],
            dim=1,
        )
        confidence = torch.cat(
            [confidence.float(), confidence.new_ones(confidence.shape[0], 1).float()],
            dim=1,
        )
        bias = confidence.clamp(min=1e-8, max=1).log().to(context.dtype)
        bias = bias[:, None, None, :].expand(-1, self.heads, context.shape[1], -1)
        bias = bias.reshape(context.shape[0] * self.heads, context.shape[1], -1)
        update, _ = self.attention(
            self.query_norm(context),
            memory,
            memory,
            attn_mask=bias,
            need_weights=False,
        )
        return context + update
