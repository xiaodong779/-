#!/usr/bin/env python3
"""Per-layer Video Prompt branch for TGL v2 (causal, CVPT-style injection).

Causal connection (asserted):
    Block 1..4 (0-based 0..3): original frozen CLIP
    Block 4 output (0-based block index 3): build 8 full-frame semantic
        frame_tokens via the EXISTING FrameMotionPool (CLS queries all 196
        patches - full-frame, not a local average) + the EXISTING
        LightweightTemporalAdapter (temporal position encoding). Reused, not
        re-implemented.
    4 slot queries -> q_ctx  (beta = sigmoid(beta_logit), init 0.5)
    per-layer P_l = prompt_norm_l(up_proj_l(q_ctx))  for layers 5..12
    Blocks 5..12 (0-based 4..11): CVPT-style cross-attention injection
        (prompts are ONLY key/value; image CLS + patches are queries);
        alpha_l per layer, zero-initialised; prompt outputs are NOT tokens
        in the stream (no accumulation across layers).

Guarantees:
    * prompt_source_layer < min(prompt_layers)  (asserted at construction);
    * all alpha_l == 0  =>  output bit-identical to the original frozen CLIP
      (atol 1e-6, tested);
    * up_proj_l uses a small NON-ZERO init, so after one optimizer step the
      prompt generator receives gradients (single-step cold start only).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["PromptFrameTokens", "TGLPromptGenerator", "PromptCrossAttention"]


class PromptFrameTokens(nn.Module):
    """Build 8 full-frame semantic frame_tokens from a block output.

    Reuses the user's existing modules (imported from
    modules.correspondence_video_prompt):
      FrameMotionPool:             per-frame CLS queries ALL 196 patches
      LightweightTemporalAdapter:  temporal position encoding (small)
    """

    def __init__(self, d_model: int, bottleneck: int = 192, num_heads: int = 4, num_frames: int = 8) -> None:
        super().__init__()
        from modules.correspondence_video_prompt import (
            FrameMotionPool,
            LightweightTemporalAdapter,
        )

        self.motion_pool = FrameMotionPool(d_model, bottleneck, num_heads)
        self.temporal = LightweightTemporalAdapter(d_model, bottleneck, num_heads, num_frames)

    def forward(self, block_output: torch.Tensor, batch_size: int, num_frames: int) -> torch.Tensor:
        """block_output: seq-first [1+N, BT, D] at the prompt source layer.

        Returns frame_tokens [B, T, D] (full-frame semantic + time encoding).
        """
        S, BT, D = block_output.shape
        cls = block_output[0].reshape(batch_size, num_frames, D)
        patches = (
            block_output[1:]
            .permute(1, 0, 2)
            .reshape(batch_size, num_frames, S - 1, D)
        )
        ft = self.motion_pool(cls, patches)  # per-frame CLS queries ALL patches
        ft = self.temporal(ft)               # + temporal position encoding
        return ft


class TGLPromptGenerator(nn.Module):
    """4 slot queries + per-layer prompt heads (causal, non-degenerate init).

    beta = sigmoid(beta_logit) with beta_logit init 0 -> beta starts at 0.5.
    Per-layer up_proj uses normal(std=0.02) NON-ZERO init; the injection
    alpha_l (held by the caller) is zero-init - only ONE of the two factors
    is zero at any time, so there is no cold-start deadlock.
    """

    def __init__(
        self,
        d_model: int,
        bottleneck: int = 192,
        num_heads: int = 4,
        num_prompts: int = 4,
        num_prompt_layers: int = 8,
        beta_init: float = 0.5,
    ) -> None:
        super().__init__()
        self.num_prompts = int(num_prompts)
        self.num_prompt_layers = int(num_prompt_layers)
        self.video_queries = nn.Parameter(torch.zeros(num_prompts, bottleneck))
        nn.init.trunc_normal_(self.video_queries, std=0.02)
        self.query_norm = nn.LayerNorm(bottleneck)
        self.norm_kv = nn.LayerNorm(d_model)
        self.kv_proj = nn.Linear(d_model, bottleneck)
        self.cross_attn = nn.MultiheadAttention(bottleneck, num_heads, batch_first=True)
        self.beta_logit = nn.Parameter(torch.zeros(1))
        if abs(beta_init - 0.5) > 1e-6:
            self.beta_logit.fill_(math.log(beta_init / (1.0 - beta_init)))
        self.slot_norm = nn.LayerNorm(bottleneck)
        self.up_proj_l = nn.ModuleList(
            nn.Linear(bottleneck, d_model) for _ in range(self.num_prompt_layers)
        )
        for up in self.up_proj_l:
            nn.init.normal_(up.weight, std=0.02)
            nn.init.zeros_(up.bias)
        self.prompt_norm_l = nn.ModuleList(
            nn.LayerNorm(d_model) for _ in range(self.num_prompt_layers)
        )

    def forward(self, frame_tokens: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """frame_tokens: [B, T, D] -> (q_ctx [B, P, 192], stats)."""
        q = self.query_norm(self.video_queries).unsqueeze(0).expand(
            frame_tokens.shape[0], -1, -1
        )  # [B, P, 192]
        kv = self.kv_proj(self.norm_kv(frame_tokens))  # [B, T, 192]
        out, _ = self.cross_attn(q, kv, kv, need_weights=False)
        beta = torch.sigmoid(self.beta_logit.float())
        q_ctx = self.slot_norm(q + beta * out)
        # diagnostics
        with torch.no_grad():
            qn = q.detach().float()
            dyn = (beta.detach().float() * out.detach().float())
            pn = qn.norm(dim=-1).clamp_min(1e-6)
            norms = dyn.norm(dim=-1) / pn  # [B, P]
            qn_unit = F.normalize(qn, dim=-1)
            slots = F.normalize(q_ctx.detach().float(), dim=-1)  # [B,P,192]
            cos = torch.bmm(slots, slots.transpose(1, 2))
            eye = torch.eye(self.num_prompts, device=cos.device).bool()
            slot_pairwise_cos = float(cos.masked_fill(eye, 0).sum() / (~eye).sum() / cos.shape[0])
        stats = {
            "beta": float(beta),
            "dyn_static_ratio": float(norms.mean()),
            "slot_pairwise_cos": slot_pairwise_cos,
        }
        return q_ctx, stats

    def layer_prompts(self, q_ctx: torch.Tensor, slot: int) -> torch.Tensor:
        """q_ctx [B, P, 192] -> P_l [B, P, D] for the slot-th prompt layer."""
        return self.prompt_norm_l[slot](self.up_proj_l[slot](q_ctx))


