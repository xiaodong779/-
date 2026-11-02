#!/usr/bin/env python3
"""Decoupled layered video adaptation for frozen-CLIP video tuning.

Two independent, individually-toggleable channels (motivated by:

* VPT (Jia et al., ECCV 2022): prompts prepended at EVERY transformer
  layer's input with INDEPENDENT per-layer parameters (VPT-Deep); the
  paper's ablations show sharing prompts across layers (Shared-all)
  under-performs per-layer prompts, and that shallow-layer prompts
  matter more than deep ones.
* AIM (Yang et al., ICCV 2023) / ST-Adapter (Pan et al., NeurIPS 2022):
  lightweight residual adapters inserted between frozen blocks, with
  zero-initialised up-projections so training starts exactly at the
  frozen baseline.
* Vita-CLIP (Thompson et al., CVPR 2023): video-level prompts are
  conditioned on the video content rather than being purely static.

Channel A -- ``MaskPatchAdapter`` (the "mask" channel):
    Inserted AFTER selected frozen blocks (residual-adapter style, like
    AIM/ST-Adapter). Per insertion layer it re-selects a top-k patch
    mask (CLS->patch attention of that block + frame-difference motion
    reliability), runs the SAME spatial positions across frames through
    a bottleneck temporal multi-head attention mixed with a per-frame
    spatial MLP, and scatters the residual back onto the selected
    patches only.  The up-projection AND a per-layer scalar gate are
    both zero-initialised, so the network starts as the frozen baseline
    and learns its own enhancement strength.  There is NO
    reconstruction/correspondence branch here: motion is only used for
    SELECTION scoring, never as a loss that fights the enhancement
    objective (the design conflict that unbounded the old CVP residual).

Channel B -- ``LayeredVideoPrompts`` (the "global video prompt" channel):
    VPT-Deep-style: every configured frozen block receives its OWN
    prompts prepended to the token sequence (appended at the tail so the
    [CLS] and patch indices are untouched), and the prompt outputs are
    discarded after the block (prompts do not propagate between layers).
    Each layer's prompts are layer-specific static embeddings plus a
    video-conditioned term produced by a shared trunk with a
    PER-LAYER projection head and a zero-initialised per-layer gate --
    guaranteeing different semantics at different depths (the property
    validated by VPT's Shared-all ablation) while remaining
    video-conditioned (Vita-CLIP style).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["MaskPatchAdapter", "LayeredVideoPrompts"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _frame_difference(patch_tokens: torch.Tensor) -> torch.Tensor:
    """Bidirectional frame difference on L2-normalised fp32 patch tokens.

    Boundary frames use only the existing neighbour.  Returns [B, T, N].
    """
    z = F.normalize(patch_tokens.detach().float(), dim=-1)
    B, T, N, _ = z.shape
    if T == 1:
        return torch.zeros(B, 1, N, device=z.device)
    cos_past = (z * torch.cat([z[:, :1], z[:, :-1]], dim=1)).sum(-1)
    cos_future = (z * torch.cat([z[:, 1:], z[:, -1:]], dim=1)).sum(-1)
    has_past = torch.cat(
        [torch.zeros(B, 1, 1, device=z.device),
         torch.ones(B, T - 1, 1, device=z.device)],
        dim=1,
    )
    has_future = torch.cat(
        [torch.ones(B, T - 1, 1, device=z.device),
         torch.zeros(B, 1, 1, device=z.device)],
        dim=1,
    )
    diff = has_past * (1.0 - cos_past) + has_future * (1.0 - cos_future)
    return diff / (has_past + has_future).clamp_min(1.0)


def _rank_norm_01(x: torch.Tensor) -> torch.Tensor:
    order = x.argsort(dim=-1)
    ranks = order.argsort(dim=-1).to(torch.float32)
    return ranks / max(x.shape[-1] - 1, 1)


# ---------------------------------------------------------------------------
# Channel A: mask patch adapter (inserted after selected frozen blocks)
# ---------------------------------------------------------------------------

class MaskPatchAdapter(nn.Module):
    """Per-layer mask residual adapter over selected patches (AIM/ST style).

    The adapter is created once and applied to several insertion layers;
    every insertion layer owns its own gate / projections (via the
    ``nn.ModuleList`` fields) so layers can learn different strengths.
    """

    def __init__(
        self,
        d_model: int,
        num_frames: int,
        insertion_layers: List[int],          # 0-based block indices
        topk: int = 64,
        bottleneck: int = 192,
        num_heads: int = 4,
        motion_max_weight: float = 0.6,
    ) -> None:
        super().__init__()
        if not insertion_layers:
            raise ValueError("insertion_layers must be non-empty")
        self.d_model = int(d_model)
        self.num_frames = int(num_frames)
        self.insertion_layers = sorted(set(int(l) for l in insertion_layers))
        self.topk = int(topk)
        self.motion_max_weight = float(motion_max_weight)

        n = len(self.insertion_layers)
        # per-layer temporal attention (same spatial position across frames)
        self.temporal_attn = nn.ModuleList(
            nn.MultiheadAttention(bottleneck, num_heads, batch_first=True)
            for _ in range(n)
        )
        self.pre_norm = nn.ModuleList(nn.LayerNorm(d_model) for _ in range(n))
        self.attn_norm = nn.ModuleList(nn.LayerNorm(bottleneck) for _ in range(n))
        self.down_proj = nn.ModuleList(
            nn.Linear(d_model, bottleneck) for _ in range(n)
        )
        self.up_proj = nn.ModuleList(
            nn.Linear(bottleneck, d_model) for _ in range(n)
        )
        # NOTE: up-projection is NOT zero-initialised. A zero gate AND a
        # zero up-projection would deadlock the gradients (both receive
        # exactly-zero gradients). The per-layer gate below is the single
        # zero-initialised factor: output is the frozen baseline at step 0
        # and the gate receives a usable gradient from the first step.
        for up in self.up_proj:
            nn.init.xavier_uniform_(up.weight, gain=0.1)
            nn.init.zeros_(up.bias)
        # per-layer residual gate, zero-initialised (identity at start)
        self.gate_scale = nn.Parameter(torch.zeros(n))

    # -- selection ---------------------------------------------------------
    def _select(
        self,
        patch_tokens: torch.Tensor,   # [B, T, N, D]
        semantic: torch.Tensor,       # [B, T, N] detached CLS->patch attn
        topk: int,
        motion_max_weight: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Adaptive attention+motion top-k selection.

        Returns (indices [B,T,K], reliability [B,T]).
        """
        B, T, N, _ = patch_tokens.shape
        k = min(topk, N)
        diff = _frame_difference(patch_tokens)              # [B,T,N] fp32
        q50 = torch.quantile(diff, 0.5, dim=-1)
        q90 = torch.quantile(diff, 0.9, dim=-1)
        reliability = torch.clamp((q90 - q50) / (q90 + 1e-8), 0.0, 1.0)
        centered = torch.relu(diff - q50[..., None])
        w_d = motion_max_weight * reliability               # [B,T]
        score = (
            (1.0 - w_d)[..., None] * _rank_norm_01(semantic.detach().float())
            + w_d[..., None] * _rank_norm_01(centered)
        )
        return torch.topk(score, k=k, dim=-1).indices, reliability  # [B,T,K],[B,T]

    # -- forward -----------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,                # [1+N(+P), BT, D] time-first
        batch_size: int,
        num_frames: int,
        layer_index: int,
        cls_to_patch_attention: Optional[torch.Tensor],  # [BT, N] or None
    ) -> torch.Tensor:
        slot = self.insertion_layers.index(layer_index)
        gate = torch.tanh(self.gate_scale[slot])

        device = x.device
        n_tokens, bt, width = x.shape
        n_patch = n_tokens - 1                       # prompts already removed
        B, T = batch_size, num_frames

        patches = x[1:]                              # [N, BT, D]
        p5 = (
            patches.permute(1, 0, 2)
            .reshape(B, T, n_patch, width)
        )                                            # [B,T,N,D]

        # ---- selection scores -------------------------------------------
        if cls_to_patch_attention is not None:
            semantic = (
                cls_to_patch_attention.detach()
                .reshape(B, T, n_patch)
                .float()
            )
        else:
            semantic = torch.full(
                (B, T, n_patch), 1.0 / n_patch, device=device
            )
        indices, reliability = self._select(
            p5, semantic, self.topk, self.motion_max_weight
        )                                            # [B,T,K]

        K = indices.shape[-1]
        sel = p5.gather(
            2, indices[..., None].expand(-1, -1, -1, width)
        )                                            # [B,T,K,D]

        # ---- bottleneck + same-position temporal mixing ------------------
        sel_db = self.down_proj[slot](self.pre_norm[slot](sel))   # [B,T,K,db]
        sel_bt = sel_db.permute(0, 2, 1, 3).reshape(B * K, T, -1) # [B*K,T,db]
        attn_in = self.attn_norm[slot](sel_bt)
        temp_out, _ = self.temporal_attn[slot](
            attn_in, attn_in, attn_in, need_weights=False
        )
        mixed = sel_bt + temp_out                                  # [B*K,T,db]

        delta = self.up_proj[slot](mixed).reshape(
            B, K, T, width
        ).permute(0, 2, 1, 3)                                      # [B,T,K,D]
        delta = gate * delta

        update = torch.zeros_like(p5)
        update.scatter_(
            2, indices[..., None].expand(-1, -1, -1, width), delta
        )
        out5 = p5 + update

        # ---- stats (last applied layer only) ------------------------------
        with torch.no_grad():
            sel_norm = delta.float().norm(dim=-1).mean()
            base_norm = p5.detach().float().norm(dim=-1).mean()
            self._last_stats = {
                "mask_gate": gate.detach(),
                "mask_res_ratio": (
                    sel_norm / base_norm.clamp_min(1e-6)
                ).detach(),
                "mask_topk": torch.tensor(float(K)),
                "mask_motion_reliability": reliability.mean().detach(),
            }

        # restore time-first layout: [B,T,N,D] -> [N,B,T,D] -> [N,BT,D]
        out = torch.cat(
            [
                x[:1],
                out5.permute(2, 0, 1, 3).reshape(n_patch, bt, width),
            ],
            dim=0,
        )
        return out

    def current_stats(self) -> Dict[str, torch.Tensor]:
        return getattr(self, "_last_stats", {})


# ---------------------------------------------------------------------------
# Channel B: layered video prompts (VPT-Deep style, video-conditioned)
# ---------------------------------------------------------------------------

class LayeredVideoPrompts(nn.Module):
    """Per-layer independent, video-conditioned prompt tokens.

    ``forward`` returns the prompt tokens for ONE layer; the caller
    prepends/appends them to that block's input and discards the prompt
    outputs afterwards (VPT-Deep semantics).
    """

    def __init__(
        self,
        d_model: int,
        num_layers: int,                # total transformer depth (12)
        num_prompts: int = 4,
        cond_rank: int = 192,
        insert_layers: Optional[List[int]] = None,   # None = all layers
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.num_layers = int(num_layers)
        self.num_prompts = int(num_prompts)
        self.insert_layers = (
            list(range(self.num_layers))
            if insert_layers is None
            else sorted(set(int(l) for l in insert_layers))
        )
        if not self.insert_layers:
            raise ValueError("insert_layers must be non-empty")
        if any(l < 0 or l >= self.num_layers for l in self.insert_layers):
            raise ValueError(
                f"insert_layers must be in [0, {self.num_layers - 1}]"
            )

        # VPT-Deep: an independent prompt set per insertion layer.
        self.base_prompts = nn.Parameter(
            torch.randn(len(self.insert_layers), self.num_prompts, d_model)
            * 0.02
        )
        # shared conditioning trunk + per-layer heads (low-rank).
        # cond_up is normally initialised; only cond_gate is zero (see the
        # dead-gradient note in MaskPatchAdapter).
        self.cond_down = nn.Linear(d_model, cond_rank)
        self.cond_up = nn.ModuleList(
            nn.Linear(cond_rank, d_model) for _ in self.insert_layers
        )
        for up in self.cond_up:
            nn.init.xavier_uniform_(up.weight, gain=0.1)
            nn.init.zeros_(up.bias)
        # per-layer gates, zero-initialised: conditioning starts silent.
        self.cond_gate = nn.Parameter(torch.zeros(len(self.insert_layers)))
        self.prompt_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        video_embedding: torch.Tensor,   # [B, D] (detached video summary)
        layer_index: int,
    ) -> torch.Tensor:
        slot = self.insert_layers.index(layer_index)
        base = self.base_prompts[slot][None]              # [1,P,D]
        cond = self.cond_up[slot](F.gelu(self.cond_down(video_embedding)))
        prompts = base + torch.tanh(self.cond_gate[slot]) * cond[:, None, :]
        prompts = self.prompt_norm(prompts)               # [B,P,D]
        with torch.no_grad():
            self._last_stats = {
                "prompt_cond_gate": torch.tanh(
                    self.cond_gate[slot].detach()
                ),
                "prompt_norm": prompts.detach().float().norm(dim=-1).mean(),
            }
        return prompts

    def current_stats(self) -> Dict[str, torch.Tensor]:
        return getattr(self, "_last_stats", {})
