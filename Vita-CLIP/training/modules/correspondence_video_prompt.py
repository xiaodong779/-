#!/usr/bin/env python3
"""Correspondence-Guided Video Prompt Block (CVP Block), v2.

Changes vs v1 (paper-method revision):
  1. Patch selection fuses attention and frame-difference scores with a
     per-frame MOTION RELIABILITY weight (c_t from the q90/q50 spread of the
     raw frame differences) instead of a fixed 0.5/0.5 mix. Boundary frames
     use only the existing neighbor (no self-copy /2).
  2. The mask query is now TWO-STAGE: base query (mask token + projected
     frozen CLIP position embedding + time embedding) first attends to the
     CURRENT frame's UNSELECTED input-level patch tokens (pre-Block1, no
     self-attention yet, detached) producing a video-conditioned
     contextualized query, which then queries past/future frames. The base
     query never reads selected current patches (no leakage), and the
     context comes from the current video (no query collapse).
  3. Correspondence scope is FULL-FRAME by default: every one of the 196
     Block4 patches of the adjacent frames is a candidate (computed via
     batched MHA, no physical repeat of source tokens), with a bounded
     learnable spatial distance bias  lambda = max*sigmoid(raw)  subtracted
     from the attention logits BEFORE softmax. The hard 7x7 window remains
     only as the `local` ablation scope.

Unchanged from v1: insertion point (after frozen Block4), residual adapter
with bounded gamma and strict scatter-add, FrameMotionPool, lightweight
temporal adapter, 4-slot video prompt generator, correspondence loss.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _trunc_normal_(tensor: torch.Tensor, std: float = 0.02) -> torch.Tensor:
    return nn.init.trunc_normal_(tensor, std=std, a=-2.0 * std, b=2.0 * std)


def _rank_norm_01(x: torch.Tensor) -> torch.Tensor:
    """Per-last-dim rank normalization to [0, 1]. x must be detached fp32."""
    order = x.argsort(dim=-1)
    ranks = order.argsort(dim=-1).to(torch.float32)
    return ranks / max(x.shape[-1] - 1, 1)


def _frame_difference(patch_tokens: torch.Tensor) -> torch.Tensor:
    """Bidirectional frame difference on L2-normalized fp32 patch tokens.

    Boundary frames use ONLY the existing neighbor (no self-copy /2):
        D_0   = 1 - cos(z_0, z_1)
        D_T-1 = 1 - cos(z_T-1, z_T-2)
        D_t   = 0.5[(1-cos(z_t,z_{t-1})) + (1-cos(z_t,z_{t+1}))]  inside
    Returns [B, T, N] fp32.
    """
    z = F.normalize(patch_tokens.detach().float(), dim=-1)
    B, T, N, _ = z.shape
    cos_past = (z * torch.cat([z[:, :1], z[:, :-1]], dim=1)).sum(-1)
    cos_future = (z * torch.cat([z[:, 1:], z[:, -1:]], dim=1)).sum(-1)
    has_past = torch.cat(
        [torch.zeros(B, 1, 1, device=z.device), torch.ones(B, T - 1, 1, device=z.device)],
        dim=1,
    ) if T > 1 else torch.zeros(B, 1, 1, device=z.device)
    has_future = torch.cat(
        [torch.ones(B, T - 1, 1, device=z.device), torch.zeros(B, 1, 1, device=z.device)],
        dim=1,
    ) if T > 1 else torch.zeros(B, 1, 1, device=z.device)
    diff = has_past * (1.0 - cos_past) + has_future * (1.0 - cos_future)
    return diff / (has_past + has_future).clamp_min(1.0)


SELECTION_METHODS = (
    "attention_only",
    "motion_only",
    "fixed_attention_motion",
    "adaptive_attention_motion",
    "random_topk",
)


class PatchSelector(nn.Module):
    """Score fusion -> strict Top-K bool selection mask (fully detached).

    selection_method:
        attention_only            S = RankNorm(A)
        motion_only               S = RankNorm(D_center)
        fixed_attention_motion    S = 0.5*RankNorm(A) + 0.5*RankNorm(D_center)
        adaptive_attention_motion S = (1-wD)*RankNorm(A) + wD*RankNorm(D_center),
                                  wD = motion_max_weight * c_t  (main method)
        random_topk               S = uniform random (sanity ablation)

    Motion reliability: c_t = clamp((q90_t - q50_t) / (q90_t + eps), 0, 1)
    computed with torch.quantile in fp32 on the raw frame differences.
    """

    def __init__(
        self,
        topk: int = 64,
        selection_method: str = "adaptive_attention_motion",
        motion_max_weight: float = 0.6,
    ) -> None:
        super().__init__()
        if selection_method not in SELECTION_METHODS:
            raise ValueError(f"unknown selection method: {selection_method}")
        self.topk = int(topk)
        self.method = selection_method
        self.motion_max_weight = float(motion_max_weight)

    @torch.no_grad()
    def forward(
        self,
        patch_tokens: torch.Tensor,    # [B, T, N, D] (block4)
        semantic_score: torch.Tensor,  # [B, T, N] (mean-head CLS attn)
    ):
        B, T, N, _ = patch_tokens.shape
        k = min(self.topk, N)

        raw_diff = _frame_difference(patch_tokens)          # [B, T, N] fp32
        q50 = torch.quantile(raw_diff, 0.5, dim=-1)         # [B, T]
        q90 = torch.quantile(raw_diff, 0.9, dim=-1)         # [B, T]
        reliability = torch.clamp((q90 - q50) / (q90 + 1e-8), 0.0, 1.0)
        centered = torch.relu(raw_diff - q50[..., None])    # [B, T, N]

        rank_a = _rank_norm_01(semantic_score.detach().float())
        rank_d = _rank_norm_01(centered)

        if self.method == "attention_only":
            score = rank_a
            w_d = torch.zeros_like(reliability)
        elif self.method == "motion_only":
            score = rank_d
            w_d = torch.ones_like(reliability)
        elif self.method == "fixed_attention_motion":
            w_d = torch.full_like(reliability, 0.5)
            score = 0.5 * rank_a + 0.5 * rank_d
        elif self.method == "random_topk":
            score = torch.rand(B, T, N, device=patch_tokens.device)
            w_d = torch.zeros_like(reliability)
        else:  # adaptive_attention_motion (main)
            w_d = self.motion_max_weight * reliability       # wD = 0.6 * c_t
            score = (1.0 - w_d)[..., None] * rank_a + w_d[..., None] * rank_d

        topk = torch.topk(score, k=k, dim=-1)
        selection_mask = torch.zeros_like(score, dtype=torch.bool)
        selection_mask.scatter_(-1, topk.indices, True)

        stats = {
            "motion_reliability_mean": reliability.mean().detach(),
            "motion_reliability_per_frame": reliability.detach(),   # [B, T]
            "attention_weight_mean": (1.0 - w_d).mean().detach(),
            "motion_weight_mean": w_d.mean().detach(),
            "q50_motion": q50.mean().detach(),
            "q90_motion": q90.mean().detach(),
            "semantic_score_mean": semantic_score.detach().float().mean(),
            "semantic_score_std": semantic_score.detach().float().std(),
            "motion_score_mean": centered.mean().detach(),
            "motion_score_std": centered.std().detach(),
            "score": score.detach(),                     # [B, T, N]
            "frame_diff_raw": raw_diff.detach(),         # [B, T, N]
        }
        return selection_mask, topk.indices, stats


class LocalCorrespondenceAttention(nn.Module):
    """Two-stage mask query + adjacent-frame correspondence.

    Stage 1 (current-frame context): the base mask query attends to the
    UNSELECTED input-level patch tokens of the current frame (pre-Block1,
    detached, no self-attention yet -> no leakage) producing a
    video-conditioned contextualized query.

    Stage 2: q_ctx queries past/future frames.
        scope="full":  all 196 Block4 patches of the adjacent frames are
            candidates (batched MHA over [B*T, 196, db]; no physical repeat
            of source tokens), with a bounded learnable spatial distance
            bias  -lambda * d^2  added to the logits before softmax
            (lambda = spatial_bias_max * sigmoid(raw), init spatial_bias_init).
        scope="local": the v1 hard window (window_size x window_size,
            zero-padded) — kept only for ablation.

    Past and future branches share all Q/K/V projections. Boundary frames
    reuse the existing branch's output (C- = C+ on frame 0, C+ = C- on the
    last frame).
    """

    def __init__(
        self,
        d_model: int,
        bottleneck: int = 192,
        num_heads: int = 4,
        window_size: int = 7,
        grid_size: int = 14,
        num_frames: int = 8,
        corr_scope: str = "full",
        spatial_bias_max: float = 4.0,
        spatial_bias_init: float = 1.0,
    ) -> None:
        super().__init__()
        if corr_scope not in ("full", "local"):
            raise ValueError(f"unknown corr_scope: {corr_scope}")
        if spatial_bias_max <= 0 and corr_scope == "full":
            # allowed: lambda collapses to 0 (full_without_spatial_bias)
            pass
        self.d_model = d_model
        self.bottleneck = bottleneck
        self.window = int(window_size)
        self.grid = int(grid_size)
        self.num_frames = int(num_frames)
        self.scope = corr_scope
        self.spatial_bias_max = float(spatial_bias_max)
        self.num_heads = int(num_heads)

        # ---- base mask query inputs ----
        self.mask_token = nn.Parameter(torch.zeros(1, 1, bottleneck))
        _trunc_normal_(self.mask_token, std=0.02)
        self.pos_proj = nn.Linear(d_model, bottleneck)
        self.time_embedding = nn.Parameter(torch.zeros(num_frames, bottleneck))
        _trunc_normal_(self.time_embedding, std=0.02)

        # ---- stage 1: current-frame context attention ----
        self.ctx_norm_q = nn.LayerNorm(bottleneck)
        self.ctx_norm_kv = nn.LayerNorm(d_model)
        self.ctx_k_proj = nn.Linear(d_model, bottleneck)
        self.ctx_v_proj = nn.Linear(d_model, bottleneck)
        self.ctx_attn = nn.MultiheadAttention(
            bottleneck, num_heads, batch_first=True
        )
        self.ctx_norm_out = nn.LayerNorm(bottleneck)

        # ---- stage 2: correspondence (shared past/future projections) ----
        self.norm_q = nn.LayerNorm(bottleneck)
        self.norm_kv = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(bottleneck, bottleneck)
        self.k_proj = nn.Linear(d_model, bottleneck)
        self.v_proj = nn.Linear(d_model, bottleneck)
        self.cross_attn = nn.MultiheadAttention(
            bottleneck, num_heads, batch_first=True
        )
        self.fusion = nn.Linear(3 * bottleneck, d_model)

        # ---- bounded learnable spatial distance bias ----
        # lambda_sp = spatial_bias_max * sigmoid(raw_lambda_sp)
        if spatial_bias_max <= 0:
            init_raw = -10.0  # lambda ~ 0
        else:
            ratio = spatial_bias_init / spatial_bias_max
            ratio = min(max(ratio, 1e-6), 1.0 - 1e-6)
            init_raw = math.log(ratio / (1.0 - ratio))
        self.raw_lambda_sp = nn.Parameter(
            torch.tensor(init_raw, dtype=torch.float32)
        )

        # normalized 2D coordinates of the patch grid, range [0, 1]
        ys = torch.linspace(0.0, 1.0, self.grid)
        xs = torch.linspace(0.0, 1.0, self.grid)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack(
            [grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1
        )  # [N, 2] (x, y)
        self.register_buffer("patch_coordinates", coords, persistent=False)

    # -- helpers ---------------------------------------------------------

    def lambda_sp(self) -> torch.Tensor:
        return self.spatial_bias_max * torch.sigmoid(self.raw_lambda_sp.float())

    def _windows(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """[B, T, N, w*w, D] zero-padded sliding windows (local scope only)."""
        B, T, N, D = patch_tokens.shape
        g, w = self.grid, self.window
        pad = w // 2
        x = (
            patch_tokens.detach()
            .transpose(-1, -2)
            .reshape(B * T, D, g, g)
        )
        x = F.pad(x, (pad, pad, pad, pad))
        x = F.unfold(x, kernel_size=w, stride=1)
        x = x.view(B, T, D, w * w, g * g).permute(0, 1, 4, 3, 2)
        return x.contiguous()

    def _current_context(
        self,
        base_queries: torch.Tensor,        # [B, T, K, db]
        input_patch_tokens: torch.Tensor,  # [B, T, N, D] (pre-Block1, detached)
        selected_indices: torch.Tensor,    # [B, T, K]
        num_visible: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, N, D = input_patch_tokens.shape
        vis_idx = torch.argsort(
            (~self._selection_mask_from_indices(selected_indices, N)).to(torch.uint8),
            dim=-1,
            descending=True,
            stable=True,
        )[:, :, :num_visible]  # [B, T, N-K]
        visible = input_patch_tokens.gather(
            2, vis_idx[:, :, :, None].expand(-1, -1, -1, D)
        )  # [B, T, N-K, D]

        q = self.ctx_norm_q(base_queries).reshape(B * T, -1, self.bottleneck)
        kv = self.ctx_norm_kv(visible).reshape(B * T, num_visible, D)
        k = self.ctx_k_proj(kv)
        v = self.ctx_v_proj(kv)
        out, _ = self.ctx_attn(q, k, v, need_weights=False)
        q_ctx = self.ctx_norm_out(
            base_queries.reshape(B, T, -1, self.bottleneck)
            + out.view(B, T, -1, self.bottleneck)
        )
        return q_ctx, vis_idx

    @staticmethod
    def _selection_mask_from_indices(
        indices: torch.Tensor, n_total: int
    ) -> torch.Tensor:
        mask = torch.zeros(
            *indices.shape[:-1], n_total, dtype=torch.bool, device=indices.device
        )
        mask.scatter_(-1, indices, True)
        return mask

    def _spatial_bias_mask(
        self,
        selected_indices: torch.Tensor,  # [B, T, K]
        lambda_sp: torch.Tensor,         # scalar tensor (fp32)
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """-lambda * d^2 attention bias, shape [B*T*num_heads, K, 196]."""
        B, T, K = selected_indices.shape
        sel_coords = self.patch_coordinates.to(torch.float32)[
            selected_indices.reshape(-1)
        ].view(B, T, K, 2)
        d2 = (
            (sel_coords[..., None, :] - self.patch_coordinates[None, None, None])
            ** 2
        ).sum(-1)  # [B, T, K, 196] fp32
        bias = (-lambda_sp.to(d2.device) * d2)  # negative for far patches
        bias = bias[:, :, None, :, :].expand(
            B, T, self.num_heads, K, self.patch_coordinates.shape[0]
        )
        return (
            bias.reshape(B * T * self.num_heads, K, -1).to(dtype),
            d2.detach(),
        )

    # -- public API ------------------------------------------------------

    def build_base_queries(
        self,
        selected_indices: torch.Tensor,    # [B, T, K]
        clip_pos_embedding: torch.Tensor,  # [N, D] (frozen, detached)
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Base mask query = mask token + projected frozen CLIP pos emb + time
        embedding. Reads NO current-patch content (no reconstruction leakage)."""
        T = selected_indices.shape[1]
        pos_q = self.pos_proj(
            clip_pos_embedding.detach().to(selected_indices.device, dtype)
        )[selected_indices]
        time_q = self.time_embedding[:T].to(dtype)[None, :, None]
        return self.mask_token.to(dtype) + pos_q + time_q

    def forward(
        self,
        patch_tokens: torch.Tensor,        # [B, T, N, D] (frozen block4)
        selected_indices: torch.Tensor,    # [B, T, K]
        clip_pos_embedding: torch.Tensor,  # [N, D] (frozen, detached)
        input_patch_tokens: torch.Tensor,  # [B, T, N, D] (pre-Block1, detached)
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B, T, N, D = patch_tokens.shape
        K = selected_indices.shape[-1]
        dtype = patch_tokens.dtype

        base_queries = self.build_base_queries(
            selected_indices, clip_pos_embedding, dtype
        )

        # ---- stage 1: current-frame visible context ----
        num_visible = N - K
        q_ctx, visible_indices = self._current_context(
            base_queries, input_patch_tokens, selected_indices, num_visible
        )

        # ---- stage 2: correspondence ----
        q = self.q_proj(self.norm_q(q_ctx)).reshape(B * T, K, self.bottleneck)
        lambda_sp = self.lambda_sp()

        corr_stats: Dict[str, torch.Tensor] = {}
        if self.scope == "full":
            # shifted views WITHOUT wrap-around: frame 0's past and frame
            # T-1's future point at the frame itself, but their outputs are
            # discarded by the boundary handling below.
            past_src = torch.cat(
                [patch_tokens[:, :1], patch_tokens[:, :-1]], dim=1
            )  # frame t-1 (frame 0 -> itself; corrected below)
            future_src = torch.cat(
                [patch_tokens[:, 1:], patch_tokens[:, -1:]], dim=1
            )  # frame t+1 (frame T-1 -> itself; corrected below)

            def _run(src: torch.Tensor, bias_mask: Optional[torch.Tensor]):
                kv = self.norm_kv(src).reshape(B * T, N, D)
                k = self.k_proj(kv)
                v = self.v_proj(kv)
                out, w = self.cross_attn(
                    q, k, v, need_weights=True, average_attn_weights=True,
                    attn_mask=bias_mask,
                )
                return out.view(B, T, K, self.bottleneck), w.view(B, T, K, N)

            bias_mask, d2 = self._spatial_bias_mask(
                selected_indices, lambda_sp, q.dtype
            )
            c_past, w_past = _run(past_src, bias_mask)
            c_future, w_future = _run(future_src, bias_mask)
            corr_stats["full_corr_attention"] = w_past.detach()  # [B,T,K,196]
            corr_stats["full_corr_attention_entropy"] = (
                -(w_past.clamp_min(1e-9).log() * w_past).sum(-1).mean().detach()
            )
        else:
            windows = self._windows(patch_tokens)  # [B, T, N, w2, D]
            gather_idx = selected_indices[:, :, :, None, None].expand(
                B, T, K, self.window * self.window, D
            )
            past_w = windows.gather(2, gather_idx)
            future_w = windows.gather(2, gather_idx)
            past_src = past_w.reshape(B, T, K, -1, D)
            future_src = future_w.reshape(B, T, K, -1, D)

            def _run_local(src: torch.Tensor):
                kv = self.norm_kv(src).reshape(B * T * K, -1, D)
                k = self.k_proj(kv)
                v = self.v_proj(kv)
                qv = q.reshape(B * T * K, 1, self.bottleneck)
                out, _ = self.cross_attn(qv, k, v, need_weights=False)
                return out.view(B, T, K, self.bottleneck)

            c_past = _run_local(past_src)
            c_future = _run_local(future_src)

        # boundary frames reuse the existing branch (never random/uninit)
        if T > 1:
            c_past = torch.cat([c_future[:, :1], c_past[:, 1:]], dim=1)
            c_future = torch.cat(
                [c_future[:, :-1], c_past[:, -1:]], dim=1
            )

        reconstructed = self.fusion(
            torch.cat([c_past, c_future, c_future - c_past], dim=-1)
        )  # [B, T, K, D]

        corr_stats["lambda_sp"] = lambda_sp.detach()
        corr_stats["current_context_query_variance"] = (
            q_ctx.detach().float().var(dim=0).mean()
        )
        corr_stats["visible_indices"] = visible_indices.detach()
        return reconstructed, corr_stats

    def window_index_map(self) -> torch.Tensor:
        """[g*g, w*w] source patch index per (center, slot); -1 = zero pad."""
        g, w = self.grid, self.window
        pad = w // 2
        centers = torch.arange(g * g).view(g, g)
        padded = torch.full((g + 2 * pad, g + 2 * pad), -1, dtype=torch.long)
        padded[pad : pad + g, pad : pad + g] = centers
        windows = padded.unfold(0, w, 1).unfold(1, w, 1)
        return windows.reshape(g * g, w * w)


class FrameMotionPool(nn.Module):
    """Frame's own CLS token queries its K enhanced patches (bottleneck MHA)."""

    def __init__(self, d_model: int, bottleneck: int = 192, num_heads: int = 4) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, bottleneck)
        self.k_proj = nn.Linear(d_model, bottleneck)
        self.v_proj = nn.Linear(d_model, bottleneck)
        self.attn = nn.MultiheadAttention(bottleneck, num_heads, batch_first=True)
        self.out_proj = nn.Linear(bottleneck, d_model)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        cls_tokens: torch.Tensor,           # [B, T, D]
        enhanced_selected: torch.Tensor,    # [B, T, K, D]
    ) -> torch.Tensor:
        B, T, K, _ = enhanced_selected.shape
        q = self.q_proj(self.norm(cls_tokens)).reshape(B * T, 1, -1)
        kv_tokens = enhanced_selected.reshape(B * T, K, -1)
        k = self.k_proj(kv_tokens)
        v = self.v_proj(kv_tokens)
        out, _ = self.attn(q, k, v, need_weights=False)
        out = self.out_proj(out.view(B, T, -1))
        return self.out_norm(cls_tokens + out)


class LightweightTemporalAdapter(nn.Module):
    """One bottleneck temporal transformer over the T frame tokens."""

    def __init__(
        self,
        d_model: int,
        bottleneck: int = 192,
        num_heads: int = 4,
        num_frames: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.down = nn.Linear(d_model, bottleneck)
        self.time_embedding = nn.Parameter(torch.zeros(num_frames, bottleneck))
        _trunc_normal_(self.time_embedding, std=0.02)
        self.norm = nn.LayerNorm(bottleneck)
        self.block = nn.TransformerEncoderLayer(
            d_model=bottleneck,
            nhead=num_heads,
            dim_feedforward=bottleneck * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.up = nn.Linear(bottleneck, d_model)

    def forward(self, frame_tokens: torch.Tensor) -> torch.Tensor:
        B, T, _ = frame_tokens.shape
        h = self.down(frame_tokens)
        h = self.norm(h + self.time_embedding[:T].to(h.dtype)[None])
        h = self.block(h)
        return frame_tokens + self.up(h)


class VideoPromptGenerator(nn.Module):
    """4 learnable slot queries read the T frame tokens -> [B, 4, D]."""

    def __init__(
        self,
        d_model: int,
        bottleneck: int = 192,
        num_heads: int = 4,
        num_prompts: int = 4,
        num_frames: int = 8,
        beta_min: float = 0.5,
        beta_max: float = 1.5,
        beta_init: float = 1.0,
    ) -> None:
        super().__init__()
        if not beta_min < beta_max:
            raise ValueError("beta_min must be < beta_max")
        if not beta_min <= beta_init <= beta_max:
            raise ValueError("beta_init outside [beta_min, beta_max]")
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)

        self.video_queries = nn.Parameter(
            torch.zeros(num_prompts, bottleneck)
        )
        _trunc_normal_(self.video_queries, std=0.02)
        self.norm_q = nn.LayerNorm(bottleneck)
        self.norm_kv = nn.LayerNorm(d_model)
        self.k_proj = nn.Linear(d_model, bottleneck)
        self.v_proj = nn.Linear(d_model, bottleneck)
        self.attn = nn.MultiheadAttention(bottleneck, num_heads, batch_first=True)
        self.prompt_up = nn.Linear(bottleneck, d_model)
        self.prompt_norm = nn.LayerNorm(d_model)

        ratio = (beta_init - beta_min) / (beta_max - beta_min)
        ratio = min(max(ratio, 1e-6), 1.0 - 1e-6)
        self.raw_beta = nn.Parameter(
            torch.tensor(math.log(ratio / (1.0 - ratio)), dtype=torch.float32)
        )

    def beta(self) -> torch.Tensor:
        return self.beta_min + (self.beta_max - self.beta_min) * torch.sigmoid(
            self.raw_beta.float()
        )

    def forward(self, frame_tokens: torch.Tensor) -> torch.Tensor:
        B = frame_tokens.shape[0]
        q = self.video_queries[None].expand(B, -1, -1)
        q = self.norm_q(q)
        k = self.k_proj(self.norm_kv(frame_tokens))
        v = self.v_proj(self.norm_kv(frame_tokens))
        out, _ = self.attn(q, k, v, need_weights=False)
        beta = self.beta().to(out.dtype)
        q_prime = q + beta * out
        return self.prompt_norm(self.prompt_up(q_prime))


class CorrespondenceVideoPromptBlock(nn.Module):
    """Full CVP Block v2."""

    def __init__(
        self,
        d_model: int,
        num_frames: int = 8,
        grid_size: int = 14,
        topk: int = 64,
        window_size: int = 7,
        num_prompts: int = 4,
        bottleneck: int = 192,
        num_heads: int = 4,
        gamma_min: float = 0.1,
        gamma_max: float = 0.5,
        gamma_init: float = 0.2,
        beta_min: float = 0.5,
        beta_max: float = 1.5,
        beta_init: float = 1.0,
        selection_method: str = "adaptive_attention_motion",
        motion_max_weight: float = 0.6,
        use_current_context_attention: bool = True,
        corr_scope: str = "full",
        spatial_bias_max: float = 4.0,
        spatial_bias_init: float = 1.0,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.num_frames = int(num_frames)
        self.grid_size = int(grid_size)
        self.topk = int(topk)
        self.gamma_min = float(gamma_min)
        self.gamma_max = float(gamma_max)
        self.use_current_context_attention = bool(use_current_context_attention)
        if not self.gamma_min < self.gamma_max:
            raise ValueError("gamma_min must be < gamma_max")
        if not self.gamma_min <= gamma_init <= self.gamma_max:
            raise ValueError("gamma_init outside [gamma_min, gamma_max]")

        self.selector = PatchSelector(
            topk=topk,
            selection_method=selection_method,
            motion_max_weight=motion_max_weight,
        )
        self.correspondence = LocalCorrespondenceAttention(
            d_model=d_model,
            bottleneck=bottleneck,
            num_heads=num_heads,
            window_size=window_size,
            grid_size=grid_size,
            num_frames=num_frames,
            corr_scope=corr_scope,
            spatial_bias_max=spatial_bias_max,
            spatial_bias_init=spatial_bias_init,
        )
        self.motion_pool = FrameMotionPool(d_model, bottleneck, num_heads)
        self.temporal = LightweightTemporalAdapter(
            d_model, bottleneck, num_heads, num_frames
        )
        self.prompt_generator = VideoPromptGenerator(
            d_model, bottleneck, num_heads, num_prompts, num_frames,
            beta_min, beta_max, beta_init,
        )

        ratio = (gamma_init - self.gamma_min) / (self.gamma_max - self.gamma_min)
        ratio = min(max(ratio, 1e-6), 1.0 - 1e-6)
        self.raw_gamma = nn.Parameter(
            torch.tensor(math.log(ratio / (1.0 - ratio)), dtype=torch.float32)
        )

        res_bottleneck = max(d_model // 4, 1)
        self.residual_down = nn.Linear(d_model, res_bottleneck)
        self.residual_norm = nn.LayerNorm(d_model)
        self.residual_up = nn.Linear(res_bottleneck, d_model)
        nn.init.xavier_uniform_(self.residual_down.weight)
        nn.init.zeros_(self.residual_down.bias)
        nn.init.normal_(self.residual_up.weight, std=0.01)
        nn.init.zeros_(self.residual_up.bias)

    def gamma(self) -> torch.Tensor:
        return self.gamma_min + (self.gamma_max - self.gamma_min) * torch.sigmoid(
            self.raw_gamma.float()
        )

    def forward(
        self,
        tokens: torch.Tensor,               # [B, T, 1+N, D] (block4)
        attn_weights: torch.Tensor,         # [B, T, heads, 1+N, 1+N]
        clip_pos_embedding: torch.Tensor,   # [N, D] frozen CLIP patch pos emb
        input_patch_tokens: torch.Tensor,   # [B, T, N, D] pre-Block1 (detached)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        B, T, L, D = tokens.shape
        N = L - 1
        cls_tokens = tokens[:, :, 0, :]
        patch_tokens = tokens[:, :, 1:, :]

        semantic = attn_weights.mean(dim=2)[:, :, 0, 1:].detach().float()

        selection_mask, selected_indices, sel_stats = self.selector(
            patch_tokens, semantic
        )

        reconstructed, corr_stats = self.correspondence(
            patch_tokens, selected_indices, clip_pos_embedding,
            input_patch_tokens,
        )  # [B, T, K, D]

        current_selected = patch_tokens.gather(
            2, selected_indices[:, :, :, None].expand(-1, -1, -1, D)
        )
        delta = reconstructed - current_selected
        residual = self.residual_up(
            F.gelu(self.residual_down(self.residual_norm(delta)))
        )
        gamma = self.gamma().to(residual.dtype)

        residual_update = torch.zeros_like(patch_tokens)
        residual_update.scatter_(
            2,
            selected_indices[:, :, :, None].expand(-1, -1, -1, D),
            gamma * residual,
        )
        enhanced_patches = patch_tokens + residual_update

        enhanced_selected = enhanced_patches.gather(
            2, selected_indices[:, :, :, None].expand(-1, -1, -1, D)
        )
        frame_tokens = self.motion_pool(cls_tokens, enhanced_selected)
        temporal_out = self.temporal(frame_tokens)
        video_prompts = self.prompt_generator(temporal_out)

        target = current_selected.detach().float()
        cos = F.cosine_similarity(reconstructed.float(), target, dim=-1)
        corr_loss = (1.0 - cos).mean()

        with torch.no_grad():
            outside = (residual_update * (~selection_mask)[..., None]).abs().max()
            residual_norm = residual_update.float().norm(dim=-1)
            selected_norm = residual_norm.gather(2, selected_indices)
            base_norm = patch_tokens.detach().float().norm(dim=-1)
            stats: Dict[str, torch.Tensor] = {
                "gamma": gamma.detach(),
                "beta": self.prompt_generator.beta().detach(),
                "selected_residual_norm": selected_norm.mean().detach(),
                "base_patch_norm": base_norm.mean().detach(),
                "residual_to_base_ratio": (
                    selected_norm.mean() / base_norm.mean().clamp_min(1e-6)
                ).detach(),
                "corr_cosine": cos.mean().detach(),
                "outside_selected_max_diff": outside.detach(),
                "prompt_norm": video_prompts.detach().float().norm(dim=-1).mean(),
                "prompt_batch_variance": video_prompts.detach().float().var(dim=0).mean(),
            }
            stats.update({k: v for k, v in sel_stats.items() if k not in ("score", "frame_diff_raw")})
            stats.update({k: v for k, v in corr_stats.items() if k != "visible_indices"})
            debug = {
                "selection_mask": selection_mask.detach(),
                "semantic_score": semantic.detach(),
                "residual_norm": residual_norm.detach(),
                "selected_indices": selected_indices.detach(),
                "score": sel_stats["score"],
                "frame_diff_raw": sel_stats["frame_diff_raw"],
                "motion_reliability_per_frame": sel_stats["motion_reliability_per_frame"],
                "visible_indices": corr_stats["visible_indices"],
                "full_corr_attention": corr_stats.get("full_corr_attention"),
            }
            # centered motion map for visualization
            diff = _frame_difference(patch_tokens)
            q50 = torch.quantile(diff, 0.5, dim=-1)
            debug["motion_score"] = torch.relu(diff - q50[..., None]).detach()

        return enhanced_patches, video_prompts, corr_loss, {**stats, "debug": debug}
