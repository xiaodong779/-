"""RAP-style sparse-and-correlated temporal module for Vita-CLIP.

Replaces the full-coverage Mamba branch of ``VitaCLIPDiscriminativeDual``
with the sparse ASA (Asynchronous Self-Attention) branch of RAP
("Efficient Text-Video Retrieval with Sparse-and-Correlated Adapter",
arXiv:2405.19465):

1. Discriminative region selection (unchanged, STOP-inspired):
   - A_t = CLS -> patch attention of frame t (real attention weights);
   - M_t = temporal motion strength (frame-difference squared mean);
   - W_t = alpha * A_t + (1 - alpha) * M_t + learnable score(patch);
   - keep the top-k patches per frame (k = top_k, default 12, alpha 0.4);
   - the same top-k support set yields a hard 0/1 mask (entry seed
     positions) and soft weights (softmax over w / tau, gradient bridge).

2. Dual-branch temporal modelling:
   - Conv3d branch (unchanged): ALL patches are written back onto the
     14x14 grid; a depthwise Conv3d(3, 3, 3) models the local spatio-temporal
     neighbourhood over the complete grid, then a pointwise 1x1x1 conv mixes
     channels. Full-coverage.
   - ASA branch (replaces Mamba): RAP asynchronous self-attention over the
     SELECTED top-k tokens only. The query stays at its original position,
     while keys/values are sampled from query-dependent offset positions
     (t + delta, n + gamma). The per-token offsets (delta, gamma) are
     predicted by a small ``offset_net`` from the selected token's OWN query
     features (zero-initialised so sampling is identity at init) - a true
     local correspondence, not a global uniform shift. Sampling uses soft
     (bilinear) indexing along both axes so the offsets stay trainable.
     Unselected tokens contribute zero to this branch - this sparsity is
     the core of RAP's efficiency.

3. Patch-level gating fusion (unchanged):
   - stats = concat(GAP_T(conv_delta), GAP_T(asa_delta)) -> Linear(2*bn->bn)
     -> GELU -> Linear(bn -> 1) -> sigmoid (last layer zero-initialised
     -> sigma = 0.5 at init);
   - fused = sigma * conv + (1 - sigma) * asa.

4. Entry seed boost (unchanged, initial guidance only):
   - before modelling, the per-frame top-k action regions are emphasised
     ONCE on the raw patch tokens: patch += tanh(seed_scale) * mask.

5. Write-back (soft-enhanced):
   - patch' = patch + tanh(residual_scale) * fused * enhance, with
     enhance = 1 + clamp(enhance_scale, 0, enhance_max) * soft_w:
     EVERY position receives its temporal delta (the conv branch modelled
     the whole grid, the ASA branch only the selected top-k - unselected
     positions therefore get the conv-only contribution, matching RAP's
     sparse semantics), and the SELECTED positions additionally get a
     confidence-proportional boost (unselected: soft_w = 0 -> enhance = 1;
     at init enhance_scale = 0 -> identity -> baseline behaviour);
   - cls' = cls + sum_j soft_w_j * fused_j (soft aggregation gradient
     bridge, required because ln_post reads only the CLS token).

LoRM (temporal low-rank scale & shift) is intentionally NOT introduced in
this first version - kept as a later ablation extension.

Input / output shape: [B*T, N, D] (same as the frequency adapter).
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple, Union, cast

import torch
from torch import Tensor, nn

from VitaCLIP_discriminative_dual import (  # pyright: ignore[reportImplicitRelativeImport]
    _Conv3DBranch,
    _fp32_layer_norm,  # noqa: F401 (re-exported for callers)
    _fp32_linear,
)


class _ASABranch(nn.Module):
    """RAP true asynchronous cross-frame attention over the top-k tokens.

    Only the per-frame selected tokens (``top_indices``) are attended to:
    the query stays at its original (t, h, w) position, while keys/values
    are sampled from query-dependent offset positions. Two RAP paper
    (arXiv:2405.19465) semantics are implemented here:

    1. TRUE 2-D spatial offsets: the flat patch index is decomposed into
       (h, w) grid coordinates, and ``offset_net`` predicts a 3-D offset
       (off_t, off_h, off_w) per query. Offsets move along the actual
       spatial axes of the H x W grid (not the flattened patch axis, which
       would break spatial adjacency at row boundaries).

    2. TRUE cross-frame attention: each query samples S = window_size^2
       candidate k/v positions inside a window_size x window_size spatial
       window centred at the offset anchor (t + off_t, h + off_h,
       w + off_w) (window_size=5 -> S=25, a reduced 7x7 paper window).
       The attention matrix becomes [B, T, H, K, S] with softmax along
       the S candidates, so the query really computes weights against
       multiple cross-frame tokens (not a single sampled point).

    The per-token offsets are predicted by a small ``offset_net`` from the
    selected token's own query features (zero-initialised so the anchor
    equals the query position at init and attention starts uniformly over
    the local window). 3-D soft (bilinear) indexing along the frame, the
    h and the w axes keeps the offsets fully differentiable; out-of-range
    candidate positions are clamped to the grid edge (re-sampled, still
    differentiable). Unselected positions output exactly zero.

    Inputs:
        x:           [B, T, N_p, D] bottleneck features (all positions)
        top_indices: [B, T, K] selected patch indices per frame

    Outputs:
        delta: [B, T, N_p, D] sparse branch output (0 where unselected).
        off_t: [B, T, K] actual per-token time offsets used (float32).
        off_h: [B, T, K] actual per-token row offsets used (float32).
        off_w: [B, T, K] actual per-token column offsets used (float32).
    """

    def __init__(
        self,
        dim: int = 192,
        num_spatial_patches: int = 196,
        heads: int = 8,
        time_offset_max: float = 1.0,
        space_offset_max: float = 4.0,
        window_size: int = 5,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive.")
        if num_spatial_patches <= 0:
            raise ValueError("num_spatial_patches must be positive.")
        if heads <= 0 or dim % heads != 0:
            raise ValueError(f"heads={heads} must divide dim={dim}.")
        if time_offset_max < 0:
            raise ValueError("time_offset_max must be non-negative.")
        if space_offset_max < 0:
            raise ValueError("space_offset_max must be non-negative.")
        if window_size <= 0 or window_size % 2 == 0:
            raise ValueError(
                f"window_size={window_size} must be a positive odd number."
            )

        self.dim = int(dim)
        self.num_spatial_patches = int(num_spatial_patches)
        self.heads = int(heads)
        self.head_dim = self.dim // self.heads
        self.time_offset_max = float(time_offset_max)
        self.space_offset_max = float(space_offset_max)
        self.eps = float(eps)
        self.window_size = int(window_size)
        self.window_half = self.window_size // 2
        self.num_candidates = self.window_size * self.window_size

        # The spatial grid is assumed square (ViT-B/16: 196 = 14 x 14);
        # a non-square grid is not supported by the 2-D offset semantics.
        grid_side = int(round(math.sqrt(self.num_spatial_patches)))
        if grid_side * grid_side != self.num_spatial_patches:
            raise ValueError(
                f"num_spatial_patches={num_spatial_patches} is not a perfect "
                "square; the RAP ASA branch requires a square spatial grid."
            )
        self.grid_h = grid_side
        self.grid_w = grid_side

        # Query/key/value projections (dim -> dim) and output projection.
        self.W_q = nn.Linear(dim, dim, bias=False)
        self.W_k = nn.Linear(dim, dim, bias=False)
        self.W_v = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        # Query-dependent offset predictor: maps the selected token's query
        # feature to a per-token (time, row, column) offset triple. The
        # last layer is zero-initialised so offsets start at exactly 0 ->
        # the anchor equals the query position at init (safe, and the
        # offsets become trainable local correlations).
        self.offset_net = nn.Sequential(
            nn.Linear(dim, max(dim // 4, 8)),
            nn.GELU(),
            nn.Linear(max(dim // 4, 8), 3),
        )
        offset_last = self.offset_net[-1]
        assert isinstance(offset_last, nn.Linear)
        nn.init.zeros_(offset_last.weight)
        nn.init.zeros_(offset_last.bias)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Xavier on all projections (same family as the dual up-projection)."""
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    @torch.cuda.amp.autocast(enabled=False)
    def forward(
        self, x: Tensor, top_indices: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Forward [B, T, N, D] + [B, T, K] -> (delta, off_t, off_h, off_w)."""
        in_dtype = x.dtype
        x = x.float()
        batch_size, num_frames, num_patches, dim = x.shape
        head_dim = self.head_dim
        heads = self.heads
        device = x.device
        grid_h = self.grid_h
        grid_w = self.grid_w
        num_candidates = self.num_candidates

        # 1. Project q/k/v and reshape to multi-head [B, T, N, H, Hd].
        q = _fp32_linear(self.W_q, x).view(
            batch_size, num_frames, num_patches, heads, head_dim
        )
        k = _fp32_linear(self.W_k, x).view(
            batch_size, num_frames, num_patches, heads, head_dim
        )
        v = _fp32_linear(self.W_v, x).view(
            batch_size, num_frames, num_patches, heads, head_dim
        )

        # 2. Gather the selected queries at their ORIGINAL positions (the
        #    query does not move; only k/v are sampled from offset sites).
        num_selected = top_indices.shape[-1]
        idx_sel = top_indices.unsqueeze(-1).unsqueeze(-1).expand(
            batch_size, num_frames, num_selected, heads, head_dim
        )
        q_sel = torch.gather(q, 2, idx_sel)  # [B, T, K, H, Hd]

        # 3. Per-token dynamic 3-D offsets from the selected query features:
        #    [B, T, K, D] -> offset_net -> [B, T, K, 3] -> tanh * max.
        q_feat = q_sel.reshape(batch_size, num_frames, num_selected, dim)
        raw_off = self.offset_net(q_feat)
        raw_off = torch.tanh(raw_off)
        off_t = raw_off[..., 0] * self.time_offset_max  # [B, T, K]
        off_h = raw_off[..., 1] * self.space_offset_max  # [B, T, K]
        off_w = raw_off[..., 2] * self.space_offset_max  # [B, T, K]

        # 4. TRUE 2-D coordinates of the selected positions: the flat patch
        #    index is decomposed into (row, col) so offsets act on the real
        #    spatial axes (row-boundary adjacency is preserved).
        h0_idx = top_indices // grid_w  # [B, T, K] row
        w0_idx = top_indices % grid_w  # [B, T, K] column

        # 5. Offset anchor (centre of the sampling window) per query:
        #    (t + off_t, h + off_h, w + off_w).
        t_c = (
            torch.arange(num_frames, device=device, dtype=torch.float32)
            .view(1, num_frames, 1)
            + off_t
        )  # [B, T, K]
        h_c = h0_idx.float() + off_h  # [B, T, K]
        w_c = w0_idx.float() + off_w  # [B, T, K]

        # 6. Candidate template: window_size x window_size spatial offsets
        #    around the anchor -> S = window_size^2 candidates per query.
        offs = torch.arange(
            -self.window_half, self.window_half + 1, device=device
        ).float()
        dh, dw = torch.meshgrid(offs, offs, indexing="ij")
        cand_dh = dh.reshape(-1)  # [S]
        cand_dw = dw.reshape(-1)  # [S]
        h_src = (h_c.unsqueeze(-1) + cand_dh.view(1, 1, 1, -1)).clamp(
            0.0, float(grid_h - 1)
        )  # [B, T, K, S]
        w_src = (w_c.unsqueeze(-1) + cand_dw.view(1, 1, 1, -1)).clamp(
            0.0, float(grid_w - 1)
        )  # [B, T, K, S]
        t_src = t_c.unsqueeze(-1).expand(
            batch_size, num_frames, num_selected, num_candidates
        ).clamp(0.0, float(num_frames - 1))  # [B, T, K, S]

        # 7. 3-D bilinear split: time 2-way, row 2-way, column 2-way.
        t0 = t_src.floor().long()
        t1 = (t0 + 1).clamp(0, num_frames - 1)
        frac_t = (t_src - t_src.floor()).clamp(0.0, 1.0)
        h0 = h_src.floor().long()
        h1 = (h0 + 1).clamp(0, grid_h - 1)
        frac_h = (h_src - h_src.floor()).clamp(0.0, 1.0)
        w0 = w_src.floor().long()
        w1 = (w0 + 1).clamp(0, grid_w - 1)
        frac_w = (w_src - w_src.floor()).clamp(0.0, 1.0)

        # 8. 8-combination gather on the flattened [B, T*N, H, Hd] grid.
        k_flat = k.reshape(batch_size, num_frames * num_patches, heads, head_dim)
        v_flat = v.reshape(batch_size, num_frames * num_patches, heads, head_dim)

        def _bilinear_gather3d(
            flat: Tensor, idx_t: Tensor, idx_h: Tensor, idx_w: Tensor
        ) -> Tensor:
            flat_idx = idx_t * num_patches + idx_h * grid_w + idx_w
            flat_idx = flat_idx.unsqueeze(-1).unsqueeze(-1).expand(
                batch_size,
                num_frames,
                num_selected,
                num_candidates,
                heads,
                head_dim,
            )
            flat_idx = flat_idx.reshape(
                batch_size,
                num_frames * num_selected * num_candidates,
                heads,
                head_dim,
            )
            sampled = torch.gather(flat, 1, flat_idx)  # [B, T*K*S, H, Hd]
            return sampled.view(
                batch_size,
                num_frames,
                num_selected,
                num_candidates,
                heads,
                head_dim,
            )

        wt0 = (1.0 - frac_t).unsqueeze(-1).unsqueeze(-1)  # [B, T, K, S, 1, 1]
        wt1 = frac_t.unsqueeze(-1).unsqueeze(-1)
        wh0 = (1.0 - frac_h).unsqueeze(-1).unsqueeze(-1)
        wh1 = frac_h.unsqueeze(-1).unsqueeze(-1)
        ww0 = (1.0 - frac_w).unsqueeze(-1).unsqueeze(-1)
        ww1 = frac_w.unsqueeze(-1).unsqueeze(-1)

        k_sel = (
            wt0 * wh0 * ww0 * _bilinear_gather3d(k_flat, t0, h0, w0)
            + wt0 * wh0 * ww1 * _bilinear_gather3d(k_flat, t0, h0, w1)
            + wt0 * wh1 * ww0 * _bilinear_gather3d(k_flat, t0, h1, w0)
            + wt0 * wh1 * ww1 * _bilinear_gather3d(k_flat, t0, h1, w1)
            + wt1 * wh0 * ww0 * _bilinear_gather3d(k_flat, t1, h0, w0)
            + wt1 * wh0 * ww1 * _bilinear_gather3d(k_flat, t1, h0, w1)
            + wt1 * wh1 * ww0 * _bilinear_gather3d(k_flat, t1, h1, w0)
            + wt1 * wh1 * ww1 * _bilinear_gather3d(k_flat, t1, h1, w1)
        )  # [B, T, K, S, H, Hd]
        v_sel = (
            wt0 * wh0 * ww0 * _bilinear_gather3d(v_flat, t0, h0, w0)
            + wt0 * wh0 * ww1 * _bilinear_gather3d(v_flat, t0, h0, w1)
            + wt0 * wh1 * ww0 * _bilinear_gather3d(v_flat, t0, h1, w0)
            + wt0 * wh1 * ww1 * _bilinear_gather3d(v_flat, t0, h1, w1)
            + wt1 * wh0 * ww0 * _bilinear_gather3d(v_flat, t1, h0, w0)
            + wt1 * wh0 * ww1 * _bilinear_gather3d(v_flat, t1, h0, w1)
            + wt1 * wh1 * ww0 * _bilinear_gather3d(v_flat, t1, h1, w0)
            + wt1 * wh1 * ww1 * _bilinear_gather3d(v_flat, t1, h1, w1)
        )

        # 9. TRUE cross-frame attention: attn [B, T, H, K, S], softmax over
        #    the S candidate keys/values (the query is compared with every
        #    candidate of the sampled cross-frame neighbourhood).
        q_h = q_sel.permute(0, 1, 3, 2, 4).unsqueeze(-2)  # [B, T, H, K, 1, Hd]
        k_h = k_sel.permute(0, 1, 4, 2, 3, 5)  # [B, T, H, K, S, Hd]
        v_h = v_sel.permute(0, 1, 4, 2, 3, 5)  # [B, T, H, K, S, Hd]
        attn = torch.matmul(q_h, k_h.transpose(-1, -2)) / math.sqrt(head_dim)
        attn = attn.squeeze(-2)  # [B, T, H, K, S]
        attn = torch.softmax(attn, dim=-1)
        out = (attn.unsqueeze(-1) * v_h).sum(dim=-2)  # [B, T, H, K, Hd]
        out = out.permute(0, 1, 3, 2, 4).reshape(
            batch_size, num_frames, num_selected, dim
        )
        out = _fp32_linear(self.out_proj, out)

        # 10. Scatter back to the selected positions; unselected stay 0.
        delta = torch.zeros_like(x)
        delta.scatter_(
            2,
            top_indices.unsqueeze(-1).expand(
                batch_size, num_frames, num_selected, dim
            ),
            out,
        )
        return delta.to(dtype=in_dtype), off_t, off_h, off_w


class VitaCLIPDiscriminativeRAP(nn.Module):
    """Discriminative region selection + conv3d / sparse-ASA dual branch.

    Same forward skeleton as ``VitaCLIPDiscriminativeDual`` (scoring ->
    top-k -> entry seed -> bottleneck -> two branches -> gate fusion ->
    soft-enhanced write-back -> CLS soft aggregation), except the Mamba
    branch is replaced by the sparse RAP-style ASA branch of ``_ASABranch``.
    """

    def __init__(
        self,
        dim: int = 768,
        bottleneck: int = 192,
        num_spatial_patches: int = 196,
        top_k: int = 12,
        alpha: float = 0.4,
        use_conv3d: bool = True,
        use_asa: bool = True,
        asa_heads: int = 8,
        asa_time_offset_max: float = 1.0,
        asa_space_offset_max: float = 4.0,
        asa_window_size: int = 5,
        residual_scale_init: float = 0.01,
        seed_scale_init: float = 0.05,
        stability_clip: float = 100.0,
        min_frames: int = 4,
        eps: float = 1e-6,
        topk_temperature: float = 0.5,
        # --- anti-drift constraints (keep the learnable selection from
        # running away; see the module docstring for details).
        sig_threshold: Optional[float] = -0.5,
        softmax_tau_min: float = 0.25,
        soft_w_max: Optional[float] = 0.5,
        enhance_max: float = 0.5,
    ) -> None:
        super().__init__()

        if dim <= 0:
            raise ValueError("dim must be positive.")
        if bottleneck <= 0:
            raise ValueError("bottleneck must be positive.")
        if num_spatial_patches <= 0:
            raise ValueError("num_spatial_patches must be positive.")
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1].")
        if stability_clip <= 0:
            raise ValueError("stability_clip must be positive.")
        if min_frames < 2:
            raise ValueError("min_frames must be at least 2.")
        if topk_temperature <= 0:
            raise ValueError("topk_temperature must be positive.")
        if softmax_tau_min <= 0:
            raise ValueError("softmax_tau_min must be positive.")
        if soft_w_max is not None and soft_w_max <= 0:
            raise ValueError("soft_w_max must be positive (or None).")
        if enhance_max <= 0:
            raise ValueError("enhance_max must be positive.")

        self.dim = int(dim)
        self.bottleneck = int(bottleneck)
        self.num_spatial_patches = int(num_spatial_patches)
        self.top_k = int(top_k)
        self.alpha = float(alpha)
        self.use_conv3d = bool(use_conv3d)
        self.use_asa = bool(use_asa)
        self.stability_clip = float(stability_clip)
        self.min_frames = int(min_frames)
        self.eps = float(eps)
        self.topk_temperature = float(topk_temperature)
        self.sig_threshold = (
            float(sig_threshold) if sig_threshold is not None else None
        )
        self.softmax_tau_min = float(softmax_tau_min)
        self.soft_w_max = float(soft_w_max) if soft_w_max is not None else None
        self.enhance_max = float(enhance_max)

        # Shared bottleneck projections around both branches.
        self.down_projection = nn.Linear(dim, bottleneck)
        self.up_projection = nn.Linear(bottleneck, dim)

        # Fixed residual strength (same verified value as the RWKV module).
        self.register_buffer(
            "residual_scale",
            torch.tensor(float(residual_scale_init), dtype=torch.float32),
        )
        assert isinstance(self.residual_scale, Tensor)
        # Entry seed boost: how strongly the top-k action regions are
        # emphasised ONCE at the module entrance (initial guidance only).
        self.register_buffer(
            "seed_scale",
            torch.tensor(float(seed_scale_init), dtype=torch.float32),
        )
        assert isinstance(self.seed_scale, Tensor)
        # Learnable soft-enhance strength for the SELECTED positions
        # (zero-init -> identity enhancement at the start; clamped to
        # [0, enhance_max] in the forward so it can only amplify the
        # selected positions, never suppress them, and cannot run away).
        self.enhance_scale = nn.Parameter(
            torch.tensor(0.0, dtype=torch.float32)
        )

        # Learnable spatial scorer -> adaptive top-k selection.
        self.scoring = nn.Sequential(
            nn.Linear(dim, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, 1),
        )
        scoring_last = self.scoring[-1]
        assert isinstance(scoring_last, nn.Linear)
        nn.init.zeros_(scoring_last.weight)
        nn.init.zeros_(scoring_last.bias)

        # Two temporal branches (both processed in the bottleneck space).
        if self.use_conv3d:
            self.conv3d_branch = _Conv3DBranch(
                dim=bottleneck,
                num_spatial_patches=num_spatial_patches,
                eps=eps,
            )
        if self.use_asa:
            self.asa_branch = _ASABranch(
                dim=bottleneck,
                num_spatial_patches=num_spatial_patches,
                heads=asa_heads,
                time_offset_max=asa_time_offset_max,
                space_offset_max=asa_space_offset_max,
                window_size=asa_window_size,
                eps=eps,
            )

        # Patch-level gating between the two temporal branches
        # (2*bottleneck -> bottleneck -> 1, sigmoid; last layer zero-init
        # so sigma = 0.5 at init -> equal mixing at the start).
        self.gate = nn.Sequential(
            nn.Linear(2 * bottleneck, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, 1),
        )
        gate_last = self.gate[-1]
        assert isinstance(gate_last, nn.Linear)
        nn.init.zeros_(gate_last.weight)
        nn.init.zeros_(gate_last.bias)

        # Stability guard: sign-preserving numerical clip (see
        # ``_sanitize_real`` below; unlike ReLU it does not truncate
        # negative deltas, which would break the residual semantics).

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _sanitize_real(self, x: Tensor) -> Tensor:
        """Numerical guard that PRESERVES sign and residual semantics.

        Only removes NaN/Inf and clamps extreme outliers; unlike ReLU it
        does NOT truncate negative deltas, which would silently destroy the
        residual update (negative motion would be erased).
        """
        limit = self.stability_clip
        return torch.nan_to_num(
            x, nan=0.0, posinf=limit, neginf=-limit
        ).clamp(min=-limit, max=limit)

    def _zscore_normalize(self, t: Tensor) -> Tensor:
        """Normalise a [B, T, N] score map to zero mean / unit variance."""
        m = t.mean(dim=-1, keepdim=True)
        s = t.std(dim=-1, keepdim=True) + self.eps
        return (t - m) / s

    def _motion_strength(self, x: Tensor) -> Tensor:
        """Temporal motion strength per frame (symmetric averaging).

        x: [B, T, N, D] patch features -> [B, T, N] squared frame-diff map.
        """
        b, t, n, _ = x.shape
        motion = torch.zeros(b, t, n, device=x.device, dtype=x.dtype)
        if t >= 2:
            diff = (x[:, 1:] - x[:, :-1]).float().pow(2.0).mean(dim=-1)
            motion[:, 1:, :] += diff
            motion[:, :-1, :] += diff
            counts = torch.ones_like(motion)
            counts[:, 1:, :] += 1.0
            counts[:, :-1, :] += 1.0
            motion = motion / counts
        return motion

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        x: Tensor,
        attn_weights: Tensor,
        *,
        num_frames: int,
        patch_start: int = 1,
        return_information: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Dict[str, object]]]:
        """Apply the RAP sparse-and-correlated temporal module.

        Args:
            x: [B*T, N_tokens, D] full token stream (CLS tokens first) of
                the inserted CLIP layer.
            attn_weights: [B*T, N_p] CLS -> patch attention of frame t
                (already sliced to the CLS row; used for the selection
                scoring).
            num_frames: number of frames of the video clip (frames are
                batched into the leading dimension of x).
            patch_start: index where the patch tokens begin (x[:, :patch_start]
                are the CLS tokens; typically 1).
            return_information: additionally return an info dict for
                tensorboard diagnostics.

        Returns:
            Updated token stream [B*T, N_tokens, D] (or (out, info)).
        """
        if x.dim() != 3:
            raise ValueError(f"expected 3D [B*T, N, D], got {x.dim()}D input.")

        flat_batch, num_tokens, dim = x.shape
        if flat_batch < 2 or num_tokens < 2:
            return x
        if num_frames < self.min_frames:
            return x
        if flat_batch % num_frames != 0:
            return x

        batch_size = flat_batch // num_frames
        num_patches = num_tokens - patch_start
        info: Dict[str, object] = {}

        # --- 0. Separate CLS tokens from patch tokens.
        tokens = x.float().view(batch_size, num_frames, num_tokens, dim)
        cls_tokens = tokens[:, :, :patch_start, :]
        patch_tokens = tokens[:, :, patch_start:, :]

        # --- 1. Temporal discriminative score (STOP-inspired, adaptive
        # threshold):
        #      a) CLS -> patch attention A (real attention weights) of the
        #         frame t, taking only the last head (the head CLIP uses
        #         for the CLS aggregation);
        #      b) motion strength M: per-frame averaged squared difference
        #         between neighbouring frames;
        #      c) a learnable scorer over the patch features.
        #      The score W = alpha * A + (1 - alpha) * M + score(patch) is
        #      z-score normalised before the top-k selection.
        # (a) CLS -> patch attention, one value per patch per frame.
        attn_last_head = attn_weights.float().view(
            batch_size, num_frames, num_patches
        )

        # (b) temporal motion strength per frame (symmetric averaging).
        motion = self._motion_strength(patch_tokens)

        # (c) learnable spatial scorer over the patch features.
        score_map = self.scoring(patch_tokens).squeeze(-1)

        # Separate z-score normalisation of each term, then the weighted
        # combination W = alpha*A_n + (1-alpha)*M_n + score_n, z-scored
        # again before the adaptive top-k selection.
        attn_n = self._zscore_normalize(attn_last_head)
        motion_n = self._zscore_normalize(motion)
        score_n = self._zscore_normalize(score_map)
        w = self.alpha * attn_n + (1.0 - self.alpha) * motion_n + score_n
        w_norm = self._zscore_normalize(w)
        if return_information:
            info["score/mean"] = float(w.detach().float().mean().item())
            info["score/std"] = float(w.detach().float().std().item() + 1e-12)
            info["motion/mean"] = float(motion.detach().float().mean().item())

        # --- 2. Adaptive top-k selection per frame + anti-drift soft
        # weights (masked full-grid softmax):
        #   a) hard selection: top-k over w_norm, intersected with the
        #      significance threshold (z-score floor: scores below the
        #      threshold do NOT count, so a quiet frame cannot force a
        #      meaningless selection);
        #   b) soft weights: FULL-GRID softmax (gradient bridge for the
        #      CLS aggregation and the soft-enhance write-back - every
        #      position keeps a gradient through the softmax denominator,
        #      so the scorer can learn to push irrelevant positions down),
        #      then masked to the selected top-k support set (clean
        #      signal, no dilution from unselected positions);
        #   c) anti-drift guards: temperature floor (prevents one-hot
        #      sharpening -> weight collapse & vanishing gradient) and a
        #      per-position weight cap (prevents a single position from
        #      hijacking the aggregation / enhancement).
        num_selected = min(self.top_k, num_patches)
        _, top_indices = w_norm.topk(num_selected, dim=-1)  # [B, T, K]
        mask = torch.zeros_like(w_norm).bool()
        mask.scatter_(-1, top_indices, True)
        if self.sig_threshold is not None:
            mask = mask & (w_norm > self.sig_threshold)
        tau = max(self.topk_temperature, self.softmax_tau_min)
        soft_all = torch.softmax(w_norm / tau, dim=-1)  # [B, T, N]
        soft_weights = soft_all * mask.to(soft_all.dtype)
        if self.soft_w_max is not None:
            soft_weights = soft_weights.clamp(max=self.soft_w_max)
        if return_information:
            info["sel/topk_frac"] = float(num_selected / num_patches)
            p = soft_all.detach().clamp(min=1e-8)
            info["sel/entropy"] = float(-(p * p.log()).sum(-1).mean().item())
            info["sel/soft_max"] = float(soft_weights.detach().max().item())
            info["sel/n_selected"] = float(mask.detach().float().mean().item())

        # Union of selected positions across frames; if empty, skip the
        # module entirely (nothing to do, keep the original features).
        union = mask.float().sum(dim=1)  # [B, N]
        union_pos = (union > 0).nonzero()
        if union_pos.shape[0] == 0:
            return (x, info) if return_information else x

        # --- 3. Entry seed boost (initial guidance only): emphasise the
        # top-k action regions ONCE on the raw patch tokens.
        patch_tokens = patch_tokens + torch.tanh(
            cast(Tensor, self.seed_scale)
        ) * mask.to(patch_tokens.dtype).unsqueeze(-1)

        # --- 4. Bottleneck projection (down -> branches -> up).
        patch_down = _fp32_linear(self.down_projection, patch_tokens)

        # --- 5a. Conv3d branch: full-coverage local spatio-temporal
        # modelling over the complete 14x14 grid.
        conv_delta = None
        if self.use_conv3d:
            conv_delta = self.conv3d_branch(patch_down)

        # --- 5b. ASA branch (RAP): sparse true asynchronous cross-frame
        # attention over the selected top-k tokens only (2-D (off_h, off_w)
        # offsets + S=window_size^2 candidate neighbourhood). Unselected
        # positions contribute exactly 0 (sparse semantics).
        asa_delta = None
        asa_off_t = None
        asa_off_h = None
        asa_off_w = None
        if self.use_asa:
            asa_delta, asa_off_t, asa_off_h, asa_off_w = self.asa_branch(
                patch_down, top_indices=top_indices
            )

        # --- 6. Patch-level gating fusion of the two temporal deltas.
        # sigma = sigmoid(gate(GAP_T(conv) || GAP_T(asa))) -> 0.5 at init.
        if conv_delta is None and asa_delta is None:
            return (x, info) if return_information else x
        if conv_delta is None:
            fused = asa_delta
            sigma = torch.zeros(
                batch_size, num_frames, num_patches, 1, device=x.device
            )
        elif asa_delta is None:
            fused = conv_delta
            sigma = torch.ones(
                batch_size, num_frames, num_patches, 1, device=x.device
            )
        else:
            conv_stats = conv_delta.float().mean(dim=1)  # [B, N, D]
            asa_stats = asa_delta.float().mean(dim=1)  # [B, N, D]
            gate_in = torch.cat([conv_stats, asa_stats], dim=-1)
            sigma = self.gate(gate_in).unsqueeze(1)  # [B, 1, N, 1]
            sigma = torch.sigmoid(sigma)
            fused = sigma * conv_delta + (1.0 - sigma) * asa_delta

        # Stability guard: clip the fused delta into a sane range so a
        # single NaN/Inf cannot poison the whole stream (sign-preserving).
        assert fused is not None
        fused = self._sanitize_real(fused)
        if return_information:
            info["delta/conv_abs_mean"] = float(
                conv_delta.detach().float().abs().mean().item()
                if conv_delta is not None
                else 0.0
            )
            info["delta/asa_abs_mean"] = float(
                asa_delta.detach().float().abs().mean().item()
                if asa_delta is not None
                else 0.0
            )
            info["gate/sigma_mean"] = float(sigma.detach().float().mean().item())
            if (
                self.use_asa
                and asa_delta is not None
                and asa_off_t is not None
                and asa_off_h is not None
                and asa_off_w is not None
            ):
                info["asa/off_time_abs"] = float(
                    asa_off_t.detach().float().abs().mean().item()
                )
                info["asa/off_time_max"] = float(
                    asa_off_t.detach().float().abs().max().item()
                )
                info["asa/off_h_abs"] = float(
                    asa_off_h.detach().float().abs().mean().item()
                )
                info["asa/off_h_max"] = float(
                    asa_off_h.detach().float().abs().max().item()
                )
                info["asa/off_w_abs"] = float(
                    asa_off_w.detach().float().abs().mean().item()
                )
                info["asa/off_w_max"] = float(
                    asa_off_w.detach().float().abs().max().item()
                )

        # --- 7. Up-projection and SOFT-ENHANCED write-back: every patch
        # position receives its temporal delta (conv modelled the whole
        # grid, the ASA branch only the selected top-k -> unselected
        # positions get the conv-only contribution, matching RAP's sparse
        # semantics); the SELECTED positions additionally get a
        # confidence-proportional boost:
        #   enhance = 1 + clamp(enhance_scale, 0, enhance_max) * soft_w
        # (unselected: soft_w = 0 -> enhance = 1 -> unchanged; at init
        #  enhance_scale = 0 -> identity -> baseline behaviour).
        delta_all = _fp32_linear(self.up_projection, fused)
        delta_all = delta_all * torch.tanh(
            cast(Tensor, self.residual_scale).float()
        )
        gamma = torch.clamp(
            cast(Tensor, self.enhance_scale).float(),
            min=0.0,
            max=self.enhance_max,
        )
        enhance = 1.0 + gamma * soft_weights.unsqueeze(-1)  # [B, T, N, 1]
        patch_tokens = patch_tokens + delta_all * enhance
        patch_tokens = self._sanitize_real(patch_tokens)
        if return_information:
            info["enhance/scale"] = float(gamma.detach().item())

        # --- 8. CLS soft aggregation: each frame's CLS token gets the
        # soft-weighted sum of the selected deltas of that frame (gradient
        # bridge for the discriminator signal, since ln_post reads only
        # the CLS token).
        sel_w = soft_weights.unsqueeze(-1)  # [B, T, N, 1]
        cls_delta = (sel_w * delta_all).sum(dim=2)  # [B, T, D]
        cls_delta = self._sanitize_real(cls_delta)
        updated_cls = cls_tokens + cls_delta.unsqueeze(2)  # [B, T, 1, D]
        if return_information:
            info["cls/delta_abs_mean"] = float(
                cls_delta.detach().float().abs().mean().item()
            )
            info["info"] = 1.0

        # --- 9. Re-assemble the full token stream.
        out = torch.cat([updated_cls, patch_tokens], dim=2)  # [B, T, N_tok, D]
        out = out.reshape(flat_batch, num_tokens, dim).to(dtype=x.dtype)
        if return_information:
            return out, info
        return out

