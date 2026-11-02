"""TGL-Adapter: Trajectory-Guided Local Spatio-Temporal Adapter.

Idea (vs ECCV 2024 TAT / Trajectory-aligned Space-time Tokens)
-------------------------------------------------------------
TAT extracts trajectories with a frozen point tracker, gathers
trajectory-aligned tokens from DINO patch tokens, and classifies with a
Masked Space-Time Transformer over the trajectory tokens alone. TGL-Adapter
instead treats trajectories purely as a *spatio-temporal alignment prior* on
the CLIP/ViT image-encoder side: patch tokens are gathered along tracks,
modeled by a light factorized masked space-time transformer, and written back
to the original patch grid as a *small residual*:

    frames -> frozen CLIP visual encoder -> patch_tokens [B,T,N,C]
           -> TGLAdapter(patch_tokens, tracks) -> enhanced_tokens [B,T,N,C]
           -> temporal head / classifier

Hard contract (enforced by construction and by tests)
-----------------------------------------------------
* The tracker (CoTracker3 offline) is NEVER inside the training graph: tracks
  are produced by a separate offline preprocessing script
  (``tgl_tracker_offline.py``) and enter this module as constant tensors.
* Trajectory tokens are never used for classification and never replace the
  original patch tokens: the only output path is the residual
  ``enhanced = patch_tokens + scale * residual_grid``.
* Identity start: ``scale = max_scale * tanh(gamma)`` with ``gamma`` init 0,
  so at step 0 the module is EXACTLY the frozen backbone
  (``enhanced == patch_tokens`` bit-for-bit). ``max_scale`` (default 0.1)
  caps the residual. NOTE on the double-zero lock: initializing ``gamma``
  AND the ``Linear up`` both to exact zero makes the gradient of the whole
  adapter identically zero FOREVER (dL/dgamma is proportional to
  ``residual_grid`` which is 0 while ``up_proj`` is 0, and dL/d(up_proj) is
  proportional to ``scale`` which is 0 while ``gamma`` is 0 - the two zero
  factors lock each other). Identity start is already guaranteed by
  ``gamma == 0`` alone, so ``up_proj`` uses a tiny (std 1e-3) non-zero init
  by default: gamma receives a (small but non-zero) gradient at step 0 and
  wakes up first, then the full path (up/down/blocks) receives gradients
  proportional to ``scale`` from step 2 on - the standard ReZero-style
  wake-up, not a permanent lock. Pass ``up_proj_init="zero"`` to follow the
  literal zero-init contract, accepting a permanently frozen adapter.
* No global T*N attention: attention is factorized over (per-trajectory time)
  x (per-frame trajectories), i.e. O(K*T^2 + T*K^2).
* Pre-birth frames of later-born trajectories are masked out of attention and
  contribute zero scatter weight.

Coordinate mapping
------------------
``tracks`` are (x, y) in the *preprocessed image coordinate system*
([0, W] x [0, H], e.g. 224x224) - identical to what the offline script saved.
Grid coordinates are defined so that patch centers map to integers:

    gx = x * (W_p / W) - 0.5        # patch i center -> gx == i
    gy = y * (H_p / H) - 0.5

``mode="nearest"`` (default, TAT-aligned) rounds to the nearest patch.
``mode="bilinear"`` interpolates the 4-neighborhood with the same weights used
at scatter time, so gather/scatter are consistent. With
``local_tube_size=3`` (default) the gather pools a 3x3 patch neighborhood
around the continuous location with a small attention pooling
(content score + learnable distance bias) - a light improvement over TAT's
single-patch alignment.

Numerics: every Linear runs through ``_fp32_linear`` (project convention,
AMP-safe); all internal math is float32 and the output is cast back to the
input dtype. Dropout is disabled by ``eval()``; all masks are additive
(-1e9, never -inf) so fully-masked softmax rows stay NaN-free.
"""

from __future__ import annotations

import math
import warnings
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

__all__ = ["TGLAdapter"]

# Additive attention bias for masked positions. Not -inf: with -inf a fully
# masked softmax row yields NaN; -1e9 under fp32 softmax gives exact zeros for
# masked entries and a well-defined (uniform) - then explicitly re-zeroed -
# output for fully masked rows.
_NEG_BIAS = -1.0e9


def _fp32_linear(module: nn.Linear, x: Tensor) -> Tensor:
    """Execute Linear in float32 while preserving gradient flow."""
    weight = module.weight.float()
    bias = None if module.bias is None else module.bias.float()
    return F.linear(x.float(), weight, bias)


class _ZeroInitLinear(nn.Linear):
    """Linear whose output is exactly zero at initialization (zero weight+bias)."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__(in_features, out_features, bias=bias)
        nn.init.zeros_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class TGLTrajectoryReadout(nn.Module):
    """TAT-style trajectory readout: learnable-query attention pooling over
    the trajectory token stream, projected to the video feature dimension.

    Lesson from TAT (ECCV'24, few-shot AR): the trajectory token stream is a
    FIRST-CLASS citizen - a trained masked space-time transformer whose pooled
    representation directly feeds the classifier, so it receives strong
    gradients. Bolting trajectory information onto a frozen backbone as gated
    residuals (our v1) leaves the gate gradient-starved (measured: the gate
    opened 1.4% of its range in 8271 steps). This readout restores the strong
    path: pooled trajectory feature + a WARM fusion gate (initialized at
    ~0.3, not zero).

    Non-zero init throughout - this branch trains from step 1.
    """

    def __init__(self, adapter_dim: int, out_dim: int) -> None:
        super().__init__()
        self.adapter_dim = int(adapter_dim)
        self.query = nn.Parameter(torch.randn(1, self.adapter_dim) * 0.02)
        self.norm = nn.LayerNorm(self.adapter_dim)
        self.proj = nn.Linear(self.adapter_dim, int(out_dim))
        nn.init.normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)
        self.out_norm = nn.LayerNorm(int(out_dim))

    def forward(self, z: Tensor, valid: Tensor) -> Tensor:
        """z: [B, K, T, d] trajectory tokens; valid: [B, K, T] bool.

        Returns the pooled trajectory video feature [B, out_dim].
        """
        B = z.shape[0]
        zf = self.norm(z).reshape(B, -1, self.adapter_dim)   # [B, K*T, d]
        v = valid.reshape(B, -1)                             # [B, K*T]
        attn = (zf @ self.query.t().expand(B, -1, 1)).squeeze(-1)
        attn = attn / math.sqrt(self.adapter_dim)
        attn = torch.softmax(attn.masked_fill(~v, _NEG_BIAS), dim=-1)
        pooled = (attn.unsqueeze(-1) * zf).sum(1)            # [B, d]
        return self.out_norm(self.proj(pooled))


class _MaskedMHSA(nn.Module):
    """Shape-agnostic multi-head self-attention with an additive mask bias.

    Input x: [L, S, D]; mask_bias: [L, S, S] additive float (0 allow /
    ``_NEG_BIAS`` block). All fp32 (callers feed fp32 tensors).
    """

    def __init__(self, dim: int, num_heads: int, attn_drop: float = 0.0, out_drop: float = 0.0) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} not divisible by num_heads {num_heads}.")
        self.num_heads = int(num_heads)
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.attn_drop = nn.Dropout(float(attn_drop))
        self.out_drop = nn.Dropout(float(out_drop))

    def forward(self, x: Tensor, mask_bias: Tensor) -> Tensor:
        L, S, D = x.shape
        qkv = _fp32_linear(self.qkv, x).reshape(L, S, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # each [L, H, S, dh]
        attn = (q @ k.transpose(-2, -1)) * self.scale + mask_bias.unsqueeze(1)
        attn = torch.softmax(attn.float(), dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(L, S, D)
        return self.out_drop(_fp32_linear(self.proj, out))


class _FactorizedSTBlock(nn.Module):
    """One pre-LN factorized space-time block with hard validity masking.

    Temporal attention mixes frames *within* one trajectory; trajectory
    attention mixes trajectories *within* one frame. After every sublayer the
    hidden state is re-multiplied by the validity mask, so positions outside
    ``[birth_frame, T)`` (or with visibility below threshold) stay EXACTLY
    zero at every depth and can never leak into valid positions (columns are
    additionally masked in the attention itself).
    """

    def __init__(self, dim: int, num_heads: int, dropout: float, ffn_mult: float) -> None:
        super().__init__()
        self.norm_t = nn.LayerNorm(dim)
        self.attn_t = _MaskedMHSA(dim, num_heads, attn_drop=dropout, out_drop=dropout)
        self.norm_k = nn.LayerNorm(dim)
        self.attn_k = _MaskedMHSA(dim, num_heads, attn_drop=dropout, out_drop=dropout)
        self.norm_f = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, int(dim * ffn_mult))
        self.fc2 = nn.Linear(int(dim * ffn_mult), dim)
        self.drop = nn.Dropout(float(dropout))

    def _ffn(self, x: Tensor) -> Tensor:
        return _fp32_linear(self.fc2, F.gelu(_fp32_linear(self.fc1, x)))

    def forward(self, x: Tensor, bias_t: Tensor, bias_k: Tensor, valid: Tensor) -> Tensor:
        # x: [B, K, T, d]; valid: [B, K, T] bool.
        B, K, T, _ = x.shape
        # --- temporal attention (per trajectory along T) ---
        h = self.norm_t(x).reshape(B * K, T, -1)
        x = x + self.drop(
            self.attn_t(h, bias_t).reshape(B, K, T, -1)
        )
        x = x * valid.unsqueeze(-1)
        # --- trajectory attention (per frame along K) ---
        xt = x.permute(0, 2, 1, 3).reshape(B * T, K, -1)
        xt = xt + self.drop(self.attn_k(xt, bias_k))
        x = xt.reshape(B, T, K, -1).permute(0, 2, 1, 3)
        x = x * valid.unsqueeze(-1)
        # --- FFN ---
        x = x + self.drop(self._ffn(self.norm_f(x)))
        x = x * valid.unsqueeze(-1)
        return x


class _LocalTubePool(nn.Module):
    """Attention pooling over a (2r+1)x(2r+1) patch neighborhood.

    For a trajectory point at continuous grid coords (gx, gy), candidates are
    the patches whose centers lie within the (2r+1)^2 window around
    ``round(g)`` (clamped to the grid; duplicated border candidates are
    harmless under softmax). Pooling score = content score (linear head on the
    candidate token, scaled by 1/sqrt(C)) - softplus(temp) * dist^2, where
    dist is the distance from (gx, gy) to the candidate patch center. The
    learnable positive ``temp`` keeps the distance bias meaningful; with a
    large temp this degenerates gracefully toward nearest-patch alignment.
    """

    def __init__(self, dim: int, tube_size: int = 3) -> None:
        super().__init__()
        if tube_size < 1 or tube_size % 2 == 0:
            raise ValueError("tube_size must be a positive odd integer.")
        self.tube_size = int(tube_size)
        self.radius = (self.tube_size - 1) // 2
        self.score = nn.Linear(dim, 1, bias=False)
        self.dist_temp = nn.Parameter(torch.tensor(1.0))

    def forward(self, grid: Tensor, gx: Tensor, gy: Tensor) -> Tensor:
        """grid: [B, T, Hp, Wp, C]; gx/gy: [B, K, T] continuous grid coords.

        Returns trajectory tokens [B, K, T, C].
        """
        B, T, Hp, Wp, C = grid.shape
        K = gx.shape[1]
        dev = grid.device
        ar = torch.arange(-self.radius, self.radius + 1, device=dev)  # [r]
        # Candidate integer patch indices: [B, K, T, M] with M = tube^2.
        base_x = torch.round(gx).long().clamp(0, Wp - 1)
        base_y = torch.round(gy).long().clamp(0, Hp - 1)
        off_y, off_x = torch.meshgrid(ar, ar, indexing="ij")
        off_y = off_y.reshape(-1)  # [M]
        off_x = off_x.reshape(-1)  # [M]
        cx = (base_x.unsqueeze(-1) + off_x).clamp(0, Wp - 1)  # [B,K,T,M]
        cy = (base_y.unsqueeze(-1) + off_y).clamp(0, Hp - 1)  # [B,K,T,M]

        # Gather candidate tokens: [B, T, K*M, C].
        flat_idx = (cy * Wp + cx).permute(0, 2, 1, 3).reshape(B, T, -1)  # [B,T,K*M]
        cand = torch.gather(
            grid.reshape(B, T, Hp * Wp, C),
            2,
            flat_idx.unsqueeze(-1).expand(-1, -1, -1, C),
        )  # [B, T, K*M, C]
        M = off_x.numel()
        cand = cand.reshape(B, T, K, M, C).permute(0, 2, 1, 3, 4)  # [B, K, T, M, C]

        # Distance from the continuous point to candidate centers, in grid units.
        gx_e = gx.unsqueeze(-1)  # [B,K,T,1]
        gy_e = gy.unsqueeze(-1)
        dist2 = (gx_e - cx.float()) ** 2 + (gy_e - cy.float()) ** 2  # [B,K,T,M]
        content = _fp32_linear(self.score, cand).squeeze(-1) / math.sqrt(C)  # [B,K,T,M]
        bias = -F.softplus(self.dist_temp.float()) * dist2
        w = torch.softmax(content + bias, dim=-1)  # [B,K,T,M]
        return (w.unsqueeze(-1) * cand).sum(-2)  # weighted sum over candidates M -> [B,K,T,C]


class TGLAdapter(nn.Module):
    """Trajectory-Guided Local Spatio-Temporal Adapter.

    Args:
        dim: patch token channel C (e.g. 768 for ViT-B/16).
        adapter_dim: internal bottleneck d (default C // 4).
        num_layers: number of factorized space-time blocks (default 2).
        num_heads: attention heads per block (default 4).
        dropout: dropout in attention/FFN (default 0.1).
        ffn_mult: FFN hidden = adapter_dim * ffn_mult (default 2, light).
        gather_mode: "nearest" (TAT-aligned, default) or "bilinear" (4-neighbor
            interpolation; used for BOTH gather and scatter weights).
        local_tube_size: 1 disables tube pooling (pure per-mode gather);
            3 (default) pools a 3x3 neighborhood with `_LocalTubePool`.
        visibility_threshold: valid = visibility > threshold AND t >= birth.
        max_scale: cap of the residual scale; scale = max_scale * tanh(gamma),
            gamma init 0 -> identity start.
        up_proj_init: "tiny" (default; std ``up_proj_init_std`` - required for
            the adapter to ever receive gradients, see docstring note on the
            double-zero lock) or "zero" (literal zero-init; PERMANENTLY locks
            the adapter - a loud warning is emitted).
        up_proj_init_std: std of the tiny up-projection init (default 1e-3).
    """

    def __init__(
        self,
        dim: int,
        adapter_dim: Optional[int] = None,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        ffn_mult: float = 2.0,
        gather_mode: str = "nearest",
        local_tube_size: int = 3,
        visibility_threshold: float = 0.5,
        max_scale: float = 0.1,
        up_proj_init: str = "tiny",
        up_proj_init_std: float = 1.0e-3,
        gate_init: float = 0.0,
    ) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive.")
        if gather_mode not in ("nearest", "bilinear"):
            raise ValueError(f"gather_mode must be 'nearest'|'bilinear', got {gather_mode!r}.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if max_scale <= 0.0:
            raise ValueError("max_scale must be positive.")

        self.dim = int(dim)
        self.adapter_dim = int(adapter_dim) if adapter_dim is not None else self.dim // 4
        if self.adapter_dim <= 0 or self.adapter_dim % num_heads != 0:
            raise ValueError(
                f"adapter_dim {self.adapter_dim} must be positive and divisible by num_heads {num_heads}."
            )
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.gather_mode = str(gather_mode)
        self.local_tube_size = int(local_tube_size)
        self.visibility_threshold = float(visibility_threshold)
        self.max_scale = float(max_scale)

        # --- gather / local pooling ---
        self.tube_pool = (
            _LocalTubePool(self.dim, self.local_tube_size)
            if self.local_tube_size > 1
            else None
        )

        # --- adapter core: LN -> down(C->d) [+ motion emb] -> blocks -> up(d->C, zero) ---
        self.norm_in = nn.LayerNorm(self.dim)
        self.down_proj = nn.Linear(self.dim, self.adapter_dim)

        # Motion embedding: 6-dim motion feature -> 2-layer MLP -> d, added to Z.
        # Feature = [x_norm, y_norm, dx, dy, speed, visibility] (coords normalized
        # to [0,1]; deltas/speed in normalized units; zeroed at invalid positions).
        self.motion_mlp = nn.Sequential(
            nn.Linear(6, self.adapter_dim),
            nn.GELU(),
            nn.Linear(self.adapter_dim, self.adapter_dim),
        )

        self.blocks = nn.ModuleList(
            [
                _FactorizedSTBlock(self.adapter_dim, self.num_heads, dropout, ffn_mult)
                for _ in range(self.num_layers)
            ]
        )
        # Linear up: d -> C. Default is a TINY non-zero init (std 1e-3): with
        # gamma == 0 the identity start is EXACT regardless of up_proj, and the
        # tiny init breaks the double-zero gradient lock (see class docstring).
        # "zero" keeps the literal contract but permanently freezes the module.
        if up_proj_init not in ("tiny", "zero"):
            raise ValueError(f"up_proj_init must be 'tiny'|'zero', got {up_proj_init!r}.")
        self.up_proj = nn.Linear(self.adapter_dim, self.dim)
        if up_proj_init == "zero":
            warnings.warn(
                "TGLAdapter(up_proj_init='zero'): with gamma also zero-init this "
                "permanently locks the adapter (dL/dgamma == 0 because "
                "residual_grid == 0, and dL/d(up_proj) == 0 because scale == 0). "
                "Use up_proj_init='tiny' unless you know what you are doing.",
                stacklevel=2,
            )
            nn.init.zeros_(self.up_proj.weight)
            nn.init.zeros_(self.up_proj.bias)
        else:
            nn.init.normal_(self.up_proj.weight, std=float(up_proj_init_std))
            nn.init.zeros_(self.up_proj.bias)
        self.up_proj_init = str(up_proj_init)

        # Residual gate: scale = max_scale * tanh(gamma), gamma init 0.
        # tanh keeps |scale| <= max_scale; the residual can never blow up.
        # Residual gate: scale = max_scale * tanh(gamma). gate_init (in SCALE
        # units) sets the starting gate opening: 0 = strict identity start
        # (legacy; in practice the zero-gate cold-start starves the whole
        # adapter of gradients - gamma is the only live parameter and its
        # signal is noise-dominated, measured: gamma crawled to 1.4% of its
        # range in 8271 steps); a warm value like 0.04 opens ~40% of the cap
        # immediately so the full adapter trains from step 1 (the
        # norm-conserving write-back bounds the perturbation at max_scale).
        gate_init_clamped = max(-0.999 * self.max_scale, min(0.999 * self.max_scale, float(gate_init)))
        gamma_init = math.atanh(gate_init_clamped / self.max_scale) if self.max_scale > 0 else 0.0
        self.gamma = nn.Parameter(torch.full((1,), float(gamma_init)))

    # ------------------------------------------------------------------ #
    #  Helpers                                                            #
    # ------------------------------------------------------------------ #
    def _grid_coords(self, tracks: Tensor, patch_hw: Tuple[int, int], image_size: Tuple[int, int]) -> Tuple[Tensor, Tensor]:
        """Image coords (x in [0,W], y in [0,H]) -> continuous patch grid coords.

        Returns (gx, gy) each [B, K, T] in [0, W_p-1] x [0, H_p-1] (clamped).
        """
        H_p, W_p = int(patch_hw[0]), int(patch_hw[1])
        H, W = int(image_size[0]), int(image_size[1])
        gx = tracks[..., 0].float() * (W_p / float(W)) - 0.5
        gy = tracks[..., 1].float() * (H_p / float(H)) - 0.5
        return gx.clamp(0.0, W_p - 1.0), gy.clamp(0.0, H_p - 1.0)

    def _nearest_indices(self, gx: Tensor, gy: Tensor, patch_hw: Tuple[int, int]) -> Tensor:
        H_p, W_p = int(patch_hw[0]), int(patch_hw[1])
        ix = torch.round(gx).long().clamp(0, W_p - 1)  # [B,K,T]
        iy = torch.round(gy).long().clamp(0, H_p - 1)
        return iy * W_p + ix  # [B,K,T] flat patch index

    def _gather_tokens(
        self,
        grid: Tensor,
        gx: Tensor,
        gy: Tensor,
        patch_hw: Tuple[int, int],
    ) -> Tensor:
        """Gather per-trajectory tokens. Returns [B, K, T, C]."""
        B, T, H_p, W_p, C = grid.shape
        if self.tube_pool is not None:
            # Optional 3x3 (or larger) local attention pooling around the
            # continuous location - the light improvement over TAT's
            # single-patch alignment.
            return self.tube_pool(grid, gx, gy)

        flat = grid.reshape(B, T, H_p * W_p, C)
        if self.gather_mode == "nearest":
            idx = self._nearest_indices(gx, gy, patch_hw)  # [B,K,T]
            idx = idx.permute(0, 2, 1)  # [B,T,K]
            out = torch.gather(flat, 2, idx.unsqueeze(-1).expand(-1, -1, -1, C))
            return out.permute(0, 2, 1, 3)  # [B,K,T,C]

        # bilinear: 4-neighbor interpolation with grid_sample(align_corners=False)
        # semantics after clamping to the valid range.
        H_p_i, W_p_i = int(patch_hw[0]), int(patch_hw[1])
        x0 = torch.floor(gx).long().clamp(0, W_p_i - 1)
        y0 = torch.floor(gy).long().clamp(0, H_p_i - 1)
        x1 = (x0 + 1).clamp(0, W_p_i - 1)
        y1 = (y0 + 1).clamp(0, H_p_i - 1)
        wx1 = (gx - x0.float()).clamp(0.0, 1.0)
        wy1 = (gy - y0.float()).clamp(0.0, 1.0)
        wx0, wy0 = 1.0 - wx1, 1.0 - wy1

        def at(iy: Tensor, ix: Tensor) -> Tensor:  # [B,K,T] -> [B,K,T,C]
            idx = (iy * W_p_i + ix).permute(0, 2, 1)
            g = torch.gather(flat, 2, idx.unsqueeze(-1).expand(-1, -1, -1, C))
            return g.permute(0, 2, 1, 3)

        return (
            at(y0, x0) * (wy0 * wx0).unsqueeze(-1)
            + at(y0, x1) * (wy0 * wx1).unsqueeze(-1)
            + at(y1, x0) * (wy1 * wx0).unsqueeze(-1)
            + at(y1, x1) * (wy1 * wx1).unsqueeze(-1)
        )

    def _motion_features(
        self,
        tracks: Tensor,
        visibility: Tensor,
        valid: Tensor,
        image_size: Tuple[int, int],
    ) -> Tensor:
        """[x_norm, y_norm, dx, dy, speed, visibility] -> [B, K, T, 6].

        dx/dy/speed are zero at t == 0 and at invalid positions, so no
        pre-birth motion leaks into the adapter (on top of attention masking).
        """
        H, W = int(image_size[0]), int(image_size[1])
        u = (tracks[..., 0].float() / float(W)).clamp(0.0, 1.0)  # [B,K,T]
        v = (tracks[..., 1].float() / float(H)).clamp(0.0, 1.0)
        du = torch.zeros_like(u)
        dv = torch.zeros_like(v)
        du[..., 1:] = u[..., 1:] - u[..., :-1]
        dv[..., 1:] = v[..., 1:] - v[..., :-1]
        speed = torch.sqrt(du * du + dv * dv)
        feat = torch.stack([u, v, du, dv, speed, visibility.float()], dim=-1)  # [B,K,T,6]
        return feat * valid.unsqueeze(-1)

    @staticmethod
    def _scatter_weighted(
        acc_r: Tensor, acc_w: Tensor, idx: Tensor, w: Tensor, r: Tensor
    ) -> None:
        """acc_r[idx] += w * r ; acc_w[idx] += w  (in-place, per B*T rows).

        acc_r: [BT, N, C]; acc_w: [BT, N]; idx/w: [BT, K]; r: [BT, K, C].
        """
        contrib = w.unsqueeze(-1) * r
        acc_r.scatter_add_(1, idx.unsqueeze(-1).expand(-1, -1, r.shape[-1]), contrib)
        acc_w.scatter_add_(1, idx, w)

    def _scatter(
        self,
        residual: Tensor,
        weight: Tensor,
        gx: Tensor,
        gy: Tensor,
        patch_hw: Tuple[int, int],
    ) -> Tuple[Tensor, Tensor]:
        """Scatter trajectory residuals back to the patch grid.

        residual: [B,K,T,C]; weight: [B,K,T] (visibility * valid gate).
        Multiple trajectories landing on the same patch are combined as a
        WEIGHTED AVERAGE (sum w*r / sum w), never an overwrite. Patches with
        zero total weight get exactly zero residual.

        Returns (residual_grid [B,T,N,C], weight_map [B,T,N]).
        """
        B, K, T, C = residual.shape
        H_p, W_p = int(patch_hw[0]), int(patch_hw[1])
        N = H_p * W_p

        r_rows = residual.permute(0, 2, 1, 3).reshape(B * T, K, C)
        w_rows = weight.permute(0, 2, 1).reshape(B * T, K)
        acc_r = r_rows.new_zeros(B * T, N, C)
        acc_w = w_rows.new_zeros(B * T, N)

        if self.gather_mode == "nearest":
            idx = self._nearest_indices(gx, gy, patch_hw).permute(0, 2, 1).reshape(B * T, K)
            self._scatter_weighted(acc_r, acc_w, idx, w_rows, r_rows)
        else:
            H_p_i, W_p_i = H_p, W_p
            x0 = torch.floor(gx).long().clamp(0, W_p_i - 1)
            y0 = torch.floor(gy).long().clamp(0, H_p_i - 1)
            x1 = (x0 + 1).clamp(0, W_p_i - 1)
            y1 = (y0 + 1).clamp(0, H_p_i - 1)
            wx1 = (gx - x0.float()).clamp(0.0, 1.0)
            wy1 = (gy - y0.float()).clamp(0.0, 1.0)
            wx0, wy0 = 1.0 - wx1, 1.0 - wy1
            for iy, ix, wy, wx in (
                (y0, x0, wy0, wx0),
                (y0, x1, wy0, wx1),
                (y1, x0, wy1, wx0),
                (y1, x1, wy1, wx1),
            ):
                idx = (iy * W_p_i + ix).permute(0, 2, 1).reshape(B * T, K)
                self._scatter_weighted(acc_r, acc_w, idx, w_rows * (wy * wx).permute(0, 2, 1).reshape(B * T, K), r_rows)

        weight_map = acc_w.reshape(B, T, N)
        denom = acc_w.clamp_min(1.0e-8).unsqueeze(-1)
        residual_grid = torch.where(
            acc_w.unsqueeze(-1) > 0, acc_r / denom, torch.zeros_like(acc_r)
        ).reshape(B, T, N, C)
        return residual_grid, weight_map

    # ------------------------------------------------------------------ #
    #  Forward                                                            #
    # ------------------------------------------------------------------ #
    def forward(
        self,
        patch_tokens: Tensor,
        tracks: Tensor,
        visibility: Tensor,
        birth_frame: Tensor,
        patch_hw: Tuple[int, int],
        image_size: Tuple[int, int],
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Enhance patch tokens with a trajectory-guided residual.

        Args:
            patch_tokens: [B, T, N, C], N = H_p * W_p (CLS token excluded).
            tracks: [B, K, T, 2] image coords (x, y) in the preprocessed
                coordinate system of ``image_size``; constant (offline).
            visibility: [B, K, T] in [0, 1]; constant (offline).
            birth_frame: [B, K] long; trajectory exists only for t >= birth.
            patch_hw: (H_p, W_p), e.g. (14, 14).
            image_size: (H, W), e.g. (224, 224).

        Returns:
            enhanced_tokens: [B, T, N, C] = patch_tokens + scale * residual_grid.
            aux: {"trajectory_tokens": [B,K,T,d] post-adapter token stream,
                  "residual_tokens": [B,K,T,C] up-projected residual per track,
                  "scatter_weight": [B,T,N] total scatter weight per patch (0
                    where no trajectory lands -> zero residual there),
                  "adapter_scale": scalar tensor max_scale * tanh(gamma),
                  "valid_mask": [B,K,T] bool attention/validity mask}.
        """
        if patch_tokens.dim() != 4:
            raise ValueError(f"patch_tokens must be [B,T,N,C], got {tuple(patch_tokens.shape)}.")
        B, T, N, C = patch_tokens.shape
        H_p, W_p = int(patch_hw[0]), int(patch_hw[1])
        if N != H_p * W_p:
            raise ValueError(f"N={N} != patch_hw product {H_p * W_p}.")
        if C != self.dim:
            raise ValueError(f"C={C} != module dim {self.dim}.")
        if tracks.dim() != 4 or tracks.shape[-1] != 2:
            raise ValueError(f"tracks must be [B,K,T,2], got {tuple(tracks.shape)}.")
        if visibility.shape != tracks.shape[:3] or birth_frame.shape != tracks.shape[:2]:
            raise ValueError(
                f"visibility {tuple(visibility.shape)} / birth_frame {tuple(birth_frame.shape)} "
                f"must match tracks {tuple(tracks.shape[:3])}."
            )
        if visibility.shape[0] != B or visibility.shape[2] != T:
            raise ValueError("Batch or frame mismatch between patch_tokens and tracks.")

        in_dtype = patch_tokens.dtype
        pt = patch_tokens.float()
        tracks = tracks.detach().float()
        visibility = visibility.detach().float()

        # --- validity mask: visible AND born (test contract #7) ---
        t_idx = torch.arange(T, device=pt.device).view(1, 1, T)
        valid = (visibility > self.visibility_threshold) & (
            t_idx >= birth_frame.long().view(B, -1, 1)
        )  # [B,K,T] bool
        vf = valid.float()

        # --- attention bias masks (additive, fp32) ---
        K = tracks.shape[1]
        allow_t = vf.unsqueeze(-1) * vf.unsqueeze(-2)  # [B,K,T,T] (q,t)*(q,k)
        bias_t = torch.where(allow_t > 0, torch.zeros_like(allow_t), torch.full_like(allow_t, _NEG_BIAS))
        bias_t = bias_t.reshape(B * K, T, T)
        v_kt = vf.permute(0, 2, 1)  # [B,T,K]
        allow_k = v_kt.unsqueeze(-1) * v_kt.unsqueeze(-2)  # [B,T,K,K]
        bias_k = torch.where(allow_k > 0, torch.zeros_like(allow_k), torch.full_like(allow_k, _NEG_BIAS))
        bias_k = bias_k.reshape(B * T, -1, tracks.shape[1])

        # --- gather trajectory tokens ---
        grid = pt.reshape(B, T, H_p, W_p, C)
        gx, gy = self._grid_coords(tracks, patch_hw, image_size)
        z = self._gather_tokens(grid, gx, gy, patch_hw)  # [B,K,T,C]

        # --- LN -> down -> + motion embedding ---
        # motion = Linear2(GELU(Linear1([x, y, dx, dy, speed, vis]))) added to Z.
        z = _fp32_linear(self.down_proj, self.norm_in(z))  # [B,K,T,d]
        m_feat = self._motion_features(tracks, visibility, valid, image_size)  # [B,K,T,6]
        m_hidden = F.gelu(_fp32_linear(self.motion_mlp[0], m_feat))
        z = z + _fp32_linear(self.motion_mlp[2], m_hidden)
        z = z * vf.unsqueeze(-1)  # hard-zero invalid positions from the start

        # --- factorized masked space-time transformer ---
        for blk in self.blocks:
            z = blk(z, bias_t, bias_k, valid)

        trajectory_tokens = z  # [B,K,T,d]
        residual_tokens = _fp32_linear(self.up_proj, z)  # [B,K,T,C], zero at init

        # --- scatter: weighted average into the patch grid ---
        weight = vf * visibility  # visibility * (born & above-threshold) gate
        residual_grid, weight_map = self._scatter(residual_tokens, weight, gx, gy, patch_hw)

        # --- bounded, NORM-CONSERVING residual write-back ---
        # The residual grid is rescaled per patch so that its norm equals
        # `scale * ||patch||`: the adapter learns the DIRECTION of the
        # perturbation, while `scale = max_scale * tanh(gamma)` directly means
        # "the fraction of the patch energy the trajectory residual may
        # perturb" (<= max_scale). Without this normalization the effective
        # residual was the product of three small factors (up_proj tiny-init
        # x max_scale x the patch-norm growth across depth, measured at
        # 0.08-0.4% of the patch norm even with the gate fully open) - the
        # adapter was a mathematical no-op. Patches with no tracks stay
        # exactly zero.
        adapter_scale = torch.tanh(self.gamma.float()) * self.max_scale
        r_norm = residual_grid.norm(dim=-1, keepdim=True)  # [B,T,N,1]
        r_dir = residual_grid / r_norm.clamp_min(1e-8)
        pt_norm = pt.norm(dim=-1, keepdim=True)            # [B,T,N,1]
        delta = torch.where(
            r_norm > 1e-8,
            adapter_scale * pt_norm * r_dir,
            torch.zeros_like(residual_grid),
        )
        enhanced = pt + delta

        aux: Dict[str, Tensor] = {
            "trajectory_tokens": trajectory_tokens,
            "residual_tokens": residual_tokens,
            "residual_grid": residual_grid,
            "scatter_weight": weight_map,
            "adapter_scale": adapter_scale,
            "valid_mask": valid,
        }
        return enhanced.to(in_dtype), aux
