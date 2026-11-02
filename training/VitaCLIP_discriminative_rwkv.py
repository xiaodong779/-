"""Discriminative region + RWKV temporal module for Vita-CLIP.

Combines the discriminative-region selection idea from STOP (CVPR 2025,
"Integrated Spatial-Temporal Dynamic Prompting for Video Understanding")
with a lightweight RWKV temporal model:

1. Discriminative region selection (same spirit as STOP):
   - A_t = CLS -> patch attention of frame t (real attention weights);
   - M_t = temporal motion strength (frame-difference squared mean);
   - W_t = alpha * A_t + (1 - alpha) * M_t;
   - keep the top-k patches per frame (k = top_k, default 6, alpha 0.4).

2. RWKV temporal modelling:
   - collect the union of selected patch positions across frames;
   - for every selected position extract the cross-frame sequence [B, T, D];
   - feed the batched sequences [B*|U|, T, D] into a shared-weight lightweight
     RWKV (time-mixing WKV + channel-mixing, RWKV-4 style core ops);
   - add the RWKV output back to the selected patch tokens with a residual
     gate, and write the mean delta of the selected positions back to the
     CLS token (guarantees gradient flow at the last transformer block).

Unlike STOP, no prompter tokens are inserted and no cross-frame MSA is
required: the CLIP transformer keeps processing every frame independently,
which preserves OpenAI CLIP alignment.

Input / output shape: [B*T, N, D] (same as the frequency adapter).

The module is not tied to exactly eight frames, so later 8/16-frame
ablation experiments can reuse the same implementation.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _fp32_layer_norm(module: nn.LayerNorm, x: Tensor) -> Tensor:
    """Execute LayerNorm in float32 (valid even after model.half())."""
    weight = None if module.weight is None else module.weight.float()
    bias = None if module.bias is None else module.bias.float()
    return F.layer_norm(x.float(), module.normalized_shape, weight, bias, module.eps)


def _fp32_linear(module: nn.Linear, x: Tensor) -> Tensor:
    """Execute Linear in float32 while preserving gradient flow."""
    weight = module.weight.float()
    bias = None if module.bias is None else module.bias.float()
    return F.linear(x.float(), weight, bias)


def _time_shift(x: Tensor) -> Tensor:
    """Shift a [B, T, D] sequence one step along time (RWKV token shift).

    The first time step has no history and is padded with zeros.
    """
    return F.pad(x, (0, 0, 1, 0))[:, :-1]


class _RWKVBlock(nn.Module):
    """
    Lightweight RWKV-4 style block operating on [B, T, D].

    Residual operations (RWKV-4 core):

    time-mixing (WKV with exponential decay over the time dimension):
        xk/xv/xr = x * mix + time_shift(x) * (1 - mix)
        k = key(xk); v = value(xv); r = sigmoid(receptance(xr))
        out_t = r_t * (a_t + u * k_t * v_t) / (b_t + u * k_t + eps)
        with  a_t = w * a_{t-1} + k_t * v_t
              b_t = w * b_{t-1} + k_t
              w   = exp(-exp(time_decay))  (learnable, in (0, 1))

    channel-mixing:
        xk/xr = x * mix + time_shift(x) * (1 - mix)
        k = square(relu(key(xk))); out = sigmoid(receptance(xr)) * value(k)

    All operations are performed in float32 for numerical stability.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive.")

        self.dim = int(dim)
        self.eps = float(eps)

        # Pre/within-block layer norms.
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)

        # --- time-mixing mix coefficients (RWKV-4 style) ---
        self.time_mix_k = nn.Parameter(torch.ones(dim))
        self.time_mix_v = nn.Parameter(torch.ones(dim))
        self.time_mix_r = nn.Parameter(torch.ones(dim))
        # --- channel-mixing mix coefficients ---
        self.time_mix_gk = nn.Parameter(torch.ones(dim))
        self.time_mix_gr = nn.Parameter(torch.ones(dim))

        # --- time-mixing projections ---
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.receptance = nn.Linear(dim, dim, bias=False)
        # WKV bonus (RWKV-4 "u" vector), positive and learnable.
        self.wkv_bonus = nn.Parameter(torch.ones(dim))
        # Log-space decay: w = exp(-exp(time_decay)) in (0, 1).
        self.time_decay = nn.Parameter(torch.zeros(dim))

        # --- channel-mixing projections ---
        self.gate_key = nn.Linear(dim, dim, bias=False)
        self.gate_value = nn.Linear(dim, dim, bias=False)
        self.gate_receptance = nn.Linear(dim, dim, bias=False)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Keep the block close to identity while preserving gradient flow.

        Linear projections use a moderate std (same trick as the frequency
        adapter): tiny weights freeze the gradient, 0.05 keeps the update
        small yet well-conditioned.
        """
        for m in (
            self.key,
            self.value,
            self.receptance,
            self.gate_key,
            self.gate_value,
            self.gate_receptance,
        ):
            nn.init.normal_(m.weight, mean=0.0, std=0.05)

        # mix coefficients start at 1 -> full current-frame reliance,
        # the time-mixing branch becomes a per-frame MLP at init, which is
        # stable and still lets the network learn temporal mixing.
        nn.init.ones_(self.time_mix_k)
        nn.init.ones_(self.time_mix_v)
        nn.init.ones_(self.time_mix_r)
        nn.init.ones_(self.time_mix_gk)
        nn.init.ones_(self.time_mix_gr)

        # decay logit 0 -> w = exp(-exp(0)) = exp(-1) ~ 0.37: a moderate
        # exponential forgetting rate over the temporal axis.
        nn.init.zeros_(self.time_decay)
        nn.init.ones_(self.wkv_bonus)

    def _time_mixing(self, x: Tensor) -> Tensor:
        """RWKV-4 time-mixing (WKV) on [B, T, D]. x is assumed pre-LN."""
        shifted = _time_shift(x)

        xk = x * self.time_mix_k.float() + shifted * (1.0 - self.time_mix_k.float())
        xv = x * self.time_mix_v.float() + shifted * (1.0 - self.time_mix_v.float())
        xr = x * self.time_mix_r.float() + shifted * (1.0 - self.time_mix_r.float())

        k = _fp32_linear(self.key, xk)
        v = _fp32_linear(self.value, xv)
        r = torch.sigmoid(_fp32_linear(self.receptance, xr))

        w = torch.exp(-torch.exp(self.time_decay.float()))
        u = torch.relu(self.wkv_bonus.float()) + self.eps

        # Linear-time recurrence over T (T is small: 8/16 frames).
        B, T, D = k.shape
        a = torch.zeros(B, D, device=x.device, dtype=torch.float32)
        b = torch.zeros(B, D, device=x.device, dtype=torch.float32)
        outs: list[Tensor] = []
        for t in range(T):
            kt = k[:, t]
            vt = v[:, t]
            kv = kt * vt
            a = w * a + kv
            b = w * b + kt
            outs.append(r[:, t] * (a + u * kv) / (b + u * kt + self.eps))
        return torch.stack(outs, dim=1)

    def _channel_mixing(self, x: Tensor) -> Tensor:
        """RWKV-4 channel-mixing on [B, T, D]. x is assumed pre-LN."""
        shifted = _time_shift(x)

        xk = x * self.time_mix_gk.float() + shifted * (1.0 - self.time_mix_gk.float())
        xr = x * self.time_mix_gr.float() + shifted * (1.0 - self.time_mix_gr.float())

        k = torch.square(F.relu(_fp32_linear(self.gate_key, xk)))
        v = _fp32_linear(self.gate_value, k)
        r = torch.sigmoid(_fp32_linear(self.gate_receptance, xr))
        return r * v

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x: Tensor) -> Tensor:
        """Forward on [B, T, D] -> [B, T, D] in float32."""
        x = x.float()
        h = _fp32_layer_norm(self.ln1, x)
        x = x + self._time_mixing(h)
        h = _fp32_layer_norm(self.ln2, x)
        x = x + self._channel_mixing(h)
        return x


class VitaCLIPDiscriminativeRWKV(nn.Module):
    """
    Discriminative region selection + shared-weight RWKV temporal module.

    Input:
        x:           [B*T, N, D] tokens (CLS at index 0, then N_p patches)
        attn_weights: [B*T, N_p] real CLS -> patch attention weights
                      (from the transformer block with return_attention=True)

    Output:
        y: [B*T, N, D]. Selected discriminative patch tokens receive the
        gated RWKV delta; the CLS token receives the mean delta of the
        selected positions (see step 9 below).

    Main procedure:
        1. Restore [B, T, N, D] and split CLS / patch tokens.
        2. Attention score A = attn_weights (per frame, per patch).
        3. Motion strength M = mean over channels of squared frame
           difference along time (FP32, zero-parameter).
        4. Fusion W = alpha * A + (1 - alpha) * M + score(h), where score
           is a small learnable spatial scorer (zero-initialised) that makes
           the top-k selection adaptive over training.
        5. Keep the top-k patch positions per frame -> mask r.
        6. Union of selected positions across the batch, extract the
           cross-frame sequences [B*|U|, T, D].
        7. Shared-weight RWKV (project D -> bottleneck -> D) models the
           temporal relation of every selected position.
        8. h' = h + tanh(residual_scale) * delta * r, only selected tokens
           are updated; unselected tokens keep their original values.
        9. CLS temporal write-back: cls' = cls + soft-weighted sum of
           delta[j] (softmax over w / tau on the selected positions of each
           frame). This is required for gradient flow at the last transformer
           block (only CLS reaches the final frame feature), mirrors the
           frequency adapter design, and is the differentiable bridge that
           makes the scorer learnable.
    """

    def __init__(
        self,
        dim: int = 768,
        bottleneck: int = 192,
        num_spatial_patches: int = 196,
        top_k: int = 6,
        alpha: float = 0.4,
        rwkv_layers: int = 1,
        residual_scale_init: float = 0.01,
        stability_clip: float = 100.0,
        min_frames: int = 4,
        eps: float = 1e-6,
        topk_temperature: float = 0.5,
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
        if rwkv_layers <= 0:
            raise ValueError("rwkv_layers must be positive.")
        if stability_clip <= 0:
            raise ValueError("stability_clip must be positive.")
        if min_frames < 2:
            raise ValueError("min_frames must be at least 2.")

        self.dim = int(dim)
        self.bottleneck = int(bottleneck)
        self.num_spatial_patches = int(num_spatial_patches)
        self.top_k = int(top_k)
        self.alpha = float(alpha)
        self.rwkv_layers = int(rwkv_layers)
        self.stability_clip = float(stability_clip)
        self.min_frames = int(min_frames)
        self.eps = float(eps)
        self.topk_temperature = float(topk_temperature)

        # Bottleneck projections around the RWKV stack.
        self.down_projection = nn.Linear(dim, bottleneck)
        self.rwkv_norm = nn.LayerNorm(bottleneck)
        self.rwkv_blocks = nn.ModuleList(
            [_RWKVBlock(bottleneck, eps=eps) for _ in range(self.rwkv_layers)]
        )
        self.up_projection = nn.Linear(bottleneck, dim)

        # Final residual strength:
        # h' = h + tanh(residual_scale) * delta * r
        # NOTE: fixed buffer (not a learnable Parameter). A learnable rs
        # self-amplified during the lr peak (gradient positive feedback) and
        # caused loss blowups at step 961 (rs=0.1) and step 1991 (rs=0.005).
        # Freezing it at 0.01 (the v3-verified magnitude) removes the entire
        # instability class while keeping the write-back magnitude fixed.
        self.register_buffer(
            "residual_scale",
            torch.tensor(float(residual_scale_init), dtype=torch.float32),
        )

        # NEW: learnable spatial scorer -> makes the top-k selection adaptive.
        # Zero-initialised last layer -> score ~ 0 at init -> the module
        # behaves exactly like the old static w during warmup, so the
        # training curve can be compared against the static version.
        # Parameters are named *discriminative_modules.*.scoring.* -> they
        # automatically fall into the adapter optimiser group.
        self.scoring = nn.Sequential(
            nn.Linear(dim, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, 1),
        )
        nn.init.zeros_(self.scoring[-1].weight)
        nn.init.zeros_(self.scoring[-1].bias)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Keep the module close to identity without blocking gradients.

        Follows the verified frequency-adapter trick: the up projection uses
        Xavier initialization (zero init froze gradients at ~1e-10) and the
        residual scale controls the perturbation magnitude.
        """
        nn.init.normal_(self.down_projection.weight, mean=0.0, std=0.05)
        nn.init.zeros_(self.down_projection.bias)
        nn.init.xavier_uniform_(self.up_projection.weight)
        nn.init.zeros_(self.up_projection.bias)

    def _sanitize_real(self, x: Tensor) -> Tensor:
        limit = self.stability_clip
        return torch.nan_to_num(
            x,
            nan=0.0,
            posinf=limit,
            neginf=-limit,
        ).clamp(min=-limit, max=limit)

    def _motion_strength(self, patch: Tensor) -> Tensor:
        """Frame-difference squared mean -> [B, T, N_p] motion strength.

        M[b, t, j] = mean_k ( (h[b,t+1,j,k] - h[b,t,j,k])^2 ) averaged over
        the neighbouring frame difference, computed in float32 with zero
        learnable parameters. First / last frames use the single available
        difference.
        """
        diff = patch[:, 1:, :, :] - patch[:, :-1, :, :]  # [B, T-1, N_p, D]
        diff_sq = (diff * diff).mean(dim=-1)             # [B, T-1, N_p]
        motion = torch.zeros_like(patch[..., 0])
        motion[:, 1:, :] += diff_sq
        motion[:, :-1, :] += diff_sq
        counts = torch.ones_like(motion)
        counts[:, 1:, :] += 1.0
        counts[:, :-1, :] += 1.0
        return motion / counts

    @torch.cuda.amp.autocast(enabled=False)
    def forward(
        self,
        x: Tensor,
        attn_weights: Tensor,
        *,
        num_frames: int,
        patch_start: int = 1,
        return_information: bool = False,
    ) -> Tensor:
        """Forward on [B*T, N, D] -> [B*T, N, D] (float32 internally)."""
        if x.ndim != 3:
            raise ValueError(f"x must be [B*T, N, D], got {tuple(x.shape)}.")
        total_tokens, num_tokens, dim = x.shape
        if num_tokens < 2:
            raise ValueError("x must contain a CLS token and at least one patch.")
        if num_frames <= 1 or total_tokens % num_frames != 0:
            raise ValueError(
                f"num_frames={num_frames} incompatible with {total_tokens} tokens."
            )
        batch_size = total_tokens // num_frames
        num_patches = num_tokens - patch_start
        if attn_weights.shape != (total_tokens, num_patches):
            raise ValueError(
                f"attn_weights must be [{total_tokens}, {num_patches}], "
                f"got {tuple(attn_weights.shape)}."
            )

        # Short-sequence protection (same as the frequency adapter).
        if num_frames < self.min_frames:
            if return_information:
                return x, {"skipped": True, "reason": "num_frames < min_frames"}
            return x

        x_float = x.float()
        tokens = x_float.view(batch_size, num_frames, num_tokens, dim)
        cls_tokens = tokens[:, :, :patch_start, :]   # [B, T, 1, D]
        patch_tokens = tokens[:, :, patch_start:, :]  # [B, T, N_p, D]

        # 1. Attention score A: [B, T, N_p].
        attn = attn_weights.float().view(batch_size, num_frames, num_patches)

        # 2. Motion strength M: [B, T, N_p].
        motion = self._motion_strength(patch_tokens)

        # 3. Fusion W = alpha * A + (1 - alpha) * M + learnable score.
        #    The scorer makes the top-k selection adaptive: as training
        #    progresses the ordering of w drifts, so the selected regions
        #    migrate towards task-relevant spatial locations.
        score = self.scoring(patch_tokens).squeeze(-1)  # [B, T, N_p]
        w = self.alpha * attn + (1.0 - self.alpha) * motion + score
        w = self._sanitize_real(w)

        # 4. Top-k selection per frame -> mask r [B, T, N_p] (bool).
        #    Hard argmax keeps the union sparse and the patch write-back
        #    magnitude intact (selected patches get the full delta).
        k = min(self.top_k, num_patches)
        top_indices = w.topk(k, dim=-1).indices  # [B, T, k]
        mask = torch.zeros_like(w, dtype=torch.bool).scatter_(
            -1, top_indices, True
        )

        # 4b. Soft top-k weights over the SAME support set:
        #     logits = -inf outside top-k -> softmax yields exactly k
        #     non-zero entries summing to 1. This is the differentiable
        #     gradient bridge for the scorer (the hard argmax above stays
        #     non-differentiable). Non-selected scores get no gradient,
        #     i.e. a soft straight-through-estimator: winners get reinforced.
        top_values, _ = w.topk(k, dim=-1)                    # [B, T, k]
        logits = torch.full_like(w, float("-inf"))
        logits.scatter_(-1, top_indices, top_values / self.topk_temperature)
        soft_weights = torch.softmax(logits, dim=-1)         # [B, T, N_p]

        # 5. Union of selected positions across the whole batch.
        any_selected = mask.any(dim=1)  # [B, N_p]
        union_pos = torch.unique(any_selected.nonzero()[:, 1])  # [|U|]
        if union_pos.numel() == 0:
            # Defensive: nothing selected (cannot happen when k >= 1).
            if return_information:
                info = {
                    "skipped": True,
                    "reason": "empty union of selected positions",
                    "attn_mean": attn.float().mean().detach(),
                    "motion_mean": motion.float().mean().detach(),
                    "top_k": k,
                    "alpha": self.alpha,
                }
                return x, info
            return x

        # 6. Extract cross-frame sequences for the selected positions.
        #    seq: [B, T, |U|, D] -> [B*|U|, T, D]
        seq = patch_tokens[:, :, union_pos, :].permute(0, 2, 1, 3)
        seq = seq.reshape(-1, num_frames, dim)

        # 7. Shared-weight RWKV temporal modelling in float32.
        bottleneck_feature = _fp32_linear(self.down_projection, seq)
        h = _fp32_layer_norm(self.rwkv_norm, bottleneck_feature)
        for block in self.rwkv_blocks:
            h = block(h)
        delta = _fp32_linear(self.up_projection, h)
        delta = self._sanitize_real(delta)

        # 8. Residual write-back with gate, only on selected tokens.
        residual_scale = torch.tanh(self.residual_scale.float())
        delta = (residual_scale * delta).to(dtype=torch.float32)

        # delta: [B*|U|, T, D] -> [B, |U|, T, D] -> [B, T, |U|, D]
        delta = delta.view(batch_size, union_pos.numel(), num_frames, dim)
        delta = delta.permute(0, 2, 1, 3)

        sel_mask = mask[:, :, union_pos].unsqueeze(-1)  # [B, T, |U|, 1]
        delta = delta * sel_mask

        updated_patch = patch_tokens.clone()
        updated_patch[:, :, union_pos, :] = (
            updated_patch[:, :, union_pos, :] + delta
        )
        updated_patch = self._sanitize_real(updated_patch)

        # 9. CLS temporal write-back: SOFT-weighted aggregation of the RWKV
        # deltas over the selected positions of every frame, added to CLS.
        #
        # The frame feature is read from the CLS token only
        # (ln_post(x[:, 0, :])), so a patch-only write-back at the last
        # transformer block (layer 11) can never reach the loss -> the module
        # parameters receive zero gradient. Adding the temporal delta to CLS
        # gives every layer a direct gradient path and mirrors the frequency
        # adapter, which also writes its temporal delta to the CLS token.
        #
        # Weights sum to 1 over each frame's top-k (convex combination), so
        # the magnitude matches the old equal-weight average and
        # residual_scale needs no retuning. In exchange we get:
        #   (a) the ordering information of w is used, and
        #   (b) gradients flow through softmax back into the scorer, making
        #       the top-k selection itself learnable.
        # delta is already zeroed outside each frame's top-k by sel_mask
        # (step 8), so the sum below equals sum_{j in top-k} soft_w_j * d_j.
        sel_w = soft_weights[:, :, union_pos].unsqueeze(-1)  # [B, T, |U|, 1]
        cls_delta = (delta * sel_w).sum(dim=2)               # [B, T, D]
        cls_delta = self._sanitize_real(cls_delta)
        updated_cls = (
            cls_tokens[..., 0, :] + cls_delta
        ).unsqueeze(2)                                        # [B, T, 1, D]

        out = torch.cat([updated_cls, updated_patch], dim=2)
        out = out.reshape(total_tokens, num_tokens, dim).to(dtype=x.dtype)

        if return_information:
            info: Dict[str, object] = {
                "skipped": False,
                "attn_mean": attn.float().mean().detach(),
                "motion_mean": motion.float().mean().detach(),
                "w_mean": w.mean().detach(),
                "score_mean": score.mean().detach(),
                "soft_entropy": (
                    -(soft_weights * soft_weights.clamp_min(1e-8).log())
                    .sum(dim=-1)
                    .mean()
                    .detach()
                ),
                "num_selected_per_frame": mask.float().sum(dim=-1).detach(),
                "union_size": int(union_pos.numel()),
                "residual_scale": residual_scale.detach(),
                "delta_abs_mean": delta.detach().abs().mean(),
                "cls_delta_abs_mean": cls_delta.detach().abs().mean(),
                "top_k": k,
                "alpha": self.alpha,
            }
            return out, info
        return out

