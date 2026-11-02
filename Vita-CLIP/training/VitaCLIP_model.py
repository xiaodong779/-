#!/usr/bin/env python3
"""OpenAI-CLIP-aligned Vita-CLIP model.

Design goals
------------
1. Preserve the original OpenAI CLIP parameter tree and computation:
   - visual.conv1
   - visual.class_embedding
   - visual.positional_embedding
   - visual.ln_pre
   - visual.transformer.resblocks
   - visual.ln_post
   - visual.proj
   - token_embedding
   - positional_embedding
   - transformer.resblocks
   - ln_final
   - text_projection
   - logit_scale
2. Load the official OpenAI ViT-B/16 TorchScript state dict directly, without
   Q/K/V conversion and without a second checkpoint load.
3. Insert the custom temporal discriminative modules before complete visual
   blocks 3, 5, 7 and 9 (zero-based indices by default).
4. Insert Text_MSTFF after text ``ln_final`` and before ``text_projection``.
5. Keep the original CLIP encoders frozen while allowing the new temporal and
   text-adaptation parameters to train.
6. Retain the old constructor arguments used by the current training script.

The external modules are intentionally kept independent:
    VitaCLIP_discriminative_rap.py
    Text_MSTFF.py
    VitaCLIP_text_encoder.py  (tokenizer + TextPromptLearner only)
"""

from __future__ import annotations

import math
import re
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from Text_MSTFF import TextMSTFF
from BIE import BIE
from SCR import SemanticClassRelationRefinement
from TGL_adapter import TGLAdapter, TGLTrajectoryReadout
from tgl_track_st import TrackSTEncoder
from tgl_late_fusion import TemporalPool, LateFusion, track_quality_stats
from tgl_geometry import track_patch_indices
from tgl_track_st import build_motion_input
from tgl_video_prompt import PromptFrameTokens, TGLPromptGenerator
from VitaCLIP_discriminative_dual import VitaCLIPDiscriminativeDual
from VitaCLIP_discriminative_rap import VitaCLIPDiscriminativeRAP
from VitaCLIP_discriminative_rwkv import VitaCLIPDiscriminativeRWKV
from VitaCLIP_text_encoder import TextPromptLearner, tokenize
from modules.correspondence_video_prompt import (
    CorrespondenceVideoPromptBlock,
)
from modules.layered_video_adaptation import (
    LayeredVideoPrompts,
    MaskPatchAdapter,
)


def _fp32_linear(module: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    """Execute a Linear layer in float32 while preserving gradient flow.

    Mirrors the project-wide ``_fp32_linear`` convention (see BIE.py /
    CMSI.py) so the temporal mixer is safe under AMP mixed precision.
    """
    weight = module.weight.float()
    if module.bias is not None:
        bias = module.bias.float()
    else:
        bias = None
    return torch.nn.functional.linear(
        x.float(),
        weight,
        bias,
    ).to(x.dtype)


class LayerNorm(nn.LayerNorm):
    """LayerNorm that preserves the input dtype, as in OpenAI CLIP."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_dtype = x.dtype
        return super().forward(x.float()).to(original_dtype)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    """Original OpenAI CLIP residual attention block."""

    def __init__(
        self,
        d_model: int,
        n_head: int,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        # Recurrent four-slot video prompt injection (TGL v2).
        # Prompt tokens are appended before self-attention and removed before
        # the frozen MLP.
        self.prompt_alpha = None

    def add_prompt_injection(
        self,
        num_prompts: int,
        num_heads: int = 8,
        alpha_init: float = 0.1,
    ) -> None:
        del num_prompts, num_heads
        if self.prompt_alpha is None:
            self.prompt_alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def attention(
        self,
        x: torch.Tensor,
        *,
        return_weights: bool = False,
        per_head: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        mask = self.attn_mask
        if mask is not None:
            mask = mask.to(device=x.device, dtype=x.dtype)

        if return_weights:
            attn_output, attn_weights = self.attn(
                x,
                x,
                x,
                need_weights=True,
                # per_head=True returns the [BT, heads, N, N] tensor required
                # by the CVP Block; the default averaged behavior is kept for
                # every pre-existing path.
                average_attn_weights=not per_head,
                attn_mask=mask,
            )
            return attn_output, attn_weights

        return self.attn(
            x,
            x,
            x,
            need_weights=False,
            attn_mask=mask,
        )[0]

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_attention: bool = False,
        attn_per_head: bool = False,
        prompt_tokens: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # `x` is seq-first [S, BT, D]. Prompt tokens are supplied batch-first
        # [BT, P, D], appended only for this block, and removed immediately
        # after self-attention. Only visual tokens enter the frozen MLP.
        visual_tokens = x.shape[0]
        if (
            prompt_tokens is not None
            and self.prompt_alpha is not None
            and float(self.prompt_alpha.detach()) == 0.0
        ):
            if return_attention:
                attn_out, attn_weights = self.attention(
                    self.ln_1(x), return_weights=True, per_head=attn_per_head
                )
                x = x + attn_out
                x_base = x + self.mlp(self.ln_2(x))
                x = x_base + (
                    self.prompt_alpha.to(x.dtype)
                    - self.prompt_alpha.detach().to(x.dtype)
                ) * x_base
                return x, attn_weights
            x = x + self.attention(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
            x = x + (
                self.prompt_alpha.to(x.dtype)
                - self.prompt_alpha.detach().to(x.dtype)
            ) * x
            return x

        if prompt_tokens is not None and self.prompt_alpha is not None:
            h_visual = self.ln_1(x)
            base_attn = self.attn(
                h_visual, h_visual, h_visual, need_weights=False,
                attn_mask=self.attn_mask,
            )[0]
            x_base = x + base_attn
            prompt_tokens = prompt_tokens.permute(1, 0, 2).to(x.dtype)
            h_prompt = self.ln_1(prompt_tokens)
            kv = torch.cat([h_visual, h_prompt], dim=0)
            if return_attention:
                prompt_attn, attn_weights = self.attn(
                    h_visual, kv, kv, need_weights=True,
                    average_attn_weights=not attn_per_head,
                    attn_mask=None,
                )
                x_prompt = x + prompt_attn
                x = x_base + self.prompt_alpha.to(x.dtype) * (
                    x_prompt - x_base
                )
                x = x + self.mlp(self.ln_2(x))
                return x, attn_weights
            prompt_attn = self.attn(
                h_visual, kv, kv, need_weights=False, attn_mask=None
            )[0]
            x_prompt = x + prompt_attn
            x = x_base + self.prompt_alpha.to(x.dtype) * (x_prompt - x_base)
            x = x + self.mlp(self.ln_2(x))
            return x

        if return_attention:
            attn_out, attn_weights = self.attention(
                self.ln_1(x), return_weights=True, per_head=attn_per_head
            )
            x = x + attn_out
            x = x + self.mlp(self.ln_2(x))
            return x, attn_weights
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    """Original OpenAI CLIP Transformer parameter tree."""

    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.width = int(width)
        self.layers = int(layers)
        self.resblocks = nn.ModuleList(
            [
                ResidualAttentionBlock(
                    d_model=width,
                    n_head=heads,
                    attn_mask=attn_mask,
                )
                for _ in range(layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        if return_attention:
            attn_weights: List[torch.Tensor] = []
            for block in self.resblocks:
                x, weights = block(x, return_attention=True)
                attn_weights.append(weights)
            return x, attn_weights
        for block in self.resblocks:
            x = block(x)
        return x


class _TemporalAggregationMixer(nn.Module):
    """Frame-aggregation mixer applied just before the temporal mean.

    out = x + tanh(scale) * depthwise_conv1d(x)      # x: [B, T, D]

    Identity start: conv weights are zero-initialized and ``scale`` starts at
    0, so the very first step is exactly the plain ``mean(dim=1)`` baseline.
    The depthwise design (groups=D) keeps the temporal mixing channel-local
    and cheap (~D * kernel params), while ``tanh`` bounds the injected delta
    to [-max_scale, +max_scale] so the CLIP feature scale is preserved.
    Computations run in fp32 (project ``_fp32_linear`` convention) and the
    output is cast back to the input dtype for AMP safety.
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int = 3,
        max_scale: float = 0.5,
    ) -> None:
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError(
                f"kernel_size must be odd for time-preserving padding, "
                f"got {kernel_size}."
            )
        self.dim = int(dim)
        self.max_scale = float(max_scale)
        self.conv1d = nn.Conv1d(
            self.dim,
            self.dim,
            kernel_size=kernel_size,
            groups=self.dim,
            padding=kernel_size // 2,
            bias=False,
        )
        nn.init.zeros_(self.conv1d.weight)
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        x: torch.Tensor,  # [B, T, D]
    ) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        gate = self.max_scale * torch.tanh(self.scale)
        delta = self.conv1d(x.transpose(1, 2)).transpose(1, 2)
        return (x + gate * delta).to(dtype)


class _TemporalTransformerMixer(nn.Module):
    """Lightweight temporal self-attention over frame features [B, T, D].

    Replaces the depthwise-conv mixer's weak temporal modelling (3-point
    channel-local smoothing, no cross-channel interaction, no long-range
    frame relation) with a single multi-head self-attention block over the
    T frame tokens. This is the shortest path from per-frame CLS features to
    the classification head, so long-range action evolution (e.g. the full
    "raise arm -> swing -> hit" sequence) can flow into the final video
    feature instead of being collapsed by the depthwise conv + mean.

    Design invariants (kept consistent with the rest of the codebase):

    1. Identity start: the whole block is a residual with a zero-initialized
       scalar gate, ``out = x + tanh(scale) * attn_block(x)``, so the very
       first step is exactly the plain ``mean(dim=1)`` baseline and the
       frozen CLIP features are never disturbed at init.
    2. Learnable per-frame positional encoding so attention is aware of
       frame order (small, zero-initialized -> identity at start).
    3. Bounded injection via ``max_scale * tanh(scale)``.
    4. fp32 math throughout (``_fp32_linear`` convention), output cast back
       to the input dtype for AMP safety.

    Parameters are tiny: D=512 -> ~1.05M params for one 8-head block.
    """

    def __init__(
        self,
        dim: int,
        num_frames: int = 8,
        num_heads: int = 8,
        max_scale: float = 0.5,
        dropout: float = 0.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive.")
        if num_frames <= 0:
            raise ValueError("num_frames must be positive.")
        if num_heads <= 0 or dim % num_heads != 0:
            raise ValueError(
                f"num_heads={num_heads} must divide dim={dim}."
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        self.dim = int(dim)
        self.num_frames = int(num_frames)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        self.max_scale = float(max_scale)
        self.eps = float(eps)

        self.ln1 = nn.LayerNorm(self.dim, eps=self.eps)
        self.q_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.k_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.v_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.out_proj = nn.Linear(self.dim, self.dim, bias=False)

        # Per-frame position embedding [1, T, D]: one independently-learnable
        # vector per frame position (NOT a single [1, 1, D] vector that is
        # linearly interpolated, which would give nearly-identical encodings
        # to neighbouring frames and destroy frame-order discrimination).
        # Zero-initialized so the first step has no positional bias.
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_frames, self.dim)
        )

        # Identity start: the entire attention block is gated by a scalar
        # that starts at 0, so step 0 == plain mean baseline.
        self.scale = nn.Parameter(torch.zeros(1))

        self.dropout = nn.Dropout(dropout)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        # out_proj must NOT be exactly zero-initialized: with a zero out_proj
        # the attention branch output ``out`` is 0 at step 0, so the scale
        # gate gradient ``max_scale * out`` is also 0 AND ``out_proj``'s own
        # gradient ``dL/doutput * gate * attn_out`` is 0 (gate == 0 at init),
        # producing a chicken-and-egg deadlock where scale stays 0 forever and
        # the whole mixer never activates. A tiny non-zero std breaks the
        # deadlock while keeping the first step effectively at the identity
        # (delta O(1e-3), indistinguishable from the baseline in practice).
        nn.init.normal_(self.out_proj.weight, std=1e-3)

    def forward(
        self,
        x: torch.Tensor,  # [B, T, D]
    ) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        batch_size, num_frames, _ = x.shape

        # Residual position encoding (zero at init -> identity).
        pos = self.pos_embed.expand(batch_size, -1, -1)  # [B, T, D]
        if num_frames != pos.shape[1]:
            # Fallback for a different clip length than the one used at init:
            # interpolate the per-frame embeddings to the requested length.
            pos = F.interpolate(
                pos.transpose(1, 2),
                size=num_frames,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)

        h = self.ln1(x) + pos  # [B, T, D]

        q = _fp32_linear(self.q_proj, h).view(
            batch_size, num_frames, self.num_heads, self.head_dim
        ).transpose(1, 2)  # [B, H, T, Hd]
        k = _fp32_linear(self.k_proj, h).view(
            batch_size, num_frames, self.num_heads, self.head_dim
        ).transpose(1, 2)
        v = _fp32_linear(self.v_proj, h).view(
            batch_size, num_frames, self.num_heads, self.head_dim
        ).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(
            self.head_dim
        )  # [B, H, T, T]
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)  # [B, H, T, Hd]
        out = out.transpose(1, 2).reshape(batch_size, num_frames, self.dim)
        out = _fp32_linear(self.out_proj, out)

        gate = self.max_scale * torch.tanh(self.scale)
        return (x + gate * out).to(dtype)


class VisualTransformer(nn.Module):
    """OpenAI CLIP ViT with external temporal discriminative modules.

    Original CLIP parameters retain their exact names. New parameters are
    registered only under:
        time_embed
        discriminative_modules
    """

    def __init__(
        self,
        input_resolution: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        output_dim: int,
        num_frames: int = 8,
        use_discriminative_module: bool = False,
        dr_layers: Sequence[int] = (3, 5, 7, 9),
        share_discriminative_module: bool = False,
        dr_top_k: int = 12,
        dr_alpha: float = 0.4,
        dr_bottleneck: int = 192,
        dr_rwkv_layers: int = 1,
        dr_residual_scale_init: float = 0.01,
        dr_topk_temperature: float = 0.5,
        dr_arch: str = "rwkv",
        dr_mamba_scope: str = "full",
        dr_use_conv3d: bool = True,
        dr_use_mamba: bool = True,
        dr_mamba_bidir: bool = False,
        dr_d_state: int = 16,
        dr_d_conv: int = 4,
        dr_use_asa: bool = True,
        dr_asa_heads: int = 8,
        dr_asa_time_offset_max: float = 1.0,
        dr_asa_space_offset_max: float = 4.0,
        dr_asa_window_size: int = 5,
        use_aggregation_mixer: bool = False,
        aggregation_mixer_kernel: int = 3,
        aggregation_mixer_type: str = "transformer",
        aggregation_mixer_heads: int = 8,
        aggregation_mixer_dropout: float = 0.0,
        # Correspondence-Guided Video Prompt Block (CVP)
        use_cvp_block: bool = False,
        cvp_insert_layer: int = 4,
        cvp_topk: int = 64,
        cvp_window_size: int = 7,
        cvp_num_prompts: int = 4,
        cvp_bottleneck_dim: int = 192,
        cvp_num_heads: int = 4,
        cvp_gamma_min: float = 0.1,
        cvp_gamma_max: float = 0.5,
        cvp_gamma_init: float = 0.2,
        cvp_beta_min: float = 0.5,
        cvp_beta_max: float = 1.5,
        cvp_beta_init: float = 1.0,
        cvp_selection_method: str = "adaptive_attention_motion",
        cvp_motion_max_weight: float = 0.6,
        cvp_current_context_attention: bool = True,
        cvp_corr_scope: str = "full",
        cvp_spatial_bias_max: float = 4.0,
        cvp_spatial_bias_init: float = 1.0,
        # Decoupled layered video adaptation (v3):
        #   mask adapter (AIM/ST-style) + layered video prompts (VPT-Deep).
        use_mask_adapter: bool = False,
        mask_adapter_layers: str = "2,4,6,8",
        mask_adapter_topk: int = 64,
        mask_adapter_bottleneck: int = 192,
        mask_adapter_heads: int = 4,
        mask_adapter_motion_max_weight: float = 0.6,
        use_layered_prompts: bool = False,
        num_video_prompts: int = 4,
        prompt_cond_rank: int = 192,
        prompt_insert_layers: str = "all",
        # Trajectory-Guided Local Spatio-Temporal Adapter (TGL), clean
        # TAT-like MID-LAYER version: one adapter per insertion layer writes
        # `patch = patch + max_scale*tanh(gamma)*residual_grid` into the patch
        # stream between frozen ViT blocks, so the enhanced patches flow
        # through all subsequent blocks and influence CLS naturally. The
        # adapter can additionally emit trajectory-residual video prompts
        # (zero-init head; source = residual tokens, NEVER
        # mean_pool(enhanced_patches)). mean_pool(enhanced) survives only as
        # an ablation switch (tgl_patch_pool_baseline, default OFF). Tracks
        # are OFFLINE caches; the tracker is NEVER run here.
        use_tgl_adapter: bool = False,
        tgl_adapter_dim: Optional[int] = None,
        tgl_num_layers: int = 2,
        tgl_num_heads: int = 4,
        tgl_dropout: float = 0.1,
        tgl_gather_mode: str = "nearest",
        tgl_local_tube_size: int = 1,
        tgl_visibility_threshold: float = 0.5,
        tgl_max_scale: float = 0.1,
        tgl_max_fusion: float = 0.5,
        tgl_up_proj_init: str = "tiny",
        tgl_insert_mode: str = "blocks",
        tgl_insert_layers: str = "6,8,10",
        tgl_num_prompts: int = 4,
        tgl_patch_pool_baseline: bool = False,
        tgl_gate_init: float = 0.0,
        tgl_readout: bool = True,
        tgl_readout_gate: float = 0.15,
        use_tgl_v2: bool = False,
        tgl_v2_prompt_source_layer: int = 4,
        tgl_v2_prompt_layers: str = "5,6,7,8,9,10,11,12",
        tgl_v2_num_prompts: int = 4,
        tgl_v2_num_classes: int = 0,
        tgl_v2_recon_ratio: float = 0.1,
        tgl_v2_readout_gate: float = 0.15,
    ) -> None:
        super().__init__()

        self.input_resolution = int(input_resolution)
        self.output_dim = int(output_dim)
        self.width = int(width)
        self.num_frames = int(num_frames)
        self.use_discriminative_module = bool(use_discriminative_module)
        self.dr_layers = self._validate_layer_indices(
            dr_layers,
            layers,
        )
        self.share_discriminative_module = bool(share_discriminative_module)
        self.dr_arch = str(dr_arch)
        self.use_aggregation_mixer = bool(use_aggregation_mixer)
        # The mixer runs on frame features AFTER ln_post + proj, whose channel
        # dimension is output_dim (e.g. 512 for ViT-B/16), not width (768).
        mixer_type = aggregation_mixer_type.strip().lower()
        if mixer_type not in ("conv", "transformer"):
            raise ValueError(
                "aggregation_mixer_type must be 'conv' or 'transformer', "
                f"got {aggregation_mixer_type!r}."
            )
        if self.use_aggregation_mixer:
            if mixer_type == "conv":
                self.aggregation_mixer = _TemporalAggregationMixer(
                    output_dim,
                    kernel_size=aggregation_mixer_kernel,
                )
            else:
                self.aggregation_mixer = _TemporalTransformerMixer(
                    output_dim,
                    num_frames=self.num_frames,
                    num_heads=aggregation_mixer_heads,
                    dropout=aggregation_mixer_dropout,
                )
        else:
            self.aggregation_mixer = None

        # ------------------------------------------------------------------
        # ------------------------------------------------------------------
        # Original OpenAI CLIP visual parameter names.
        # ------------------------------------------------------------------
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )

        scale = width ** -0.5
        grid_size = input_resolution // patch_size
        num_tokens = grid_size * grid_size + 1

        self.class_embedding = nn.Parameter(
            scale * torch.randn(width)
        )
        self.positional_embedding = nn.Parameter(
            scale * torch.randn(num_tokens, width)
        )
        self.ln_pre = LayerNorm(width)
        self.transformer = Transformer(
            width=width,
            layers=layers,
            heads=heads,
        )
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(
            scale * torch.randn(width, output_dim)
        )
        # Correspondence-Guided Video Prompt Block (inserted after frozen
        # Block `cvp_insert_layer`). D is read from the actual CLIP width;
        # the frozen CLIP patch position embedding is passed detached so the
        # mask query cannot train through it.
        # ------------------------------------------------------------------
        self.use_cvp_block = bool(use_cvp_block)
        self.cvp_insert_layer = int(cvp_insert_layer)
        if self.use_cvp_block:
            if not 1 <= self.cvp_insert_layer < layers:
                raise ValueError(
                    f"cvp_insert_layer must be in [1, {layers - 1}] "
                    f"(CVP runs after resblocks[{self.cvp_insert_layer - 1}]), "
                    f"got {self.cvp_insert_layer}."
                )
            grid = input_resolution // patch_size
            self.cvp_block = CorrespondenceVideoPromptBlock(
                d_model=width,
                num_frames=self.num_frames,
                grid_size=grid,
                topk=cvp_topk,
                window_size=cvp_window_size,
                num_prompts=cvp_num_prompts,
                bottleneck=cvp_bottleneck_dim,
                num_heads=cvp_num_heads,
                gamma_min=cvp_gamma_min,
                gamma_max=cvp_gamma_max,
                gamma_init=cvp_gamma_init,
                beta_min=cvp_beta_min,
                beta_max=cvp_beta_max,
                beta_init=cvp_beta_init,
                selection_method=cvp_selection_method,
                motion_max_weight=cvp_motion_max_weight,
                use_current_context_attention=cvp_current_context_attention,
                corr_scope=cvp_corr_scope,
                spatial_bias_max=cvp_spatial_bias_max,
                spatial_bias_init=cvp_spatial_bias_init,
            )
            self.register_buffer(
                "cvp_clip_patch_pos_embedding",
                self.positional_embedding[1:].detach().clone(),
                persistent=False,
            )
        else:
            self.cvp_block = None

        # ------------------------------------------------------------------
        # Decoupled layered video adaptation (v3):
        #   * mask adapter  -- AIM/ST-Adapter-style residual adapter applied
        #     after selected frozen blocks; per-layer mask re-selection;
        #     zero-initialised up-projection and gate (identity at start);
        #     NO reconstruction branch (motion is used for selection only).
        #   * layered prompts -- VPT-Deep-style per-layer independent
        #     video-conditioned prompts, prepended at each block input and
        #     discarded after the block.
        # Both are mutually exclusive with the single-shot CVP block.
        # ------------------------------------------------------------------
        self.use_mask_adapter = bool(use_mask_adapter)
        self.use_layered_prompts = bool(use_layered_prompts)
        self.grid_size = input_resolution // patch_size

        # ------------------------------------------------------------------
        # TGL trajectory adapters, clean TAT-like mid-layer version. One
        # adapter per insertion layer; inert unless track tensors are passed
        # to encode_frames. Tracks are OFFLINE-precomputed (CoTracker3 runs
        # in tgl_tracker_offline.py, never here / never in the dataloader).
        # ------------------------------------------------------------------
        self.use_tgl_adapter = bool(use_tgl_adapter)
        self.tgl_max_fusion = float(tgl_max_fusion)
        self.tgl_insert_mode = str(tgl_insert_mode)
        if self.tgl_insert_mode != "blocks":
            raise ValueError(
                f"tgl_insert_mode must be 'blocks', got {tgl_insert_mode!r}"
            )
        self.tgl_insert_layers = tuple(
            int(s) for s in str(tgl_insert_layers).split(",") if s.strip()
        )
        self.tgl_num_prompts = int(tgl_num_prompts)
        self.tgl_patch_pool_baseline = bool(tgl_patch_pool_baseline)
        if self.use_tgl_adapter:
            if not self.tgl_insert_layers:
                raise ValueError("tgl_insert_layers must not be empty.")
            for layer_index in self.tgl_insert_layers:
                if not 0 < layer_index < len(self.transformer.resblocks):
                    raise ValueError(
                        f"tgl_insert_layers entry {layer_index} outside "
                        f"[1, {len(self.transformer.resblocks) - 1}]."
                    )
            self.tgl_adapters = nn.ModuleList(
                [
                    TGLAdapter(
                        dim=width,
                        adapter_dim=tgl_adapter_dim,
                        num_layers=tgl_num_layers,
                        num_heads=tgl_num_heads,
                        dropout=tgl_dropout,
                        gather_mode=tgl_gather_mode,
                        local_tube_size=tgl_local_tube_size,
                        visibility_threshold=tgl_visibility_threshold,
                        max_scale=tgl_max_scale,
                        up_proj_init=tgl_up_proj_init,
                        gate_init=tgl_gate_init,
                    )
                    for _ in self.tgl_insert_layers
                ]
            )
            # Optional ablation baseline: mean_pool(enhanced_patches) readout
            # with its own zero-init gate. DEFAULT OFF - it is exactly the
            # confounded branch (gate opens for raw patch content, not for
            # trajectory content) that the residual-sourced design removes.
            self.tgl_patch_pool_baseline = self.tgl_patch_pool_baseline
            self.tgl_baseline_logit = (
                nn.Parameter(torch.zeros(1))
                if self.tgl_patch_pool_baseline
                else None
            )
            # TAT-style trajectory readout: the trajectory token stream is a
            # first-class, trained representation with a DIRECT path into the
            # video feature (learnable-query attention pooling + warm fusion
            # gate). This is the evidence-based fix for the gradient-starved
            # residual-only design.
            self.tgl_readout = (
                TGLTrajectoryReadout(
                    self.tgl_adapters[-1].adapter_dim, output_dim
                )
                if bool(tgl_readout)
                else None
            )
            self.tgl_readout_max = float(tgl_readout_gate) / 0.6 if tgl_readout_gate > 0 else 0.0
            self.tgl_readout_logit = (
                nn.Parameter(
                    torch.full(
                        (1,),
                        math.atanh(
                            min(0.99, float(tgl_readout_gate) / max(self.tgl_readout_max, 1e-6))
                        ),
                    )
                )
                if self.tgl_readout is not None and self.tgl_readout_max > 0
                else None
            )
            self._last_tgl_z: Optional[torch.Tensor] = None
            self._last_tgl_valid: Optional[torch.Tensor] = None
            self._last_tgl_stats: Dict[str, object] = {}
        else:
            self.tgl_adapters = None
            self.tgl_baseline_logit = None
            self.tgl_readout = None
            self.tgl_readout_logit = None
            self._last_tgl_z = None
            self._last_tgl_valid = None
            self._last_tgl_stats: Dict[str, object] = {}
        # ---- TGL v2: per-layer video prompts (CVPT) + Track-ST + fusion ----
        self.use_tgl_v2 = bool(use_tgl_v2)
        self.tgl_v2_prompt_source_layer = int(tgl_v2_prompt_source_layer)  # 1-based
        self.tgl_v2_prompt_layers = [
            int(s) for s in str(tgl_v2_prompt_layers).split(",") if s.strip()
        ]  # 1-based block numbers receiving prompts
        self.tgl_v2_num_prompts = int(tgl_v2_num_prompts)
        self.tgl_v2_recon_ratio = float(tgl_v2_recon_ratio)
        self.v2_frame_tokens = None
        self.v2_prompt_gen = None
        self.v2_track_st = None
        self.temporal_pool = None
        self.v2_late_fusion = None
        self.v2_prompt_tokens: List[Optional[torch.Tensor]] = []
        self._last_v2_patches: Optional[torch.Tensor] = None
        self._last_v2_clean_patches: Optional[torch.Tensor] = None
        self._last_v2_cls: Optional[torch.Tensor] = None
        self._last_v2_aux: Dict[str, torch.Tensor] = {}
        if self.use_tgl_v2:
            assert self.tgl_v2_prompt_source_layer < min(self.tgl_v2_prompt_layers), (
                "prompt_source_layer must precede every prompt layer "
                f"(source={self.tgl_v2_prompt_source_layer}, "
                f"layers={self.tgl_v2_prompt_layers})"
            )
            self.v2_frame_tokens = None
            self.v2_prompt_gen = TGLPromptGenerator(
                width,
                192,
                4,
                self.tgl_v2_num_prompts,
                num_prompt_layers=len(self.tgl_v2_prompt_layers),
                beta_init=0.5,
            )
            for layer_number in self.tgl_v2_prompt_layers:
                self.transformer.resblocks[layer_number - 1].add_prompt_injection(
                    self.tgl_v2_num_prompts,
                    alpha_init=0.1,
                )
            self.v2_track_st = TrackSTEncoder(
                clip_dim=width,
                track_dim=192,
                num_layers=2,
                num_heads=4,
                num_classes=tgl_v2_num_classes,
            )
            self.temporal_pool = TemporalPool()
            self.v2_late_fusion = LateFusion(dim=output_dim, quality_dim=4, gate_init=0.1)
        if not self.use_tgl_v2:
            self.v2_prompt_gen = None
            self.v2_track_st = None
            self.temporal_pool = None
            self.v2_late_fusion = None
            self.tgl_adapters = None
            self.tgl_baseline_logit = None
            self.tgl_readout = None
            self.tgl_readout_logit = None
            self._last_tgl_z = None
            self._last_tgl_valid = None
            self._last_tgl_stats = {}
        if self.use_mask_adapter and self.use_cvp_block:
            raise ValueError(
                "use_mask_adapter and use_cvp_block are mutually exclusive: "
                "both rewrite the patch token stream between frozen blocks."
            )
        if self.use_layered_prompts and self.use_cvp_block:
            raise ValueError(
                "use_layered_prompts and use_cvp_block are mutually "
                "exclusive: both inject prompt tokens into the stream."
            )
        self.mask_adapter = None
        self.layered_prompts = None
        if self.use_mask_adapter:
            try:
                mask_layers_1b = [
                    int(v)
                    for v in str(mask_adapter_layers).split(",")
                    if str(v).strip()
                ]
            except ValueError as exc:
                raise ValueError(
                    f"mask_adapter_layers must be comma-separated ints, "
                    f"got {mask_adapter_layers!r}"
                ) from exc
            if not mask_layers_1b:
                raise ValueError("mask_adapter_layers is empty")
            mask_layers_0b = [v - 1 for v in mask_layers_1b]
            if any(v < 0 or v >= layers for v in mask_layers_0b):
                raise ValueError(
                    f"mask_adapter_layers (1-based) must be within "
                    f"[1, {layers}], got {mask_layers_1b}"
                )
            self.mask_adapter = MaskPatchAdapter(
                d_model=width,
                num_frames=self.num_frames,
                insertion_layers=mask_layers_0b,
                topk=mask_adapter_topk,
                bottleneck=mask_adapter_bottleneck,
                num_heads=mask_adapter_heads,
                motion_max_weight=mask_adapter_motion_max_weight,
            )
        if self.use_layered_prompts:
            text = str(prompt_insert_layers).strip().lower()
            if text in ("", "all"):
                insert_layers = None
            else:
                try:
                    insert_layers = [
                        int(v) - 1 for v in text.split(",") if v.strip()
                    ]
                except ValueError as exc:
                    raise ValueError(
                        f"prompt_insert_layers must be 'all' or "
                        f"comma-separated 1-based ints, got "
                        f"{prompt_insert_layers!r}"
                    ) from exc
                if any(v < 0 or v >= layers for v in insert_layers):
                    raise ValueError(
                        f"prompt_insert_layers (1-based) must be within "
                        f"[1, {layers}], got {prompt_insert_layers!r}"
                    )
            self.layered_prompts = LayeredVideoPrompts(
                d_model=width,
                num_layers=layers,
                num_prompts=num_video_prompts,
                cond_rank=prompt_cond_rank,
                insert_layers=insert_layers,
            )


        # ------------------------------------------------------------------
        # New video parameters.
        # ``time_embed`` is exactly zero at initialization, so the visual
        # backbone reproduces frame-wise OpenAI CLIP until the temporal
        # discriminative modules are activated.
        # ------------------------------------------------------------------
        self.time_embed = nn.Parameter(
            torch.zeros(num_frames, width)
        )

        num_spatial_patches = grid_size * grid_size

        # ------------------------------------------------------------------
        # Discriminative region + temporal module (RWKV or dual-branch).
        # ------------------------------------------------------------------
        if self.use_discriminative_module:
            if dr_arch == "rap":
                def build_dr() -> nn.Module:
                    return VitaCLIPDiscriminativeRAP(
                        dim=width,
                        bottleneck=dr_bottleneck,
                        num_spatial_patches=num_spatial_patches,
                        top_k=dr_top_k,
                        alpha=dr_alpha,
                        use_conv3d=dr_use_conv3d,
                        use_asa=dr_use_asa,
                        asa_heads=dr_asa_heads,
                        asa_time_offset_max=dr_asa_time_offset_max,
                        asa_space_offset_max=dr_asa_space_offset_max,
                        asa_window_size=dr_asa_window_size,
                        residual_scale_init=dr_residual_scale_init,
                        topk_temperature=dr_topk_temperature,
                    )
            elif dr_arch == "dual":
                def build_dr() -> nn.Module:
                    return VitaCLIPDiscriminativeDual(
                        dim=width,
                        bottleneck=dr_bottleneck,
                        num_spatial_patches=num_spatial_patches,
                        top_k=dr_top_k,
                        alpha=dr_alpha,
                        mamba_scope=dr_mamba_scope,
                        use_conv3d=dr_use_conv3d,
                        use_mamba=dr_use_mamba,
                        mamba_bidir=dr_mamba_bidir,
                        d_state=dr_d_state,
                        d_conv=dr_d_conv,
                        residual_scale_init=dr_residual_scale_init,
                        topk_temperature=dr_topk_temperature,
                    )
            else:

                def build_dr() -> nn.Module:
                    return VitaCLIPDiscriminativeRWKV(
                        dim=width,
                        bottleneck=dr_bottleneck,
                        num_spatial_patches=num_spatial_patches,
                        top_k=dr_top_k,
                        alpha=dr_alpha,
                        rwkv_layers=dr_rwkv_layers,
                        residual_scale_init=dr_residual_scale_init,
                        topk_temperature=dr_topk_temperature,
                    )

            if self.share_discriminative_module:
                self.discriminative_modules = nn.ModuleDict(
                    {"shared": build_dr()}
                )
            else:
                self.discriminative_modules = nn.ModuleDict(
                    {
                        str(layer_index): build_dr()
                        for layer_index in self.dr_layers
                    }
                )
        else:
            self.discriminative_modules = nn.ModuleDict()

    @staticmethod
    def _validate_layer_indices(
        layer_indices: Sequence[int],
        num_layers: int,
    ) -> Tuple[int, ...]:
        layers = tuple(int(index) for index in layer_indices)

        if len(layers) != len(set(layers)):
            raise ValueError(
                f"layer indices contain duplicates: {layers}."
            )

        invalid = [
            index
            for index in layers
            if index < 0 or index >= num_layers
        ]
        if invalid:
            raise ValueError(
                "layer indices use zero-based block indices in "
                f"[0, {num_layers - 1}], got {invalid}."
            )

        return layers

    def _temporal_embedding(
        self,
        num_frames: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        time_embed = self.time_embed

        if num_frames != time_embed.shape[0]:
            time_embed = F.interpolate(
                time_embed.transpose(0, 1).unsqueeze(0).float(),
                size=num_frames,
                mode="linear",
                align_corners=False,
            ).squeeze(0).transpose(0, 1)

        return time_embed.to(device=device, dtype=dtype)

    def _add_temporal_embedding(
        self,
        x: torch.Tensor,
        batch_size: int,
        num_frames: int,
    ) -> torch.Tensor:
        """Add frame-position embeddings to every token.

        Args:
            x: [B*T, N, D]
        """
        bt, num_tokens, width = x.shape
        if bt != batch_size * num_frames:
            raise ValueError(
                "Invalid flattened video size: "
                f"BT={bt}, B={batch_size}, T={num_frames}."
            )

        time_embed = self._temporal_embedding(
            num_frames=num_frames,
            dtype=x.dtype,
            device=x.device,
        )

        x = x.reshape(
            batch_size,
            num_frames,
            num_tokens,
            width,
        )
        x = x + time_embed.view(1, num_frames, 1, width)
        return x.reshape(bt, num_tokens, width)

    def encode_frames(
        self,
        images: torch.Tensor,
        *,
        num_frames: int,
        apply_temporal_modules: bool = True,
        tracks: Optional[torch.Tensor] = None,
        visibility: Optional[torch.Tensor] = None,
        birth_frame: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode flattened video frames.

        Args:
            images: [B*T, 3, H, W]
            num_frames: temporal length T

        Returns:
            Frame features [B*T, output_dim].
        """
        if images.ndim != 4:
            raise ValueError(
                "VisualTransformer expects [B*T, 3, H, W], "
                f"got {tuple(images.shape)}."
            )
        if images.shape[1] != 3:
            raise ValueError(
                f"Expected RGB input, got C={images.shape[1]}."
            )
        if images.shape[-2:] != (
            self.input_resolution,
            self.input_resolution,
        ):
            raise ValueError(
                "Unexpected spatial size: "
                f"expected {(self.input_resolution, self.input_resolution)}, "
                f"got {tuple(images.shape[-2:])}."
            )
        if num_frames <= 0:
            raise ValueError(
                f"num_frames must be positive, got {num_frames}."
            )
        if images.shape[0] % num_frames != 0:
            raise ValueError(
                f"BT={images.shape[0]} is not divisible by T={num_frames}."
            )

        self._last_cvp_corr_loss = None
        self._last_cvp_stats = None
        self._last_tgl_stats = {}

        batch_size = images.shape[0] // num_frames

        # OpenAI CLIP patch embedding.
        x = self.conv1(images)
        x = x.reshape(
            x.shape[0],
            x.shape[1],
            -1,
        )
        x = x.permute(0, 2, 1)

        # CVP: capture the pre-Block1 patch tokens (patch projection +
        # original CLIP spatial position embedding, through ln_pre; no
        # self-attention has happened yet). Detached — the current-frame
        # context branch must not train through them, but they carry the
        # video content that conditions the mask queries.
        if (
            self.cvp_block is not None
            or getattr(self, "layered_prompts", None) is not None
            or getattr(self, "mask_adapter", None) is not None
        ) and apply_temporal_modules:
            pre_blocks = images.shape[0] // num_frames
            self._last_input_patch_tokens = (
                self.ln_pre(
                    x + self.positional_embedding[1:]
                    .to(x.dtype)[None]
                )
                .detach()
                .reshape(pre_blocks, num_frames, -1, x.shape[-1])
            )
        else:
            self._last_input_patch_tokens = None

        class_token = (
            self.class_embedding.to(x.dtype)
            + torch.zeros(
                x.shape[0],
                1,
                x.shape[-1],
                dtype=x.dtype,
                device=x.device,
            )
        )
        x = torch.cat([class_token, x], dim=1)
        x = x + self.positional_embedding.to(
            device=x.device,
            dtype=x.dtype,
        )

        if apply_temporal_modules:
            x = self._add_temporal_embedding(
                x,
                batch_size=batch_size,
                num_frames=num_frames,
            )

        x = self.ln_pre(x)

        # OpenAI CLIP uses [N, batch, D] inside the Transformer.
        x = x.permute(1, 0, 2)

        # Temporal module placement (position experiment):
        #   * runs *before* its own transformer block, so the block's
        #     multi-head attention sees cross-frame mixed tokens (time-first);
        #   * top-k region selection uses the CLS -> patch attention of the
        #     PREVIOUS block (causal: no own attention exists yet), which is
        #     semantically valid -- the previous layer's saliency guides the
        #     temporal mixing of the current layer.
        # Blocks right before each inserted layer must materialize their
        # attention weights (cached into prev_cls_to_patch).
        attn_src_layers = {layer - 1 for layer in self.dr_layers if layer > 0}
        prev_cls_to_patch = None

        use_cvp = (
            self.cvp_block is not None
            and apply_temporal_modules
            and not self.use_discriminative_module
        )
        cvp_attn = None
        use_tgl_v2_fwd = self.use_tgl_v2 and apply_temporal_modules

        # ------------------------------------------------------------------
        # Decoupled layered video adaptation (v3). Mutually exclusive with
        # CVP (enforced at construction) and with the discriminative module.
        #   * layered prompts: VPT-Deep-style per-layer injection
        #   * mask adapter:    AIM/ST-style post-block residual adapter
        # ------------------------------------------------------------------
        use_layered_prompts = (
            self.layered_prompts is not None
            and apply_temporal_modules
            and not self.use_discriminative_module
        )
        use_mask_adapter = (
            self.mask_adapter is not None
            and apply_temporal_modules
            and not self.use_discriminative_module
        )
        if use_mask_adapter:
            # the mask adapter at block L re-selects patches using the
            # CLS->patch attention of the PREVIOUS block (causal, same
            # convention as the discriminative modules).
            attn_src_layers |= {
                l - 1 for l in self.mask_adapter.insertion_layers if l > 0
            }
        video_embedding = None
        if use_layered_prompts:
            if self._last_input_patch_tokens is None:
                raise RuntimeError(
                    "Layered prompts require the pre-Block1 patch tokens; "
                    "check apply_temporal_modules."
                )
            video_embedding = (
                self._last_input_patch_tokens.float()
                .mean(dim=(1, 2))
                .to(x.dtype)
            )  # [B, D], detached by construction

        self._last_layered_stats: Dict[str, torch.Tensor] = {}

        for layer_index, block in enumerate(
            self.transformer.resblocks
        ):
            n_patch = self.grid_size * self.grid_size  # constant patch count

            # ---- per-layer prompts BEFORE the block (VPT-Deep) ----------
            append_prompts = (
                use_layered_prompts
                and layer_index in self.layered_prompts.insert_layers
            )
            if append_prompts:
                p5 = self.layered_prompts(video_embedding, layer_index)
                # [B, P, D] -> [P, B*T, D] with bt ordered as b*T+t (same
                # (b,t) split convention as x's time-first batch dim).
                p_tokens = (
                    p5[:, :, None, :]
                    .expand(-1, -1, num_frames, -1)
                    .permute(1, 0, 2, 3)
                    .reshape(p5.shape[1], x.shape[1], x.shape[2])
                    .contiguous()
                )
                x = torch.cat([x, p_tokens], dim=0)
                for k, v in self.layered_prompts.current_stats().items():
                    self._last_layered_stats[
                        f"prompt_L{layer_index}_{k}"
                    ] = v

            # CVP needs the REAL per-head attention computed by Block
            # `cvp_insert_layer` (executed exactly once, no re-run).
            if use_cvp and layer_index == self.cvp_insert_layer - 1:
                x, cvp_attn = block(x, return_attention=True, attn_per_head=True)
                continue
            # CVP insertion point: after resblocks[cvp_insert_layer - 1],
            # before resblocks[cvp_insert_layer].
            if use_cvp and layer_index == self.cvp_insert_layer:
                if cvp_attn is None:
                    raise RuntimeError(
                        "CVP Block4 attention was not captured."
                    )
                n_tokens, bt, width = x.shape
                batch_size = bt // num_frames
                heads = cvp_attn.shape[1]
                tokens_5d = (
                    x.permute(1, 0, 2)
                    .reshape(batch_size, num_frames, n_tokens, width)
                )  # [B, T, 1+N, D]
                attn_5d = cvp_attn.reshape(
                    batch_size, num_frames, heads, n_tokens, n_tokens
                )  # [B, T, heads, 1+N, 1+N]

                (
                    enhanced_patches,
                    video_prompts,
                    corr_loss,
                    cvp_stats,
                ) = self.cvp_block(
                    tokens_5d,
                    attn_5d,
                    self.cvp_clip_patch_pos_embedding,
                    self._last_input_patch_tokens,
                )
                self._last_cvp_corr_loss = corr_loss
                self._last_cvp_stats = cvp_stats

                merged = torch.cat(
                    [
                        tokens_5d[:, :, 0:1],
                        video_prompts[:, None].expand(
                            -1, num_frames, -1, -1
                        ),
                        enhanced_patches,
                    ],
                    dim=2,
                )  # [B, T, 1+P+N, D]
                x = (
                    merged.reshape(bt, merged.shape[2], width)
                    .permute(1, 0, 2)
                    .contiguous()
                )  # [1+P+N, BT, D]
                continue

            use_dr = (
                apply_temporal_modules
                and self.use_discriminative_module
                and layer_index in self.dr_layers
                and num_frames >= 4
            )
            if use_dr:
                if prev_cls_to_patch is None:
                    # Defensive fallback (e.g. dr_layers starts at 0): uniform
                    # CLS -> patch attention instead of a missing tensor.
                    BT = x.shape[1]
                    N_p = x.shape[0] - 1
                    prev_cls_to_patch = torch.full(
                        (BT, N_p),
                        1.0 / N_p,
                        device=x.device,
                        dtype=x.dtype,
                    )
                adapter_key = (
                    "shared"
                    if self.share_discriminative_module
                    else str(layer_index)
                )
                tokens = x.permute(1, 0, 2).contiguous()
                tokens = self.discriminative_modules[adapter_key](
                    tokens,
                    prev_cls_to_patch,
                    num_frames=num_frames,
                    patch_start=1,
                )
                x = tokens.permute(1, 0, 2).contiguous()

            # ---- TGL v2: recurrent four-slot prompt -----------------------
            # The current layer's slots are generated from the previous
            # layer's frame CLS tokens, then used only by this block.
            prompt_tokens_i = None
            if (
                use_tgl_v2_fwd
                and layer_index + 1 in self.tgl_v2_prompt_layers
                and layer_index >= self.tgl_v2_prompt_source_layer
            ):
                frame_cls = x[0].reshape(batch_size, num_frames, self.width)
                q_ctx, pstats = self.v2_prompt_gen(frame_cls)
                self._last_tgl_stats[f"v2_prompt_gen_L{layer_index + 1}"] = dict(pstats)
                slot = self.tgl_v2_prompt_layers.index(layer_index + 1)
                pl = self.v2_prompt_gen.layer_prompts(q_ctx, slot)
                prompt_tokens_i = (
                    pl[:, None]
                    .expand(-1, num_frames, -1, -1)
                    .reshape(batch_size * num_frames, pl.shape[1], pl.shape[2])
                )

            if layer_index in attn_src_layers:
                x, attn_weights = block(x, return_attention=True, prompt_tokens=prompt_tokens_i)
                prev_cls_to_patch = attn_weights[:, 0, 1 : 1 + n_patch].contiguous()
            else:
                x = block(x, prompt_tokens=prompt_tokens_i)

            if (
                use_tgl_v2_fwd
                and layer_index + 1 == self.tgl_v2_prompt_source_layer
            ):
                self._last_v2_clean_patches = (
                    x[1:].permute(1, 0, 2)
                    .reshape(batch_size, num_frames, n_patch, self.width)
                    .detach()
                )

            # ---- mask adapter AFTER the block (AIM/ST style) ------------
            if use_mask_adapter and (
                layer_index in self.mask_adapter.insertion_layers
            ):
                if prev_cls_to_patch is None:
                    # defensive fallback: first-layer insertion (l == 0 has
                    # no previous block) uses uniform attention.
                    BT = x.shape[1]
                    N_p = x.shape[0] - 1
                    prev_cls_to_patch = torch.full(
                        (BT, N_p),
                        1.0 / N_p,
                        device=x.device,
                        dtype=x.dtype,
                    )
                x = self.mask_adapter(
                    x,
                    batch_size=x.shape[1] // num_frames,
                    num_frames=num_frames,
                    layer_index=layer_index,
                    cls_to_patch_attention=prev_cls_to_patch,
                )
                for k, v in self.mask_adapter.current_stats().items():
                    self._last_layered_stats[
                        f"mask_L{layer_index}_{k}"
                    ] = v

            # ---- TGL mid-layer insertion (clean TAT-like adapter) -------
            # After frozen block (layer_index), i.e. BEFORE block
            # (layer_index + 1): patch = patch + scale * residual_grid, with
            # scale = max_scale * tanh(gamma) bounded inside the adapter. The
            # enhanced patches then flow through all subsequent blocks, so
            # CLS is influenced through the attention itself - no pooled side
            # branch is needed. Optionally emits trajectory-residual video
            # prompts (zero-init head, source = residual tokens).
            if (
                apply_temporal_modules
                and self.tgl_adapters is not None
                and tracks is not None
                and visibility is not None
                and birth_frame is not None
                and (layer_index + 1) in self.tgl_insert_layers
            ):
                a_idx = self.tgl_insert_layers.index(layer_index + 1)
                adapter = self.tgl_adapters[a_idx]
                n_prompt_tail = x.shape[0] - 1 - n_patch
                patch_bt = x[1 + n_prompt_tail:].permute(1, 0, 2)  # [BT, N, D]
                pt5 = patch_bt.reshape(
                    batch_size, num_frames, n_patch, self.width
                )
                enhanced, tgl_aux = adapter(
                    pt5.float(),
                    tracks=tracks,
                    visibility=visibility,
                    birth_frame=birth_frame,
                    patch_hw=(self.grid_size, self.grid_size),
                    image_size=(self.input_resolution, self.input_resolution),
                )
                # capture for the TAT-style readout (last insertion wins)
                self._last_tgl_z = tgl_aux["trajectory_tokens"]
                self._last_tgl_valid = tgl_aux["valid_mask"]
                delta = (enhanced - pt5.float()).to(x.dtype)
                patch_ref = patch_bt.float().norm().clamp_min(1e-6)
                vp_slice = x[1 : 1 + n_prompt_tail]
                vp_norm = (
                    float(vp_slice.float().norm() / math.sqrt(max(1, vp_slice.numel())))
                    if n_prompt_tail > 0
                    else 0.0
                )
                self._last_tgl_stats[f"L{layer_index + 1}"] = {
                    "adapter_scale": float(tgl_aux["adapter_scale"]),
                    "delta_over_patch": float(
                        delta.float().norm() / patch_ref
                    ),
                    "scatter_coverage": float(
                        (tgl_aux["scatter_weight"] > 0).float().mean()
                    ),
                    "video_prompt_norm": vp_norm,
                }
                enhanced_seq = enhanced.reshape(
                    batch_size * num_frames, n_patch, self.width
                ).permute(1, 0, 2).to(x.dtype)
                x = torch.cat(
                    [x[: 1 + n_prompt_tail], enhanced_seq, x[1 + n_prompt_tail:]],
                    dim=0,
                )

        # ---- TGL v2: stash the FINAL raw patches and per-frame CLS -------
        if use_tgl_v2_fwd:
            self._last_v2_patches = (
                x[1:].permute(1, 0, 2)
                .reshape(batch_size, num_frames, n_patch, self.width)
            )  # [B, T, 196, 768] - final visual patches, row-major order kept
            self._last_v2_cls = x[0].reshape(batch_size, num_frames, self.width)

        x = x.permute(1, 0, 2)
        frame_features = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            frame_features = frame_features @ self.proj

        # ------------------------------------------------------------------
        # TGL aggregate stats + OPTIONAL ablation baseline (default OFF).
        # The main TGL path is the mid-layer insertion above; the mean_pool
        # (enhanced_patches) readout survives ONLY behind
        # tgl_patch_pool_baseline for attribution ablations - it must never
        # be used to claim trajectory gains (the gate opens for raw patch
        # content, not for trajectory content).
        # ------------------------------------------------------------------
        if self.tgl_adapters is not None:
            gates = [
                float(torch.tanh(a.gamma.detach().float())) * a.max_scale
                for a in self.tgl_adapters
            ]
            self._last_tgl_stats["tgl_gate_mean"] = (
                sum(gates) / len(gates) if gates else 0.0
            )
            if self.tgl_patch_pool_baseline and self.tgl_baseline_logit is not None:
                n_prompt_tail = x.shape[1] - 1 - n_patch
                pooled = x[:, 1 + n_prompt_tail:, :].mean(dim=1)
                patch_feat = self.ln_post(pooled)
                if self.proj is not None:
                    patch_feat = patch_feat @ self.proj
                baseline_gate = (
                    torch.tanh(self.tgl_baseline_logit.float())
                    * self.tgl_max_fusion
                )
                frame_features = frame_features + baseline_gate * patch_feat.to(
                    frame_features.dtype
                )
                self._last_tgl_stats["baseline_gate"] = float(
                    torch.tanh(self.tgl_baseline_logit.detach().float())
                    * self.tgl_max_fusion
                )

        return frame_features

    def forward(
        self,
        images: torch.Tensor,
        *,
        num_frames: Optional[int] = None,
        apply_temporal_modules: bool = True,
        tracks: Optional[torch.Tensor] = None,
        visibility: Optional[torch.Tensor] = None,
        birth_frame: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return one video feature per clip.

        Args:
            images: flattened frames [B*T, 3, H, W]
        Returns:
            video features [B, output_dim]
        """
        temporal_length = (
            self.num_frames
            if num_frames is None
            else int(num_frames)
        )

        frame_features = self.encode_frames(
            images,
            num_frames=temporal_length,
            apply_temporal_modules=apply_temporal_modules,
            tracks=tracks,
            visibility=visibility,
            birth_frame=birth_frame,
            labels=labels,
        )

        batch_size = frame_features.shape[0] // temporal_length
        frame_features = frame_features.reshape(
            batch_size,
            temporal_length,
            self.output_dim,
        )
        if self.aggregation_mixer is not None and temporal_length > 1:
            frame_features = self.aggregation_mixer(frame_features)
        # --- TAT-style trajectory readout: the pooled trajectory feature is
        # fused into the video feature through a WARM gate (init ~0.3). The
        # trajectory stream is a first-class trained representation with a
        # direct, strongly-supervised path into the classifier. ---
        video = frame_features.mean(dim=1)
        if (
            self.tgl_readout is not None
            and self._last_tgl_z is not None
            and self._last_tgl_valid is not None
        ):
            traj = self.tgl_readout(
                self._last_tgl_z.float(), self._last_tgl_valid
            )
            gate = torch.tanh(self.tgl_readout_logit.float()) * self.tgl_readout_max
            vn = video.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
            tn = traj.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            video = video + gate * (traj / tn * vn).to(video.dtype)
            self._last_tgl_stats["readout_gate"] = float(
                torch.tanh(self.tgl_readout_logit.detach().float())
                * self.tgl_readout_max
            )
            self._last_tgl_stats["traj_feat_norm"] = float(
                traj.detach().float().norm() / math.sqrt(max(1, traj.numel()))
            )
        # ---- TGL v2: video_visual (TemporalPool) + trajectory branch + ----
        # ---- pure-image adaptive late fusion -> final_visual [B, 512] ----
        if (
            self.use_tgl_v2
            and self.tgl_readout is None
            and self._last_v2_patches is not None
            and tracks is not None
        ):
            video_visual = F.normalize(
                self.temporal_pool(frame_features).float(), dim=-1
            ).to(video.dtype)
            pi, in_bounds, _ = track_patch_indices(
                tracks[..., 0], tracks[..., 1],
                self.grid_size, self.grid_size,
                self.input_resolution, self.input_resolution,
            )
            frame_ids = torch.arange(
                tracks.shape[2], device=tracks.device
            )[None, None, :]
            birth_valid = frame_ids >= birth_frame.to(
                device=tracks.device
            ).long().unsqueeze(-1)
            valid = (visibility.float() > 0.5) & in_bounds & birth_valid
            gather_idx = (
                pi.permute(0, 2, 1)  # [B, T, K]
                .unsqueeze(-1)
                .expand(-1, -1, -1, self.width)
            )
            patch_source = self._last_v2_clean_patches
            if patch_source is None:
                patch_source = self._last_v2_patches
            appearance = patch_source.float().gather(
                2, gather_idx
            ).permute(0, 2, 1, 3).contiguous()  # [B, K, T, 768]
            motion = build_motion_input(
                tracks, visibility, valid,
                self.input_resolution, self.input_resolution,
            )
            st = self.v2_track_st(
                appearance, motion, valid,
                labels=labels if self.training else None,
                recon_mask_ratio=self.tgl_v2_recon_ratio if self.training else 0.0,
                clip_target=appearance.detach(),
                dxdy_target=motion[..., 2:4].detach(),
            )
            trajectory_visual = st["trajectory_visual"]
            quality = track_quality_stats(valid, visibility, in_bounds, st["any_valid"])
            final_visual, fusion_gate = self.v2_late_fusion(
                video_visual, trajectory_visual, quality, st["any_valid"]
            )
            self._last_v2_aux = {
                "video_visual": video_visual,
                "trajectory_visual": trajectory_visual,
                "trajectory_cls": st["trajectory_cls"],
                "fusion_gate": fusion_gate,
                "track_quality": quality,
                "any_valid": st["any_valid"],
                "valid": valid,
                "in_bounds": in_bounds,
            }
            self._last_tgl_stats.update({
                "v2_fusion_gate": float(fusion_gate.detach().float().mean()),
                "v2_valid_ratio": float(valid.float().mean()),
                "v2_video_norm": float(video_visual.detach().float().norm(dim=-1).mean()),
                "v2_trajectory_norm": float(trajectory_visual.detach().float().norm(dim=-1).mean()),
                "v2_valid_track_ratio": float(quality[:, 0].detach().float().mean()),
                "v2_mean_track_length": float(quality[:, 2].detach().float().mean()),
            })
            if "trajectory_logits" in st and labels is not None:
                self._last_v2_aux["loss_traj_aux"] = F.cross_entropy(
                    st["trajectory_logits"].float(), labels
                )
            if "loss_recon_clip" in st:
                self._last_v2_aux["loss_recon"] = (
                    st["loss_recon_clip"] + st["loss_recon_dxdy"]
                )
            return final_visual
        return video


class VitaCLIP(nn.Module):
    """OpenAI-CLIP-aligned video action-recognition model."""

    def __init__(
        self,
        # Checkpoint
        backbone_path: str = "",
        # Input / dataset
        input_size: Tuple[int, int] = (224, 224),
        num_frames: int = 8,
        num_classes: int = 101,
        batch_size: Optional[int] = None,  # retained for old scripts
        # Vision
        feature_dim: int = 768,
        patch_size: Tuple[int, int] = (16, 16),
        num_heads: int = 12,
        num_layers: int = 12,
        mlp_factor: float = 4.0,  # retained for old scripts
        embed_dim: int = 512,
        # Old Vita-CLIP visual-prompt switches (must remain disabled)
        use_original_visual_prompts: bool = False,
        use_summary_token: bool = False,
        use_local_prompts: bool = False,
        use_global_prompts: bool = False,
        num_global_prompts: int = 8,
        # Discriminative region + temporal module (RWKV or dual-branch)
        use_discriminative_module: bool = False,
        dr_layers: Sequence[int] = (3, 5, 7, 9),
        share_discriminative_module: bool = False,
        dr_top_k: int = 12,
        dr_alpha: float = 0.4,
        dr_bottleneck: int = 192,
        dr_rwkv_layers: int = 1,
        dr_residual_scale_init: float = 0.01,
        dr_topk_temperature: float = 0.5,
        dr_arch: str = "rwkv",
        dr_mamba_scope: str = "full",
        dr_use_conv3d: bool = True,
        dr_use_mamba: bool = True,
        dr_mamba_bidir: bool = False,
        dr_d_state: int = 16,
        dr_d_conv: int = 4,
        dr_use_asa: bool = True,
        dr_asa_heads: int = 8,
        dr_asa_time_offset_max: float = 1.0,
        dr_asa_space_offset_max: float = 4.0,
        dr_asa_window_size: int = 5,
        # Text/prompt compatibility
        use_text_prompt_learning: bool = True,
        use_fixed_text_branch: bool = True,
        use_learnable_text_branch: bool = True,
        use_text_mstff: bool = True,
        fixed_text_template: str = "a video of a person {}",
        text_context_length: int = 77,
        text_vocab_size: int = 49408,
        text_transformer_width: int = 512,
        text_transformer_heads: int = 8,
        text_transformer_layers: int = 12,
        text_num_prompts: int = 8,
        text_prompt_pos: str = "end",
        text_prompt_init: str = "",
        text_prompt_CSC: bool = False,
        text_prompt_classes_path: str = "",
        naturalize_class_names: bool = True,
        text_mstff_gate_init: float = 0.01,
        text_fusion_fixed_logit: float = 3.0,
        text_fusion_learnable_logit: float = 0.0,
        # Zero-shot evaluation
        zeroshot_evaluation: bool = False,
        zeroshot_text_features_path: str = "",
        # Bidirectional Interpolation Enhancement (supervised only)
        use_bie: bool = False,
        bie_p_temperature: float = 10.0,
        bie_max_gate: float = 0.5,
        bie_gate_warmup_steps: int = 0,
        bie_use_transform: bool = True,
        bie_transform_bottleneck: int = 128,
        bie_gate_bottleneck: int = 128,
        bie_enh_thr: float = 0.0,
        bie_max_dist: float = 0.05,
        bie_max_dist_t: float = 0.05,
        bie_alpha_aux_lambda: float = 0.1,
        # Semantic Class-Relation Refinement (supervised only, static graph)
        use_scr: bool = False,
        scr_top_k: int = 5,
        scr_max_scale: float = 0.5,
        # Frame-aggregation temporal mixer (before the temporal mean)
        use_aggregation_mixer: bool = False,
        aggregation_mixer_kernel: int = 3,
        aggregation_mixer_type: str = "transformer",
        aggregation_mixer_heads: int = 8,
        aggregation_mixer_dropout: float = 0.0,
        # Correspondence-Guided Video Prompt Block (CVP)
        use_cvp_block: bool = False,
        cvp_insert_layer: int = 4,
        cvp_topk: int = 64,
        cvp_window_size: int = 7,
        cvp_num_prompts: int = 4,
        cvp_bottleneck_dim: int = 192,
        cvp_num_heads: int = 4,
        cvp_gamma_min: float = 0.1,
        cvp_gamma_max: float = 0.5,
        cvp_gamma_init: float = 0.2,
        cvp_beta_min: float = 0.5,
        cvp_beta_max: float = 1.5,
        cvp_beta_init: float = 1.0,
        cvp_corr_loss_weight: float = 0.1,
        cvp_selection_method: str = "adaptive_attention_motion",
        cvp_motion_max_weight: float = 0.6,
        cvp_current_context_attention: bool = True,
        cvp_corr_scope: str = "full",
        cvp_spatial_bias_max: float = 4.0,
        cvp_spatial_bias_init: float = 1.0,
        # Decoupled layered video adaptation (v3)
        use_mask_adapter: bool = False,
        mask_adapter_layers: str = "2,4,6,8",
        mask_adapter_topk: int = 64,
        mask_adapter_bottleneck: int = 192,
        mask_adapter_heads: int = 4,
        mask_adapter_motion_max_weight: float = 0.6,
        use_layered_prompts: bool = False,
        num_video_prompts: int = 4,
        prompt_cond_rank: int = 192,
        prompt_insert_layers: str = "all",
        use_tgl_adapter: bool = False,
        tgl_adapter_dim: Optional[int] = None,
        tgl_num_layers: int = 2,
        tgl_num_heads: int = 4,
        tgl_dropout: float = 0.1,
        tgl_gather_mode: str = "nearest",
        tgl_local_tube_size: int = 1,
        tgl_visibility_threshold: float = 0.5,
        tgl_max_scale: float = 0.1,
        tgl_max_fusion: float = 0.5,
        tgl_up_proj_init: str = "tiny",
        tgl_insert_mode: str = "blocks",
        tgl_insert_layers: str = "6,8,10",
        tgl_num_prompts: int = 4,
        tgl_patch_pool_baseline: bool = False,
        tgl_gate_init: float = 0.0,
        tgl_readout: bool = True,
        tgl_readout_gate: float = 0.15,
        use_tgl_v2: bool = False,
        tgl_v2_prompt_source_layer: int = 4,
        tgl_v2_prompt_layers: str = "5,6,7,8,9,10,11,12",
        tgl_v2_num_prompts: int = 4,
        tgl_v2_recon_ratio: float = 0.1,
        tgl_v2_readout_gate: float = 0.15,
        _openai_state_dict: Optional[
            Mapping[str, torch.Tensor]
        ] = None,
        **legacy_kwargs: object,
    ) -> None:
        super().__init__()

        del batch_size, mlp_factor, num_global_prompts, legacy_kwargs

        if use_original_visual_prompts or any(
            (
                use_summary_token,
                use_local_prompts,
                use_global_prompts,
            )
        ):
            raise ValueError(
                "The OpenAI-aligned model intentionally disables the old "
                "summary/local/global visual prompt path. Set all old visual "
                "prompt switches to False."
            )

        if len(input_size) != 2 or input_size[0] != input_size[1]:
            raise ValueError(
                "OpenAI ViT expects a square input_size, "
                f"got {input_size}."
            )
        if patch_size[0] != patch_size[1]:
            raise ValueError(
                f"Patch size must be square, got {patch_size}."
            )
        if num_frames <= 0:
            raise ValueError(
                f"num_frames must be positive, got {num_frames}."
            )
        if num_classes <= 0:
            raise ValueError(
                f"num_classes must be positive, got {num_classes}."
            )

        self.input_size = tuple(int(value) for value in input_size)
        self.num_frames = int(num_frames)
        self.num_classes = int(num_classes)
        self.embed_dim = int(embed_dim)
        self.context_length = int(text_context_length)
        self.vocab_size = int(text_vocab_size)

        self.use_discriminative_module = bool(use_discriminative_module)
        self.dr_layers = tuple(int(index) for index in dr_layers)
        self.share_discriminative_module = bool(
            share_discriminative_module
        )

        self.zeroshot_evaluation = bool(zeroshot_evaluation)
        self.use_fixed_text_branch = bool(use_fixed_text_branch)
        self.use_learnable_text_branch = bool(
            use_learnable_text_branch
            and use_text_prompt_learning
        )
        self.use_text_prompt_learning = (
            self.use_learnable_text_branch
        )
        self.use_text_mstff = bool(use_text_mstff)
        self.use_bie = bool(use_bie) and not self.zeroshot_evaluation
        self.use_scr = bool(use_scr) and not self.zeroshot_evaluation
        if self.use_bie and self.use_scr:
            raise ValueError(
                "use_bie and use_scr are mutually exclusive: both own the "
                "classifier-weight refinement stage and would fight over the "
                "same text prototypes. Enable exactly one of them."
            )
        self.use_aggregation_mixer = (
            bool(use_aggregation_mixer)
            and not self.zeroshot_evaluation
        )

        # CVP Block: supervised-only, mutually exclusive with the
        # discriminative temporal modules (both rewrite the token stream
        # between frozen blocks and would double-charge the positions).
        self.use_cvp_block = bool(use_cvp_block) and not self.zeroshot_evaluation
        # TGL v2 replaces the CVP residual path with CVPT-style per-layer
        # prompt injection: when v2 is on, the legacy CVP block is bypassed
        # entirely (its code stays reachable via the legacy flags).
        if use_tgl_v2:
            self.use_cvp_block = False
        self.cvp_corr_loss_weight = float(cvp_corr_loss_weight)
        if self.use_cvp_block and self.use_discriminative_module:
            raise ValueError(
                "use_cvp_block and use_discriminative_module are mutually "
                "exclusive: both modify the frozen-CLIP token stream between "
                "blocks. Enable exactly one of them."
            )
        self.fixed_text_template = fixed_text_template.strip()

        # Define optional adaptation attributes for both supervised and
        # zero-shot construction paths.
        self.text_mstff = None
        self.register_parameter("text_mstff_gate", None)
        self.register_parameter("text_fusion_logits", None)

        if not self.zeroshot_evaluation:
            if not (
                self.use_fixed_text_branch
                or self.use_learnable_text_branch
            ):
                raise ValueError(
                    "Supervised training requires at least one text branch."
                )
            if not text_prompt_classes_path:
                raise ValueError(
                    "text_prompt_classes_path is required in supervised mode."
                )

        # ------------------------------------------------------------------
        # Original OpenAI CLIP parameter tree.
        # ------------------------------------------------------------------
        self.visual = VisualTransformer(
            input_resolution=self.input_size[0],
            patch_size=patch_size[0],
            width=feature_dim,
            layers=num_layers,
            heads=num_heads,
            output_dim=self.embed_dim,
            num_frames=self.num_frames,
            use_discriminative_module=self.use_discriminative_module,
            dr_layers=self.dr_layers,
            share_discriminative_module=self.share_discriminative_module,
            dr_top_k=dr_top_k,
            dr_alpha=dr_alpha,
            dr_bottleneck=dr_bottleneck,
            dr_rwkv_layers=dr_rwkv_layers,
            dr_residual_scale_init=dr_residual_scale_init,
            dr_topk_temperature=dr_topk_temperature,
            dr_arch=dr_arch,
            dr_mamba_scope=dr_mamba_scope,
            dr_use_conv3d=dr_use_conv3d,
            dr_use_mamba=dr_use_mamba,
            dr_mamba_bidir=dr_mamba_bidir,
            dr_d_state=dr_d_state,
            dr_d_conv=dr_d_conv,
            dr_use_asa=dr_use_asa,
            dr_asa_heads=dr_asa_heads,
            dr_asa_time_offset_max=dr_asa_time_offset_max,
            dr_asa_space_offset_max=dr_asa_space_offset_max,
            dr_asa_window_size=dr_asa_window_size,
            use_aggregation_mixer=self.use_aggregation_mixer,
            aggregation_mixer_kernel=aggregation_mixer_kernel,
            aggregation_mixer_type=aggregation_mixer_type,
            aggregation_mixer_heads=aggregation_mixer_heads,
            aggregation_mixer_dropout=aggregation_mixer_dropout,
            use_cvp_block=self.use_cvp_block,
            cvp_insert_layer=cvp_insert_layer,
            cvp_topk=cvp_topk,
            cvp_window_size=cvp_window_size,
            cvp_num_prompts=cvp_num_prompts,
            cvp_bottleneck_dim=cvp_bottleneck_dim,
            cvp_num_heads=cvp_num_heads,
            cvp_gamma_min=cvp_gamma_min,
            cvp_gamma_max=cvp_gamma_max,
            cvp_gamma_init=cvp_gamma_init,
            cvp_beta_min=cvp_beta_min,
            cvp_beta_max=cvp_beta_max,
            cvp_beta_init=cvp_beta_init,
            cvp_selection_method=cvp_selection_method,
            cvp_motion_max_weight=cvp_motion_max_weight,
            cvp_current_context_attention=cvp_current_context_attention,
            cvp_corr_scope=cvp_corr_scope,
            cvp_spatial_bias_max=cvp_spatial_bias_max,
            cvp_spatial_bias_init=cvp_spatial_bias_init,
            use_mask_adapter=(
                use_mask_adapter and not self.zeroshot_evaluation
            ),
            use_layered_prompts=(
                use_layered_prompts and not self.zeroshot_evaluation
            ),
            mask_adapter_layers=mask_adapter_layers,
            mask_adapter_topk=mask_adapter_topk,
            mask_adapter_bottleneck=mask_adapter_bottleneck,
            mask_adapter_heads=mask_adapter_heads,
            mask_adapter_motion_max_weight=mask_adapter_motion_max_weight,
            num_video_prompts=num_video_prompts,
            prompt_cond_rank=prompt_cond_rank,
            prompt_insert_layers=prompt_insert_layers,
            use_tgl_adapter=use_tgl_adapter,
            tgl_adapter_dim=tgl_adapter_dim,
            tgl_num_layers=tgl_num_layers,
            tgl_num_heads=tgl_num_heads,
            tgl_dropout=tgl_dropout,
            tgl_gather_mode=tgl_gather_mode,
            tgl_local_tube_size=tgl_local_tube_size,
            tgl_visibility_threshold=tgl_visibility_threshold,
            tgl_max_scale=tgl_max_scale,
            tgl_max_fusion=tgl_max_fusion,
            tgl_up_proj_init=tgl_up_proj_init,
            tgl_insert_mode=tgl_insert_mode,
            tgl_insert_layers=tgl_insert_layers,
            tgl_num_prompts=tgl_num_prompts,
            tgl_patch_pool_baseline=tgl_patch_pool_baseline,
            tgl_gate_init=tgl_gate_init,
            tgl_readout=tgl_readout,
            tgl_readout_gate=tgl_readout_gate,
            use_tgl_v2=use_tgl_v2,
            tgl_v2_prompt_source_layer=tgl_v2_prompt_source_layer,
            tgl_v2_prompt_layers=tgl_v2_prompt_layers,
            tgl_v2_num_prompts=tgl_v2_num_prompts,
            tgl_v2_num_classes=num_classes,
            tgl_v2_recon_ratio=tgl_v2_recon_ratio,
            tgl_v2_readout_gate=tgl_v2_readout_gate,
        )

        self.transformer = Transformer(
            width=text_transformer_width,
            layers=text_transformer_layers,
            heads=text_transformer_heads,
            attn_mask=self.build_attention_mask(),
        )
        self.token_embedding = nn.Embedding(
            text_vocab_size,
            text_transformer_width,
        )
        self.positional_embedding = nn.Parameter(
            torch.empty(
                self.context_length,
                text_transformer_width,
            )
        )
        self.ln_final = LayerNorm(text_transformer_width)
        self.text_projection = nn.Parameter(
            torch.empty(
                text_transformer_width,
                self.embed_dim,
            )
        )
        self.logit_scale = nn.Parameter(
            torch.ones([], dtype=torch.float32)
            * np.log(1.0 / 0.07)
        )

        # Bidirectional Interpolation Enhancement. Both directions are always
        # on (V->T text interpolation + T->V video interpolation), identity
        # start is preserved by the zero-init gates (g_t == g_v == 0 at init,
        # optionally ramped by gate_warmup_steps), and the optional residual
        # transform MLPs (mlp1/mlp2) also start as identity.
        self.bie: Optional[BIE] = None
        self._last_gate_t: Optional[torch.Tensor] = None
        self._last_gate_v: Optional[torch.Tensor] = None
        self._last_gate_scale: Optional[float] = None
        if self.use_bie:
            self.bie = BIE(
                num_classes=self.num_classes,
                dim=self.embed_dim,
                p_temperature=bie_p_temperature,
                max_gate=bie_max_gate,
                gate_warmup_steps=bie_gate_warmup_steps,
                use_transform=bie_use_transform,
                transform_bottleneck=bie_transform_bottleneck,
                gate_bottleneck=bie_gate_bottleneck,
                enh_thr=bie_enh_thr,
                max_dist=bie_max_dist,
                max_dist_t=bie_max_dist_t,
                alpha_aux_lambda=bie_alpha_aux_lambda,
                external_logit_scale=self.logit_scale,
            )

        # Semantic Class-Relation Refinement (static text graph). Mutually
        # exclusive with BIE: both own the classifier-weight refinement stage
        # and would otherwise fight over the same text prototypes.
        self.scr: Optional[SemanticClassRelationRefinement] = None
        if self.use_scr:
            self.scr = SemanticClassRelationRefinement(
                dim=self.embed_dim,
                top_k=scr_top_k,
                max_scale=scr_max_scale,
            )

        self.initialize_parameters()

        # Load the official CLIP backbone exactly once, before constructing
        # prompt buffers from token embeddings.
        self.backbone_missing_keys: Tuple[str, ...] = ()
        self.backbone_unexpected_keys: Tuple[str, ...] = ()

        if backbone_path and _openai_state_dict is not None:
            raise ValueError(
                "Specify either backbone_path or _openai_state_dict, not both."
            )

        if _openai_state_dict is not None:
            self._load_openai_state_dict(
                _openai_state_dict
            )
        elif backbone_path:
            self._load_backbone(backbone_path)

        if self.zeroshot_evaluation:
            self._load_zeroshot_text_features(
                zeroshot_text_features_path
            )
        else:
            classnames = self._load_classnames(
                text_prompt_classes_path,
                naturalize=naturalize_class_names,
            )
            if len(classnames) != self.num_classes:
                raise ValueError(
                    "The class file and num_classes do not match: "
                    f"found={len(classnames)}, expected={self.num_classes}."
                )
            self.classnames = tuple(classnames)

            # Text_MSTFF is shared by whichever branch is selected for
            # enhancement. Its residual gate keeps initial text features close
            # to the original CLIP EOT representation.
            if self.use_text_mstff:
                self.text_mstff = TextMSTFF(
                    embed_dim=text_transformer_width,
                    max_weights=(0.6, 0.3, 0.5),
                    smooth_weight_constraint=True,
                )
                self.text_mstff_gate = nn.Parameter(
                    torch.tensor(
                        float(text_mstff_gate_init),
                        dtype=torch.float32,
                    )
                )
            # Fixed handcrafted branch: use actual formatted text rather than
            # passing a literal "{}" through TextPromptLearner.
            if self.use_fixed_text_branch:
                fixed_texts = [
                    self._format_fixed_prompt(
                        self.fixed_text_template,
                        class_name,
                    )
                    for class_name in classnames
                ]
                fixed_tokens = tokenize(
                    fixed_texts,
                    context_length=self.context_length,
                )
                self.register_buffer(
                    "fixed_tokenized_prompts",
                    fixed_tokens,
                    persistent=False,
                )

                # If another learnable branch exists, keep this branch as a
                # stable frozen anchor and cache its ordinary CLIP features.
                if self.use_learnable_text_branch:
                    with torch.no_grad():
                        fixed_features = self.encode_text(
                            fixed_tokens,
                            apply_mstff=False,
                        )
                    self.register_buffer(
                        "fixed_template_text_features",
                        fixed_features.float(),
                        persistent=False,
                    )

            if self.use_learnable_text_branch:
                self.prompt_learner = TextPromptLearner(
                    classnames=classnames,
                    text_model=self,
                    num_prompts=text_num_prompts,
                    prompts_init=text_prompt_init,
                    CSC=text_prompt_CSC,
                    ctx_pos=text_prompt_pos,
                )
                self.register_buffer(
                    "tokenized_prompts",
                    self.prompt_learner.tokenized_prompts,
                    persistent=False,
                )

            if (
                self.use_fixed_text_branch
                and self.use_learnable_text_branch
            ):
                self.text_fusion_logits = nn.Parameter(
                    torch.tensor(
                        [
                            float(text_fusion_fixed_logit),
                            float(text_fusion_learnable_logit),
                        ],
                        dtype=torch.float32,
                    )
                )
        self._configure_trainable_parameters()

    @property
    def dtype(self) -> torch.dtype:
        return self.visual.conv1.weight.dtype

    # ======================================================================
    # OpenAI CLIP initialization and loading
    # ======================================================================

    def initialize_parameters(self) -> None:
        nn.init.normal_(
            self.token_embedding.weight,
            std=0.02,
        )
        nn.init.normal_(
            self.positional_embedding,
            std=0.01,
        )

        projection_std = (
            self.transformer.width ** -0.5
            * (2 * self.transformer.layers) ** -0.5
        )
        attention_std = (
            self.transformer.width ** -0.5
        )
        fully_connected_std = (
            2 * self.transformer.width
        ) ** -0.5

        for block in self.transformer.resblocks:
            nn.init.normal_(
                block.attn.in_proj_weight,
                std=attention_std,
            )
            nn.init.normal_(
                block.attn.out_proj.weight,
                std=projection_std,
            )
            nn.init.normal_(
                block.mlp.c_fc.weight,
                std=fully_connected_std,
            )
            nn.init.normal_(
                block.mlp.c_proj.weight,
                std=projection_std,
            )

        nn.init.normal_(
            self.text_projection,
            std=self.transformer.width ** -0.5,
        )

    def build_attention_mask(self) -> torch.Tensor:
        mask = torch.empty(
            self.context_length,
            self.context_length,
        )
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    @staticmethod
    def _unwrap_state_dict(
        checkpoint: object,
    ) -> Dict[str, torch.Tensor]:
        if not isinstance(checkpoint, Mapping):
            raise TypeError(
                "Checkpoint must be a mapping, "
                f"got {type(checkpoint).__name__}."
            )

        state_dict: Mapping[str, object] = checkpoint
        for key in (
            "state_dict",
            "model_state_dict",
            "model",
        ):
            candidate = state_dict.get(key)
            if isinstance(candidate, Mapping):
                state_dict = candidate
                break

        result: Dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if not torch.is_tensor(value):
                continue

            clean_key = str(key)
            if clean_key.startswith("module."):
                clean_key = clean_key[7:]

            result[clean_key] = value.detach().cpu()

        if not result:
            raise ValueError(
                "No tensor parameters were found in the checkpoint."
            )

        return result

    @staticmethod
    def _clean_openai_state_dict(
        state_dict: Mapping[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        cleaned = {
            key: value
            for key, value in state_dict.items()
            if torch.is_tensor(value)
        }

        for metadata_key in (
            "input_resolution",
            "context_length",
            "vocab_size",
        ):
            cleaned.pop(metadata_key, None)

        return cleaned

    @staticmethod
    def _allowed_backbone_missing_key(key: str) -> bool:
        return (
            key == "visual.time_embed"
            or key.startswith("visual.discriminative_modules.")
            or key.startswith("visual.aggregation_mixer.")
            or key.startswith("bie.")
            or key.startswith("scr.")
            or key.startswith("visual.cvp_block.")
            or key.startswith("visual.mask_adapter.")
            or key.startswith("visual.layered_prompts.")
            or key.startswith("visual.tgl_adapter.")
            or key.startswith("visual.tgl_adapters.")
            or key.startswith("visual.tgl_readout.")
            or key.startswith("visual.v2_frame_tokens.")
            or key.startswith("visual.v2_prompt_gen.")
            or key.startswith("visual.v2_track_st.")
            or key.startswith("visual.v2_late_fusion.")
            or key == "visual.tgl_fusion_logit"
            or key == "visual.tgl_baseline_logit"
            or key == "visual.tgl_readout_logit"
            or (
                key.startswith("visual.transformer.resblocks.")
                and ".prompt_" in key
            )
        )

    def _load_openai_state_dict(
        self,
        state_dict: Mapping[str, torch.Tensor],
    ) -> None:
        cleaned = self._clean_openai_state_dict(state_dict)
        incompatible = self.load_state_dict(
            cleaned,
            strict=False,
        )

        self.backbone_missing_keys = tuple(
            incompatible.missing_keys
        )
        self.backbone_unexpected_keys = tuple(
            incompatible.unexpected_keys
        )

        bad_missing = [
            key
            for key in self.backbone_missing_keys
            if not self._allowed_backbone_missing_key(key)
        ]
        bad_unexpected = list(
            self.backbone_unexpected_keys
        )

        visual_loaded = sum(
            any(
                key.startswith(
                    "visual.transformer.resblocks."
                    f"{layer_index}."
                )
                for key in cleaned
            )
            for layer_index in range(
                len(self.visual.transformer.resblocks)
            )
        )
        text_loaded = sum(
            any(
                key.startswith(
                    "transformer.resblocks."
                    f"{layer_index}."
                )
                for key in cleaned
            )
            for layer_index in range(
                len(self.transformer.resblocks)
            )
        )

        print(
            "[VitaCLIP] OpenAI CLIP backbone audit: "
            f"visual_blocks={visual_loaded}/"
            f"{len(self.visual.transformer.resblocks)}, "
            f"text_blocks={text_loaded}/"
            f"{len(self.transformer.resblocks)}, "
            f"allowed_missing={len(self.backbone_missing_keys)}, "
            f"unexpected={len(self.backbone_unexpected_keys)}"
        )

        if bad_missing:
            raise RuntimeError(
                "OpenAI CLIP backbone parameters were not loaded:\n  "
                + "\n  ".join(bad_missing)
            )
        if bad_unexpected:
            raise RuntimeError(
                "Unexpected OpenAI CLIP parameters remain:\n  "
                + "\n  ".join(bad_unexpected)
            )
        if visual_loaded != len(
            self.visual.transformer.resblocks
        ):
            raise RuntimeError(
                "Not all visual Transformer blocks were loaded."
            )
        if text_loaded != len(
            self.transformer.resblocks
        ):
            raise RuntimeError(
                "Not all text Transformer blocks were loaded."
            )

    def _load_backbone(
        self,
        backbone_path: str,
    ) -> None:
        path = Path(backbone_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Backbone checkpoint not found: {path}"
            )

        try:
            jit_model = torch.jit.load(
                str(path),
                map_location="cpu",
            ).eval()
            state_dict = jit_model.state_dict()
            print(
                f"[VitaCLIP] loaded TorchScript backbone: {path}"
            )
        except RuntimeError:
            checkpoint = torch.load(
                str(path),
                map_location="cpu",
            )
            state_dict = self._unwrap_state_dict(
                checkpoint
            )
            print(
                f"[VitaCLIP] loaded state-dict backbone: {path}"
            )

        self._load_openai_state_dict(state_dict)

    # ======================================================================
    # Class names and prompts
    # ======================================================================

    @staticmethod
    def _naturalize_class_name(name: str) -> str:
        name = name.replace("_", " ").replace("-", " ")
        name = re.sub(
            r"(?<=[a-z0-9])(?=[A-Z])",
            " ",
            name,
        )
        name = re.sub(
            r"(?<=[A-Z])(?=[A-Z][a-z])",
            " ",
            name,
        )
        return " ".join(name.split())

    @classmethod
    def _load_classnames(
        cls,
        classes_path: str,
        naturalize: bool,
    ) -> List[str]:
        path = Path(classes_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Class file not found: {path}"
            )

        classnames: List[str] = []
        seen = set()

        for line_number, raw_line in enumerate(
            path.read_text(
                encoding="utf-8"
            ).splitlines(),
            start=1,
        ):
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split(maxsplit=1)
            if (
                len(parts) == 2
                and parts[0].isdigit()
            ):
                class_name = parts[1]
            else:
                class_name = line

            class_name = class_name.strip()
            if naturalize:
                class_name = (
                    cls._naturalize_class_name(
                        class_name
                    )
                )

            if not class_name:
                raise ValueError(
                    f"Empty class name at line {line_number}."
                )
            if class_name in seen:
                raise ValueError(
                    f"Duplicate class name: {class_name!r}."
                )

            classnames.append(class_name)
            seen.add(class_name)

        if not classnames:
            raise ValueError(
                f"No classes were found in {path}."
            )

        return classnames

    @staticmethod
    def _format_fixed_prompt(
        template: str,
        class_name: str,
    ) -> str:
        template = template.strip()

        if "{}" in template:
            prompt = template.format(class_name)
        else:
            prompt = f"{template} {class_name}"

        prompt = prompt.strip()
        if not prompt.endswith("."):
            prompt += "."
        return prompt

    # ======================================================================
    # Text encoding
    # ======================================================================

    def encode_text_tokens(
        self,
        prompt_embeddings: torch.Tensor,
        tokenized_prompts: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the original CLIP text Transformer through ``ln_final``."""
        if prompt_embeddings.ndim != 3:
            raise ValueError(
                "prompt_embeddings must be [N, L, D], "
                f"got {tuple(prompt_embeddings.shape)}."
            )
        if tokenized_prompts.ndim != 2:
            raise ValueError(
                "tokenized_prompts must be [N, L], "
                f"got {tuple(tokenized_prompts.shape)}."
            )
        if prompt_embeddings.shape[:2] != (
            tokenized_prompts.shape[0],
            tokenized_prompts.shape[1],
        ):
            raise ValueError(
                "Prompt embeddings and token IDs disagree: "
                f"embeddings={tuple(prompt_embeddings.shape)}, "
                f"tokens={tuple(tokenized_prompts.shape)}."
            )

        sequence_length = prompt_embeddings.shape[1]
        if sequence_length > self.context_length:
            raise ValueError(
                f"Text length {sequence_length} exceeds "
                f"context_length={self.context_length}."
            )

        x = prompt_embeddings.to(self.dtype)
        x = x + self.positional_embedding[
            :sequence_length
        ].to(device=x.device, dtype=x.dtype)

        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        token_features = self.ln_final(x).to(
            x.dtype
        )

        eot_indices = tokenized_prompts.argmax(
            dim=-1
        )
        batch_indices = torch.arange(
            token_features.shape[0],
            device=token_features.device,
        )
        eot_feature = token_features[
            batch_indices,
            eot_indices,
        ]

        return token_features, eot_feature

    def _apply_text_mstff(
        self,
        token_features: torch.Tensor,
        eot_feature: torch.Tensor,
        tokenized_prompts: torch.Tensor,
    ) -> torch.Tensor:
        if (
            not self.use_text_mstff
            or self.text_mstff is None
        ):
            return eot_feature

        fused_feature, _ = self.text_mstff(
            token_features,
            eot_feature,
            token_mask=tokenized_prompts.ne(0),
        )

        gate = torch.tanh(
            self.text_mstff_gate.float()
        ).to(
            device=eot_feature.device,
            dtype=eot_feature.dtype,
        )

        return (
            eot_feature
            + gate * (
                fused_feature.to(eot_feature.dtype)
                - eot_feature
            )
        )

    def encode_text_embeddings(
        self,
        prompt_embeddings: torch.Tensor,
        tokenized_prompts: torch.Tensor,
        *,
        apply_mstff: bool,
    ) -> torch.Tensor:
        token_features, eot_feature = (
            self.encode_text_tokens(
                prompt_embeddings,
                tokenized_prompts,
            )
        )

        if apply_mstff:
            eot_feature = self._apply_text_mstff(
                token_features,
                eot_feature,
                tokenized_prompts,
            )

        return eot_feature @ self.text_projection

    def encode_text(
        self,
        tokenized_text: torch.Tensor,
        *,
        apply_mstff: bool = False,
    ) -> torch.Tensor:
        embeddings = self.token_embedding(
            tokenized_text
        )
        return self.encode_text_embeddings(
            embeddings,
            tokenized_text,
            apply_mstff=apply_mstff,
        )

    def _fixed_text_features(
        self,
    ) -> torch.Tensor:
        if self.use_learnable_text_branch:
            return self.fixed_template_text_features

        # With no learnable branch, Text_MSTFF is allowed to enhance the
        # standard handcrafted template directly.
        return self.encode_text(
            self.fixed_tokenized_prompts,
            apply_mstff=self.use_text_mstff,
        )

    def _learnable_text_features(
        self,
    ) -> torch.Tensor:
        prompts = self.prompt_learner()
        return self.encode_text_embeddings(
            prompts,
            self.tokenized_prompts,
            apply_mstff=self.use_text_mstff,
        )

    def _build_supervised_text_features(
        self,
    ) -> torch.Tensor:
        branch_features: List[torch.Tensor] = []

        if self.use_fixed_text_branch:
            fixed = self._fixed_text_features()
            branch_features.append(
                F.normalize(
                    fixed.float(),
                    dim=-1,
                    eps=1e-6,
                )
            )

        if self.use_learnable_text_branch:
            learnable = (
                self._learnable_text_features()
            )
            branch_features.append(
                F.normalize(
                    learnable.float(),
                    dim=-1,
                    eps=1e-6,
                )
            )

        if len(branch_features) == 2:
            weights = F.softmax(
                self.text_fusion_logits.float(),
                dim=0,
            )
            text_features = (
                weights[0] * branch_features[0]
                + weights[1] * branch_features[1]
            )
        elif len(branch_features) == 1:
            text_features = branch_features[0]
        else:
            raise RuntimeError(
                "No supervised text branch is enabled."
            )

        return F.normalize(
            text_features,
            dim=-1,
            eps=1e-6,
        )

    # ======================================================================
    # Video/image encoding and forward
    # ======================================================================

    def encode_image(
        self,
        image: torch.Tensor,
    ) -> torch.Tensor:
        """Original frame-wise CLIP image encoding for alignment tests."""
        return self.visual.encode_frames(
            image.to(self.dtype),
            num_frames=1,
            apply_temporal_modules=False,
        )

    def encode_video(
        self,
        video: torch.Tensor,
        tracks: Optional[torch.Tensor] = None,
        visibility: Optional[torch.Tensor] = None,
        birth_frame: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ):
        if video.ndim != 5:
            raise ValueError(
                "Expected video [B, 3, T, H, W], "
                f"got {tuple(video.shape)}."
            )

        batch_size, channels, frames, height, width = (
            video.shape
        )

        if channels != 3:
            raise ValueError(
                f"Expected RGB input, got C={channels}."
            )
        if frames != self.num_frames:
            raise ValueError(
                f"Expected T={self.num_frames}, got T={frames}."
            )
        if (height, width) != self.input_size:
            raise ValueError(
                f"Expected spatial size {self.input_size}, "
                f"got {(height, width)}."
            )

        images = video.permute(
            0,
            2,
            1,
            3,
            4,
        ).reshape(
            batch_size * frames,
            channels,
            height,
            width,
        )

        feats = self.visual(
            images.to(self.dtype),
            num_frames=frames,
            apply_temporal_modules=True,
            tracks=tracks,
            visibility=visibility,
            birth_frame=birth_frame,
            labels=labels,
        )
        if return_aux:
            return feats, dict(getattr(self.visual, "_last_v2_aux", {}) or {})
        return feats

    def forward(
        self,
        video: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        bie_step: Optional[int] = None,
        tracks: Optional[torch.Tensor] = None,
        visibility: Optional[torch.Tensor] = None,
        birth_frame: Optional[torch.Tensor] = None,
    ):
        video_features = self.encode_video(
            video,
            tracks=tracks,
            visibility=visibility,
            birth_frame=birth_frame,
            labels=labels,
        )

        if self.zeroshot_evaluation:
            text_features = self.text_features.to(
                device=video_features.device,
                dtype=torch.float32,
            )
            text_features = F.normalize(
                text_features,
                dim=-1,
                eps=1e-6,
            )
        else:
            text_features = (
                self._build_supervised_text_features()
            )

        if text_features.shape != (
            self.num_classes,
            self.embed_dim,
        ):
            raise RuntimeError(
                "Text feature shape mismatch: "
                f"expected {(self.num_classes, self.embed_dim)}, "
                f"got {tuple(text_features.shape)}."
            )

        video_features = F.normalize(
            video_features.float(),
            dim=-1,
            eps=1e-6,
        )
        text_features = text_features.to(
            video_features.device
        )

        # Bidirectional Interpolation Enhancement. When enabled, BIE owns the
        # final logits (it reuses the shared logit_scale); it has no
        # auxiliary losses, the training loop uses the CE loss only. The
        # gate/scale monitors are refreshed every forward for logging.
        self._last_gate_t = None
        self._last_gate_v = None
        self._last_gate_scale = None
        self._last_consistency_loss = None
        self._last_alpha_aux_loss = None
        self._last_cvp_corr_loss = getattr(
            self.visual, "_last_cvp_corr_loss", None
        )
        self._last_cvp_stats = getattr(
            self.visual, "_last_cvp_stats", None
        )
        self._last_layered_stats = getattr(
            self.visual, "_last_layered_stats", {}
        )
        if self.bie is not None:
            out = self.bie(
                V=video_features,
                Q=text_features,
                y=labels,
                step=bie_step,
            )
            self._last_gate_t = out.get("gate_t")
            self._last_gate_v = out.get("gate_v")
            self._last_gate_scale = out.get("gate_scale")
            self._last_consistency_loss = out.get("consistency_loss")
            self._last_alpha_aux_loss = out.get("alpha_aux_loss")
            return out["logits"]

        # Semantic Class-Relation Refinement: refine the text prototypes via
        # their static relation graph before the cosine matmul. Identity start
        # (lambda == 0 at init) keeps step 0 identical to the baseline.
        if self.scr is not None:
            text_features = self.scr(text_features)

        logit_scale = self.logit_scale.float().exp()
        return (
            logit_scale
            * video_features
            @ text_features.t()
        )

    # ======================================================================
    # Freezing and checkpoints
    # ======================================================================

    def _configure_trainable_parameters(
        self,
    ) -> None:
        # Freeze every original CLIP parameter and every auxiliary module.
        for parameter in self.parameters():
            parameter.requires_grad = False

        # New visual parameters.
        self.visual.time_embed.requires_grad = True
        for parameter in (
            self.visual.discriminative_modules.parameters()
        ):
            parameter.requires_grad = True
        if self.visual.aggregation_mixer is not None:
            for parameter in (
                self.visual.aggregation_mixer.parameters()
            ):
                parameter.requires_grad = True

        # New text parameters.
        if (
            self.use_text_mstff
            and getattr(self, "text_mstff", None) is not None
        ):
            for parameter in (
                self.text_mstff.parameters()
            ):
                parameter.requires_grad = True
            self.text_mstff_gate.requires_grad = True

        if hasattr(self, "prompt_learner"):
            for parameter in (
                self.prompt_learner.parameters()
            ):
                parameter.requires_grad = True

        if getattr(self, "text_fusion_logits", None) is not None:
            self.text_fusion_logits.requires_grad = True

        # Bidirectional Interpolation Enhancement parameters.
        if self.bie is not None:
            for parameter in self.bie.parameters():
                parameter.requires_grad = True

        # Semantic Class-Relation Refinement parameters.
        if self.scr is not None:
            for parameter in self.scr.parameters():
                parameter.requires_grad = True

        # Correspondence-Guided Video Prompt Block parameters.
        if self.visual.cvp_block is not None:
            for parameter in self.visual.cvp_block.parameters():
                parameter.requires_grad = True

        # Layered v3 adaptation parameters (mask adapter + layered prompts).
        if getattr(self.visual, "mask_adapter", None) is not None:
            for parameter in self.visual.mask_adapter.parameters():
                parameter.requires_grad = True
        if getattr(self.visual, "layered_prompts", None) is not None:
            for parameter in self.visual.layered_prompts.parameters():
                parameter.requires_grad = True

        # TGL trajectory adapter parameters (per-insertion adapters +
        # optional zero-init baseline gate). The frozen CLIP backbone stays
        # untouched; only the TGL residual path trains (identity start
        # preserved by the warm-gate + bounded write-back design).
        if getattr(self.visual, "tgl_adapters", None) is not None:
            for parameter in self.visual.tgl_adapters.parameters():
                parameter.requires_grad = True
        # TGL v2 trainable modules (Track-ST / Trajectory CLS / prompt
        # generator / late fusion) — the frozen CLIP backbone stays untouched.
        if getattr(self.visual, "v2_track_st", None) is not None:
            for parameter in self.visual.v2_track_st.parameters():
                parameter.requires_grad = True
        if getattr(self.visual, "v2_prompt_gen", None) is not None:
            for parameter in self.visual.v2_prompt_gen.parameters():
                parameter.requires_grad = True
        if getattr(self.visual, "v2_late_fusion", None) is not None:
            for parameter in self.visual.v2_late_fusion.parameters():
                parameter.requires_grad = True
        # Recurrent four-slot prompts use only the per-block alpha gate;
        # prompt content is generated by `v2_prompt_gen` above.
        for blk in self.visual.transformer.resblocks:
            if getattr(blk, "prompt_alpha", None) is not None:
                blk.prompt_alpha.requires_grad = True
        if getattr(self.visual, "tgl_baseline_logit", None) is not None:
            self.visual.tgl_baseline_logit.requires_grad = True

        # CLIP similarity temperature remains trainable.
        self.logit_scale.requires_grad = True

    def trainable_parameter_names(
        self,
    ) -> Tuple[str, ...]:
        return tuple(
            name
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        )

    def _load_zeroshot_text_features(
        self,
        features_path: str,
    ) -> None:
        if not features_path:
            raise ValueError(
                "zeroshot_text_features_path is required."
            )

        path = Path(features_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Zero-shot features not found: {path}"
            )

        features = torch.load(
            str(path),
            map_location="cpu",
        )
        if isinstance(features, Mapping):
            for key in (
                "text_features",
                "features",
            ):
                if key in features:
                    features = features[key]
                    break

        if (
            not torch.is_tensor(features)
            or features.ndim != 2
        ):
            raise ValueError(
                "Zero-shot text features must be "
                "[num_classes, embed_dim]."
            )
        if tuple(features.shape) != (
            self.num_classes,
            self.embed_dim,
        ):
            raise ValueError(
                "Zero-shot text feature shape mismatch: "
                f"got {tuple(features.shape)}."
            )

        self.register_buffer(
            "text_features",
            features.float(),
            persistent=False,
        )


def _infer_openai_clip_configuration(
    state_dict: Mapping[str, torch.Tensor],
) -> Dict[str, int]:
    """Infer the OpenAI CLIP architecture from its state dict."""
    vision_width = int(
        state_dict["visual.conv1.weight"].shape[0]
    )
    vision_patch_size = int(
        state_dict["visual.conv1.weight"].shape[-1]
    )

    visual_block_indices = {
        int(key.split(".")[3])
        for key in state_dict
        if key.startswith(
            "visual.transformer.resblocks."
        )
    }
    vision_layers = len(visual_block_indices)

    vision_grid_size = round(
        (
            int(
                state_dict[
                    "visual.positional_embedding"
                ].shape[0]
            )
            - 1
        )
        ** 0.5
    )
    image_resolution = (
        vision_patch_size * vision_grid_size
    )

    embed_dim = int(
        state_dict["text_projection"].shape[1]
    )
    context_length = int(
        state_dict["positional_embedding"].shape[0]
    )
    vocab_size = int(
        state_dict["token_embedding.weight"].shape[0]
    )
    transformer_width = int(
        state_dict["ln_final.weight"].shape[0]
    )
    transformer_heads = transformer_width // 64

    text_block_indices = {
        int(key.split(".")[2])
        for key in state_dict
        if key.startswith(
            "transformer.resblocks."
        )
    }
    transformer_layers = len(text_block_indices)

    return {
        "input_resolution": image_resolution,
        "vision_width": vision_width,
        "vision_layers": vision_layers,
        "vision_patch_size": vision_patch_size,
        "embed_dim": embed_dim,
        "context_length": context_length,
        "vocab_size": vocab_size,
        "transformer_width": transformer_width,
        "transformer_heads": transformer_heads,
        "transformer_layers": transformer_layers,
    }


def build_model(
    state_dict: Mapping[str, torch.Tensor],
    **kwargs: object,
) -> VitaCLIP:
    """Build an aligned VitaCLIP directly from an OpenAI CLIP state dict.

    This mirrors the ActionCLIP/MIP-CLIP ``build_model`` entry point while
    retaining the current training script's VitaCLIP class.
    """
    configuration = _infer_openai_clip_configuration(
        state_dict
    )

    constructor_kwargs = dict(kwargs)
    constructor_kwargs.update(
        {
            "backbone_path": "",
            "input_size": (
                configuration["input_resolution"],
                configuration["input_resolution"],
            ),
            "feature_dim": configuration[
                "vision_width"
            ],
            "patch_size": (
                configuration["vision_patch_size"],
                configuration["vision_patch_size"],
            ),
            "num_heads": (
                configuration["vision_width"] // 64
            ),
            "num_layers": configuration[
                "vision_layers"
            ],
            "embed_dim": configuration["embed_dim"],
            "text_context_length": configuration[
                "context_length"
            ],
            "text_vocab_size": configuration[
                "vocab_size"
            ],
            "text_transformer_width": configuration[
                "transformer_width"
            ],
            "text_transformer_heads": configuration[
                "transformer_heads"
            ],
            "text_transformer_layers": configuration[
                "transformer_layers"
            ],
        }
    )

    constructor_kwargs["_openai_state_dict"] = state_dict
    return VitaCLIP(**constructor_kwargs)


def load_pretrained_vitaclip(
    backbone_path: str,
    **kwargs: object,
) -> VitaCLIP:
    """Load an OpenAI TorchScript checkpoint and call ``build_model``."""
    path = Path(backbone_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Backbone checkpoint not found: {path}"
        )

    try:
        jit_model = torch.jit.load(
            str(path),
            map_location="cpu",
        ).eval()
        state_dict = jit_model.state_dict()
    except RuntimeError:
        checkpoint = torch.load(
            str(path),
            map_location="cpu",
        )
        state_dict = VitaCLIP._unwrap_state_dict(
            checkpoint
        )

    return build_model(
        state_dict,
        **kwargs,
    )


def convert_weights(model: nn.Module) -> None:
    """Convert applicable CLIP weights to FP16, matching OpenAI CLIP."""

    def _convert(module: nn.Module) -> None:
        if isinstance(
            module,
            (nn.Conv1d, nn.Conv2d, nn.Linear),
        ):
            module.weight.data = (
                module.weight.data.half()
            )
            if module.bias is not None:
                module.bias.data = (
                    module.bias.data.half()
                )

        if isinstance(
            module,
            nn.MultiheadAttention,
        ):
            for attribute in (
                "in_proj_weight",
                "in_proj_bias",
                "bias_k",
                "bias_v",
            ):
                tensor = getattr(
                    module,
                    attribute,
                    None,
                )
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for attribute in (
            "text_projection",
            "proj",
        ):
            tensor = getattr(
                module,
                attribute,
                None,
            )
            if tensor is not None:
                tensor.data = tensor.data.half()

    model.apply(_convert)