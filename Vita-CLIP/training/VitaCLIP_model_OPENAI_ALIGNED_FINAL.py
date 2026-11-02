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
3. Insert the custom Frequency Adapter only after complete visual blocks
   3, 7 and 11 (zero-based indices by default).
4. Insert Text_MSTFF after text ``ln_final`` and before ``text_projection``.
5. Keep the original CLIP encoders frozen while allowing the new temporal and
   text-adaptation parameters to train.
6. Retain the old constructor arguments used by the current training script.

The external modules are intentionally kept independent:
    VitaCLIP_frequency_adapter_simple.py
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
from VitaCLIP_discriminative_rwkv import VitaCLIPDiscriminativeRWKV
from VitaCLIP_frequency_adapter_simple import VitaCLIPFrequencyAdapter
from VitaCLIP_text_encoder import TextPromptLearner, tokenize


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

    def attention(
        self,
        x: torch.Tensor,
        *,
        return_weights: bool = False,
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
                average_attn_weights=True,
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
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if return_attention:
            attn_out, attn_weights = self.attention(
                self.ln_1(x), return_weights=True
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


class VisualTransformer(nn.Module):
    """OpenAI CLIP ViT with external temporal frequency adapters.

    Original CLIP parameters retain their exact names. New parameters are
    registered only under:
        time_embed
        frequency_adapters
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
        use_frequency_adapter: bool = True,
        frequency_layers: Sequence[int] = (3, 7, 11),
        share_frequency_adapter: bool = False,
        frequency_bottleneck: int = 192,
        frequency_residual_scale_init: float = 0.01,
        use_discriminative_module: bool = False,
        dr_layers: Sequence[int] = (3, 7, 11),
        share_discriminative_module: bool = False,
        dr_top_k: int = 6,
        dr_alpha: float = 0.4,
        dr_bottleneck: int = 192,
        dr_rwkv_layers: int = 1,
        dr_residual_scale_init: float = 0.01,
    ) -> None:
        super().__init__()

        self.input_resolution = int(input_resolution)
        self.output_dim = int(output_dim)
        self.width = int(width)
        self.num_frames = int(num_frames)
        self.use_frequency_adapter = bool(use_frequency_adapter)
        self.frequency_layers = self._validate_frequency_layers(
            frequency_layers,
            layers,
        )
        self.share_frequency_adapter = bool(share_frequency_adapter)
        self.use_discriminative_module = bool(use_discriminative_module)
        self.dr_layers = self._validate_frequency_layers(
            dr_layers,
            layers,
        )
        self.share_discriminative_module = bool(share_discriminative_module)

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

        # ------------------------------------------------------------------
        # New video parameters.
        # ``time_embed`` is exactly zero at initialization, so disabling the
        # frequency adapters reproduces frame-wise OpenAI CLIP.
        # ------------------------------------------------------------------
        self.time_embed = nn.Parameter(
            torch.zeros(num_frames, width)
        )

        num_spatial_patches = grid_size * grid_size

        if self.use_frequency_adapter:
            if self.share_frequency_adapter:
                self.frequency_adapters = nn.ModuleDict(
                    {
                        "shared": VitaCLIPFrequencyAdapter(
                            dim=width,
                            bottleneck=frequency_bottleneck,
                            num_spatial_patches=num_spatial_patches,
                            residual_scale_init=frequency_residual_scale_init,
                        )
                    }
                )
            else:
                self.frequency_adapters = nn.ModuleDict(
                    {
                        str(layer_index): VitaCLIPFrequencyAdapter(
                            dim=width,
                            bottleneck=frequency_bottleneck,
                            num_spatial_patches=num_spatial_patches,
                            residual_scale_init=frequency_residual_scale_init,
                        )
                        for layer_index in self.frequency_layers
                    }
                )
        else:
            self.frequency_adapters = nn.ModuleDict()

        # ------------------------------------------------------------------
        # Discriminative region + RWKV temporal module.
        # ------------------------------------------------------------------
        if self.use_discriminative_module:
            if self.share_discriminative_module:
                self.discriminative_modules = nn.ModuleDict(
                    {
                        "shared": VitaCLIPDiscriminativeRWKV(
                            dim=width,
                            bottleneck=dr_bottleneck,
                            num_spatial_patches=num_spatial_patches,
                            top_k=dr_top_k,
                            alpha=dr_alpha,
                            rwkv_layers=dr_rwkv_layers,
                            residual_scale_init=dr_residual_scale_init,
                        )
                    }
                )
            else:
                self.discriminative_modules = nn.ModuleDict(
                    {
                        str(layer_index): VitaCLIPDiscriminativeRWKV(
                            dim=width,
                            bottleneck=dr_bottleneck,
                            num_spatial_patches=num_spatial_patches,
                            top_k=dr_top_k,
                            alpha=dr_alpha,
                            rwkv_layers=dr_rwkv_layers,
                            residual_scale_init=dr_residual_scale_init,
                        )
                        for layer_index in self.dr_layers
                    }
                )
        else:
            self.discriminative_modules = nn.ModuleDict()

    @staticmethod
    def _validate_frequency_layers(
        frequency_layers: Sequence[int],
        num_layers: int,
    ) -> Tuple[int, ...]:
        layers = tuple(int(index) for index in frequency_layers)

        if len(layers) != len(set(layers)):
            raise ValueError(
                f"frequency_layers contains duplicates: {layers}."
            )

        invalid = [
            index
            for index in layers
            if index < 0 or index >= num_layers
        ]
        if invalid:
            raise ValueError(
                "frequency_layers uses zero-based block indices in "
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

    def _apply_frequency_adapter(
        self,
        x: torch.Tensor,
        layer_index: int,
        num_frames: int,
    ) -> torch.Tensor:
        """Apply one adapter to tokens in [N, B*T, D] layout."""
        tokens = x.permute(1, 0, 2).contiguous()

        adapter_key = (
            "shared"
            if self.share_frequency_adapter
            else str(layer_index)
        )
        adapter = self.frequency_adapters[adapter_key]

        tokens = adapter(
            tokens,
            num_frames=num_frames,
            patch_start=1,
        )

        return tokens.permute(1, 0, 2).contiguous()

    def encode_frames(
        self,
        images: torch.Tensor,
        *,
        num_frames: int,
        apply_temporal_modules: bool = True,
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

        batch_size = images.shape[0] // num_frames

        # OpenAI CLIP patch embedding.
        x = self.conv1(images)
        x = x.reshape(
            x.shape[0],
            x.shape[1],
            -1,
        )
        x = x.permute(0, 2, 1)

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

        for layer_index, block in enumerate(
            self.transformer.resblocks
        ):
            use_dr = (
                apply_temporal_modules
                and self.use_discriminative_module
                and layer_index in self.dr_layers
                and num_frames >= 4
            )
            if use_dr:
                # Get the real CLS -> patch attention weights from this block
                # (default path is unchanged: only enabled when the module is
                # active, and the block output itself is identical).
                x, attn_weights = block(x, return_attention=True)
                # attn_weights: [B*T, N, N] (batch-first MultiheadAttention
                # output in PyTorch 2.x), CLS row -> patches: [B*T, N_p].
                cls_to_patch = attn_weights[:, 0, 1:].contiguous()

                adapter_key = (
                    "shared"
                    if self.share_discriminative_module
                    else str(layer_index)
                )
                tokens = x.permute(1, 0, 2).contiguous()
                tokens = self.discriminative_modules[adapter_key](
                    tokens,
                    cls_to_patch,
                    num_frames=num_frames,
                    patch_start=1,
                )
                x = tokens.permute(1, 0, 2).contiguous()
            else:
                x = block(x)

            if (
                apply_temporal_modules
                and self.use_frequency_adapter
                and layer_index in self.frequency_layers
                and num_frames >= 4
            ):
                x = self._apply_frequency_adapter(
                    x,
                    layer_index=layer_index,
                    num_frames=num_frames,
                )

        x = x.permute(1, 0, 2)
        frame_features = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            frame_features = frame_features @ self.proj

        return frame_features

    def forward(
        self,
        images: torch.Tensor,
        *,
        num_frames: Optional[int] = None,
        apply_temporal_modules: bool = True,
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
        )

        batch_size = frame_features.shape[0] // temporal_length
        frame_features = frame_features.reshape(
            batch_size,
            temporal_length,
            self.output_dim,
        )
        return frame_features.mean(dim=1)


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
        # Frequency adapter
        use_frequency_adapter: bool = True,
        frequency_layers: Sequence[int] = (3, 7, 11),
        share_frequency_adapter: bool = False,
        frequency_bottleneck: int = 192,
        frequency_residual_scale_init: float = 0.01,
        # Discriminative region + RWKV temporal module
        use_discriminative_module: bool = False,
        dr_layers: Sequence[int] = (3, 7, 11),
        share_discriminative_module: bool = False,
        dr_top_k: int = 6,
        dr_alpha: float = 0.4,
        dr_bottleneck: int = 192,
        dr_rwkv_layers: int = 1,
        dr_residual_scale_init: float = 0.01,
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

        self.use_frequency_adapter = bool(use_frequency_adapter)
        self.frequency_layers = tuple(
            int(index) for index in frequency_layers
        )
        self.share_frequency_adapter = bool(
            share_frequency_adapter
        )
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
            use_frequency_adapter=self.use_frequency_adapter,
            frequency_layers=self.frequency_layers,
            share_frequency_adapter=self.share_frequency_adapter,
            frequency_bottleneck=frequency_bottleneck,
            frequency_residual_scale_init=(
                frequency_residual_scale_init
            ),
            use_discriminative_module=self.use_discriminative_module,
            dr_layers=self.dr_layers,
            share_discriminative_module=self.share_discriminative_module,
            dr_top_k=dr_top_k,
            dr_alpha=dr_alpha,
            dr_bottleneck=dr_bottleneck,
            dr_rwkv_layers=dr_rwkv_layers,
            dr_residual_scale_init=dr_residual_scale_init,
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
            or key.startswith("visual.frequency_adapters.")
            or key.startswith("visual.discriminative_modules.")
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
    ) -> torch.Tensor:
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

        return self.visual(
            images.to(self.dtype),
            num_frames=frames,
            apply_temporal_modules=True,
        )

    def forward(
        self,
        video: torch.Tensor,
    ) -> torch.Tensor:
        video_features = self.encode_video(video)

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
            self.visual.frequency_adapters.parameters()
        ):
            parameter.requires_grad = True
        for parameter in (
            self.visual.discriminative_modules.parameters()
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