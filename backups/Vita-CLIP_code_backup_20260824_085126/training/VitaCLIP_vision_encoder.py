#!/usr/bin/env python
"""CLIP visual encoder for fully supervised UCF101 action recognition.

Default path:
    [B, 3, 8, 224, 224]
    -> [B*8, 196, 768] patch tokens
    -> add CLS: [B*8, 197, 768]
    -> add spatial and temporal position embeddings
    -> 12 frozen CLIP Transformer blocks
    -> independent frequency adapters after blocks 3, 7 and 11
    -> frame CLS features [B, 8, 512]
    -> temporal mean pooling [B, 512]

The original Vita-CLIP global/local/summary prompt mechanisms are retained
for ablation, but they are disabled by default by the top-level model.
"""

from __future__ import annotations

import math
from functools import reduce
from operator import mul
from typing import Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from VitaCLIP_frequency_adapter_simple import VitaCLIPFrequencyAdapter
from VitaCLIP_vision_encoder_utils import (
    ImagePatchEmbed2D,
    LayerNorm,
    QuickGELU,
    TransformerEncoderLayer,
)


class CLIPVisionEncoder(nn.Module):

    def __init__(
        self,
        # data shape
        input_size: Tuple[int, int] = (224, 224),
        num_frames: int = 8,
        # model definition
        feature_dim: int = 768,
        patch_size: Tuple[int, int] = (16, 16),
        num_heads: int = 12,
        num_layers: int = 12,
        mlp_factor: float = 4.0,
        act: nn.Module = QuickGELU,
        embed_dim: int = 512,
        # original Vita-CLIP visual prompts
        use_summary_token: bool = False,
        use_local_prompts: bool = False,
        use_global_prompts: bool = False,
        num_global_prompts: int = 8,
        # frequency adapter
        use_frequency_adapter: bool = True,
        frequency_layers: Sequence[int] = (3, 7, 11),
        share_frequency_adapter: bool = False,
        frequency_bottleneck: int = 192,
    ) -> None:
        super().__init__()

        self.input_size = tuple(int(v) for v in input_size)
        self.patch_size = tuple(int(v) for v in patch_size)
        self.num_frames = int(num_frames)
        self.feature_dim = int(feature_dim)
        self.embed_dim = int(embed_dim)
        self.num_layers = int(num_layers)

        if self.num_frames <= 0:
            raise ValueError(
                f"num_frames must be positive, got {self.num_frames}."
            )
        if self.num_layers <= 0:
            raise ValueError(
                f"num_layers must be positive, got {self.num_layers}."
            )
        if len(self.input_size) != 2 or len(self.patch_size) != 2:
            raise ValueError(
                "input_size and patch_size must both contain two integers."
            )
        if any(size <= 0 for size in self.input_size + self.patch_size):
            raise ValueError(
                "input_size and patch_size values must be positive."
            )
        if any(
            image_size % patch_size_value != 0
            for image_size, patch_size_value in zip(
                self.input_size,
                self.patch_size,
            )
        ):
            raise ValueError(
                f"input_size={self.input_size} must be divisible by "
                f"patch_size={self.patch_size}."
            )

        # ------------------------------------------------------------------
        # Patch embedding and CLIP tokens
        # ------------------------------------------------------------------
        self.patch_embed = ImagePatchEmbed2D(
            img_size=self.input_size[0],
            patch_size=self.patch_size[0],
            in_chans=3,
            embed_dim=self.feature_dim,
        )

        self.num_patch_tokens = int(
            np.prod(
                [
                    image_size // patch_size_value
                    for image_size, patch_size_value in zip(
                        self.input_size,
                        self.patch_size,
                    )
                ]
            )
        )
        self.num_tokens = self.num_patch_tokens + 1

        # Keep the original attribute name for old checkpoints/code.
        # In the original project this value includes CLS.
        self.num_patches = self.num_tokens

        self.cls_token = nn.Parameter(
            torch.zeros(self.feature_dim)
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(self.num_tokens, self.feature_dim)
        )
        self.time_embed = nn.Parameter(
            torch.zeros(self.num_frames, self.feature_dim)
        )

        # ------------------------------------------------------------------
        # CLIP visual Transformer
        # ------------------------------------------------------------------
        self.blocks = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    in_feature_dim=self.feature_dim,
                    qkv_dim=self.feature_dim,
                    num_heads=num_heads,
                    mlp_factor=mlp_factor,
                    act=act,
                    use_summary_token=use_summary_token,
                    use_local_prompts=use_local_prompts,
                    num_frames=self.num_frames,
                    patch_size=self.patch_size,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.ln_pre = LayerNorm(self.feature_dim)
        self.ln_post = LayerNorm(self.feature_dim)

        projection_scale = self.feature_dim ** -0.5
        self.proj = nn.Parameter(
            projection_scale
            * torch.randn(self.feature_dim, self.embed_dim)
        )

        # ------------------------------------------------------------------
        # Original global visual prompts
        # ------------------------------------------------------------------
        self.use_global_prompts = bool(use_global_prompts)
        self.num_global_prompts = int(num_global_prompts)

        if self.use_global_prompts:
            if self.num_global_prompts <= 0:
                raise ValueError(
                    "num_global_prompts must be positive when "
                    "use_global_prompts=True."
                )

            self.global_prompts = nn.Parameter(
                torch.zeros(
                    self.num_layers,
                    self.num_global_prompts,
                    self.feature_dim,
                )
            )
            self._initialize_global_prompts(
                self.patch_size,
                self.feature_dim,
            )

        # ------------------------------------------------------------------
        # Frequency adapters
        # ------------------------------------------------------------------
        self.use_frequency_adapter = bool(use_frequency_adapter)
        self.share_frequency_adapter = bool(share_frequency_adapter)
        self.frequency_layers = self._validate_frequency_layers(
            frequency_layers,
            self.num_layers,
        )

        if self.use_frequency_adapter:
            if self.share_frequency_adapter:
                self.frequency_adapter_shared = (
                    VitaCLIPFrequencyAdapter(
                        dim=self.feature_dim,
                        bottleneck=frequency_bottleneck,
                    )
                )
            else:
                self.frequency_adapters = nn.ModuleDict(
                    {
                        str(layer_index): VitaCLIPFrequencyAdapter(
                            dim=self.feature_dim,
                            bottleneck=frequency_bottleneck,
                        )
                        for layer_index in self.frequency_layers
                    }
                )

        self._initialize_weights()

    # ------------------------------------------------------------------
    # Initialization and validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_frequency_layers(
        frequency_layers: Sequence[int],
        num_layers: int,
    ) -> Tuple[int, ...]:
        layers = tuple(int(layer) for layer in frequency_layers)

        if len(set(layers)) != len(layers):
            raise ValueError(
                f"frequency_layers contains duplicates: {layers}."
            )

        invalid_layers = [
            layer
            for layer in layers
            if layer < 0 or layer >= num_layers
        ]
        if invalid_layers:
            raise ValueError(
                f"frequency_layers={invalid_layers} are outside the valid "
                f"zero-based range [0, {num_layers - 1}]."
            )

        return layers

    def _initialize_weights(self) -> None:
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.time_embed, std=0.02)

    def _initialize_global_prompts(
        self,
        patch_size: Tuple[int, int],
        prompt_dim: int,
    ) -> None:
        value = math.sqrt(
            6.0
            / float(
                3 * reduce(mul, patch_size, 1)
                + prompt_dim
            )
        )
        nn.init.uniform_(
            self.global_prompts.data,
            -value,
            value,
        )

    # ------------------------------------------------------------------
    # Temporal position encoding
    # ------------------------------------------------------------------

    def temporal_encoding(
        self,
        x: torch.Tensor,
        num_frames: int,
        batch_size: int,
    ) -> torch.Tensor:
        """Add a temporal position embedding to every token trajectory.

        Input/output:
            [B*T, N, D] -> [B*T, N, D]
        """
        if x.ndim != 3:
            raise ValueError(
                "temporal_encoding expects [B*T, N, D], "
                f"got {tuple(x.shape)}."
            )

        x = rearrange(
            x,
            "(b t) n d -> (b n) t d",
            b=batch_size,
            t=num_frames,
        )

        if num_frames == self.time_embed.shape[0]:
            temporal_position = self.time_embed
        else:
            # Resize only when inference/training uses a different T.
            temporal_position = self.time_embed
            temporal_position = temporal_position.unsqueeze(0).transpose(1, 2)
            temporal_position = F.interpolate(
                temporal_position,
                size=num_frames,
                mode="linear",
                align_corners=False,
            )
            temporal_position = temporal_position.transpose(1, 2).squeeze(0)

        x = x + temporal_position.to(
            device=x.device,
            dtype=x.dtype,
        )

        x = rearrange(
            x,
            "(b n) t d -> (b t) n d",
            b=batch_size,
            t=num_frames,
        )
        return x

    # ------------------------------------------------------------------
    # Frequency adapter
    # ------------------------------------------------------------------

    def _apply_frequency_adapter(
        self,
        x: torch.Tensor,
        layer_index: int,
        num_frames: int,
    ) -> torch.Tensor:
        """Apply the adapter after a complete visual Transformer block.

        Standard token layout:
            [CLS] + [196 patch tokens]

        Input/output:
            [B*T, 197, 768] -> [B*T, 197, 768]
        """
        input_shape = x.shape

        if self.share_frequency_adapter:
            output = self.frequency_adapter_shared(
                x,
                num_frames=num_frames,
                patch_start=1,
            )
        else:
            output = self.frequency_adapters[str(layer_index)](
                x,
                num_frames=num_frames,
                patch_start=1,
            )

        if output.shape != input_shape:
            raise RuntimeError(
                "Frequency adapter changed the token shape at block "
                f"{layer_index}: input={tuple(input_shape)}, "
                f"output={tuple(output.shape)}."
            )

        return output

    # ------------------------------------------------------------------
    # Transformer execution
    # ------------------------------------------------------------------

    def _run_transformer_blocks(
        self,
        x: torch.Tensor,
        batch_size: int,
        num_frames: int,
    ) -> torch.Tensor:
        """Run CLIP blocks and insert frequency adapters at selected layers."""
        for layer_index, block in enumerate(self.blocks):

            if self.use_global_prompts:
                global_prompts = self.global_prompts[
                    layer_index
                ].unsqueeze(0).expand(
                    batch_size * num_frames,
                    -1,
                    -1,
                )

                # Insert temporary global prompts after CLS.
                x = torch.cat(
                    (
                        x[:, :1, :],
                        global_prompts,
                        x[:, 1:, :],
                    ),
                    dim=1,
                )

                x = block(x)

                # Remove temporary global prompts before any frequency adapter.
                x = torch.cat(
                    (
                        x[:, :1, :],
                        x[:, self.num_global_prompts + 1 :, :],
                    ),
                    dim=1,
                )
            else:
                x = block(x)

            # Adapter is placed after the complete Transformer block.
            if (
                self.use_frequency_adapter
                and layer_index in self.frequency_layers
            ):
                x = self._apply_frequency_adapter(
                    x,
                    layer_index=layer_index,
                    num_frames=num_frames,
                )

        return x

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(
                "Expected video input [B, C, T, H, W], "
                f"got {tuple(x.shape)}."
            )

        batch_size, channels, num_frames, height, width = x.shape

        if channels != 3:
            raise ValueError(
                f"Expected 3 input channels, got {channels}."
            )
        if (height, width) != self.input_size:
            raise ValueError(
                f"Expected spatial size {self.input_size}, "
                f"got {(height, width)}."
            )
        if num_frames <= 0:
            raise ValueError(
                f"Input contains an invalid frame count: {num_frames}."
            )

        # [B, 3, T, H, W] -> [B*T, 3, H, W]
        x = x.permute(0, 2, 1, 3, 4).flatten(0, 1)

        # [B*T, 3, 224, 224] -> [B*T, 196, 768]
        x = self.patch_embed(x)

        if x.ndim != 3:
            raise RuntimeError(
                "ImagePatchEmbed2D must return [B*T, N, D], "
                f"got {tuple(x.shape)}."
            )
        if x.shape[1] != self.num_patch_tokens:
            raise RuntimeError(
                "Unexpected patch-token count: "
                f"expected {self.num_patch_tokens}, got {x.shape[1]}."
            )
        if x.shape[2] != self.feature_dim:
            raise RuntimeError(
                "Unexpected patch embedding dimension: "
                f"expected {self.feature_dim}, got {x.shape[2]}."
            )

        # Add the original CLIP CLS token:
        # [B*T, 196, 768] -> [B*T, 197, 768]
        cls_token = self.cls_token.to(
            device=x.device,
            dtype=x.dtype,
        ).view(1, 1, -1).expand(
            x.shape[0],
            1,
            -1,
        )
        x = torch.cat((cls_token, x), dim=1)

        # Spatial position embedding.
        x = x + self.pos_embed.to(
            device=x.device,
            dtype=x.dtype,
        ).unsqueeze(0)

        # Temporal position embedding.
        x = self.temporal_encoding(
            x,
            num_frames=num_frames,
            batch_size=batch_size,
        )

        x = self.ln_pre(x)

        # [B*T, 197, 768] -> [B*T, 197, 768]
        x = self._run_transformer_blocks(
            x,
            batch_size=batch_size,
            num_frames=num_frames,
        )

        if x.shape[1] != self.num_tokens:
            raise RuntimeError(
                "Visual token layout was not restored before CLS extraction: "
                f"expected {self.num_tokens} tokens, got {x.shape[1]}."
            )

        # Per-frame CLIP projection:
        # [B*T, 197, 768] -> [B*T, 768] -> [B*T, 512]
        cls_x = self.ln_post(x[:, 0, :])
        cls_x = cls_x @ self.proj

        # Video-level temporal aggregation:
        # [B*T, 512] -> [B, T, 512] -> [B, 512]
        cls_x = rearrange(
            cls_x,
            "(b t) e -> b t e",
            b=batch_size,
            t=num_frames,
        )
        video_feature = cls_x.mean(dim=1)

        return video_feature