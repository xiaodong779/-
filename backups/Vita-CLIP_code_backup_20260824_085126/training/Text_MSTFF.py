#!/usr/bin/env python
"""Adaptive Multi-Strategy Text Feature Fusion for supervised action recognition.

The module is applied after the frozen CLIP text Transformer and ``ln_final``,
but before ``text_projection``.

Input:
    token_features: [N, L, D]
    eot_feature:    [N, D]
    token_mask:     optional [N, L], True for valid non-padding tokens

Output:
    fused_feature:  [N, D]
    weights:        [3], corresponding to EOT / Mean / Attention branches

The optional token mask is important for CLIP's fixed context length because
mean and attention pooling should not include padding positions.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _smooth_cap(
    values: torch.Tensor,
    caps: torch.Tensor,
    sharpness: float,
) -> torch.Tensor:
    """Apply a differentiable soft upper bound.

    This is a soft constraint rather than a mathematically exact cap after
    re-normalization. It avoids the zero-gradient region introduced by a hard
    ``torch.minimum`` in most practical operating ranges.
    """
    gate = torch.sigmoid((caps - values) * sharpness)
    return gate * values + (1.0 - gate) * caps


class TextMSTFF(nn.Module):
    """Fuse EOT, masked-mean and masked-attention text representations."""

    def __init__(
        self,
        embed_dim: int = 512,
        max_weights: Sequence[float] = (0.6, 0.3, 0.5),
        smooth_weight_constraint: bool = True,
        weight_constraint_sharpness: float = 10.0,
        selector_hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        sanitize_output: bool = False,
    ) -> None:
        super().__init__()

        self.embed_dim = int(embed_dim)
        if self.embed_dim <= 0:
            raise ValueError(
                f"embed_dim must be positive, got {self.embed_dim}."
            )

        max_weights = tuple(float(value) for value in max_weights)
        if len(max_weights) != 3:
            raise ValueError(
                "max_weights must contain exactly three values for "
                "EOT, Mean and Attention branches."
            )
        if any(value <= 0.0 for value in max_weights):
            raise ValueError(
                f"All max_weights must be positive, got {max_weights}."
            )

        if weight_constraint_sharpness <= 0.0:
            raise ValueError(
                "weight_constraint_sharpness must be positive."
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                f"dropout must be in [0, 1), got {dropout}."
            )

        hidden_dim = (
            max(self.embed_dim // 2, 1)
            if selector_hidden_dim is None
            else int(selector_hidden_dim)
        )
        if hidden_dim <= 0:
            raise ValueError(
                f"selector_hidden_dim must be positive, got {hidden_dim}."
            )

        self.smooth_weight_constraint = bool(
            smooth_weight_constraint
        )
        self.weight_constraint_sharpness = float(
            weight_constraint_sharpness
        )
        self.sanitize_output = bool(sanitize_output)

        # Token importance estimator for masked attention pooling.
        self.token_selector = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # Independent projections prevent the three pooling strategies from
        # being forced into exactly the same feature transformation.
        self.eot_proj = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.embed_dim),
        )
        self.mean_proj = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.embed_dim),
        )
        self.att_proj = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.embed_dim),
        )
        self.output_dropout = nn.Dropout(dropout)

        # Prior ratio EOT : Mean : Attention = 1 : 0.5 : 0.5.
        initial_weights = torch.tensor(
            [1.0, 0.5, 0.5],
            dtype=torch.float32,
        )
        self.fusion_logits = nn.Parameter(
            initial_weights.log()
        )

        self.register_buffer(
            "max_weights",
            torch.tensor(
                max_weights,
                dtype=torch.float32,
            ),
        )

        # Diagnostic value only; it is not part of the state dict.
        self._last_weights: Optional[torch.Tensor] = None

    @staticmethod
    def _validate_inputs(
        token_features: torch.Tensor,
        eot_feature: torch.Tensor,
        token_mask: Optional[torch.Tensor],
        embed_dim: int,
    ) -> Tuple[int, int, int]:
        if token_features.ndim != 3:
            raise ValueError(
                "token_features must have shape [N, L, D], "
                f"got {tuple(token_features.shape)}."
            )
        if eot_feature.ndim != 2:
            raise ValueError(
                "eot_feature must have shape [N, D], "
                f"got {tuple(eot_feature.shape)}."
            )

        batch_size, context_length, feature_dim = (
            token_features.shape
        )

        if eot_feature.shape != (
            batch_size,
            feature_dim,
        ):
            raise ValueError(
                "eot_feature shape mismatch: "
                f"expected {(batch_size, feature_dim)}, "
                f"got {tuple(eot_feature.shape)}."
            )
        if feature_dim != embed_dim:
            raise ValueError(
                f"Expected feature dimension {embed_dim}, "
                f"got {feature_dim}."
            )

        if token_mask is not None:
            if token_mask.ndim != 2:
                raise ValueError(
                    "token_mask must have shape [N, L], "
                    f"got {tuple(token_mask.shape)}."
                )
            if token_mask.shape != (
                batch_size,
                context_length,
            ):
                raise ValueError(
                    "token_mask shape mismatch: "
                    f"expected {(batch_size, context_length)}, "
                    f"got {tuple(token_mask.shape)}."
                )

        return batch_size, context_length, feature_dim

    @staticmethod
    def _prepare_mask(
        token_features: torch.Tensor,
        token_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch_size, context_length, _ = token_features.shape

        if token_mask is None:
            mask = torch.ones(
                batch_size,
                context_length,
                device=token_features.device,
                dtype=torch.bool,
            )
        else:
            mask = token_mask.to(
                device=token_features.device,
                dtype=torch.bool,
            )

        valid_counts = mask.sum(dim=1)
        if torch.any(valid_counts == 0):
            invalid_rows = torch.nonzero(
                valid_counts == 0,
                as_tuple=False,
            ).flatten().tolist()
            raise ValueError(
                "token_mask contains samples with no valid tokens: "
                f"indices={invalid_rows[:10]}."
            )

        return mask

    def _compute_fusion_weights(self) -> torch.Tensor:
        weights = F.softmax(
            self.fusion_logits.float(),
            dim=0,
        )
        caps = self.max_weights.to(
            device=weights.device,
            dtype=weights.dtype,
        )

        if self.smooth_weight_constraint:
            weights = _smooth_cap(
                weights,
                caps,
                sharpness=self.weight_constraint_sharpness,
            )
        else:
            weights = torch.minimum(
                weights,
                caps,
            )

        weights = weights.clamp_min(1e-6)
        weights = weights / weights.sum().clamp_min(1e-6)

        if weights.shape != (3,):
            raise RuntimeError(
                "Expected three fusion weights, "
                f"got shape {tuple(weights.shape)}."
            )

        return weights

    def forward(
        self,
        token_features: torch.Tensor,
        eot_feature: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fuse three text representations.

        ``token_mask`` should normally be:
            ``tokenized_prompts.ne(0)``

        The EOT token is non-zero and therefore remains included.
        """
        self._validate_inputs(
            token_features,
            eot_feature,
            token_mask,
            self.embed_dim,
        )
        mask = self._prepare_mask(
            token_features,
            token_mask,
        )

        working_dtype = token_features.dtype
        mask_float = mask.unsqueeze(-1).to(
            dtype=working_dtype
        )

        # 1. EOT branch.
        eot_branch = self.eot_proj(
            eot_feature
        )

        # 2. Masked mean branch. Padding positions do not contribute.
        valid_counts = mask_float.sum(
            dim=1
        ).clamp_min(1.0)
        mean_feature = (
            token_features * mask_float
        ).sum(dim=1) / valid_counts
        mean_branch = self.mean_proj(
            mean_feature
        )

        # 3. Masked attention branch.
        token_scores = self.token_selector(
            token_features
        ).squeeze(-1)

        # Compute softmax in FP32 for numerical stability.
        negative_large = torch.finfo(
            torch.float32
        ).min
        token_scores_fp32 = token_scores.float().masked_fill(
            ~mask,
            negative_large,
        )
        token_weights = F.softmax(
            token_scores_fp32,
            dim=1,
        ).to(dtype=working_dtype)

        attention_feature = torch.sum(
            token_weights.unsqueeze(-1)
            * token_features,
            dim=1,
        )
        attention_branch = self.att_proj(
            attention_feature
        )

        # 4. Adaptive branch fusion.
        weights = self._compute_fusion_weights()

        fused_feature = (
            weights[0].to(working_dtype) * eot_branch
            + weights[1].to(working_dtype) * mean_branch
            + weights[2].to(working_dtype) * attention_branch
        )
        fused_feature = self.output_dropout(
            fused_feature
        )

        if self.sanitize_output:
            fused_feature = torch.nan_to_num(
                fused_feature,
                nan=0.0,
                posinf=1e6,
                neginf=-1e6,
            )

        self._last_weights = (
            weights.detach().clone()
        )

        return fused_feature, weights