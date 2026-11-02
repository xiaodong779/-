#!/usr/bin/env python3
"""Video-level heads for TGL v2: TemporalPool, track quality, late fusion.

* TemporalPool: masked mean over the T frame-level CLS features (no heavy
  temporal transformer - the frozen CLIP + prompts already encode time).
* track_quality: [B, 4] reliability statistics ONLY
  (valid track ratio, mean visibility, mean valid temporal length,
  out-of-bounds ratio). No text features, no logits.
* LateFusion: per-sample SCALAR gate fusion of video_visual and
  trajectory_visual.
    gated_residual (default):  final = norm(video + gate * trajectory)
    concat_linear (ablation):  final = norm(concat_projection([video, traj]))
  Gate bias is initialised to logit(0.1) ~= -2.197 (trajectory starts as a
  small supplement). All-invalid trajectory videos force gate = 0 and
  final == video_visual exactly (no NaN anywhere).
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["TemporalPool", "track_quality_stats", "LateFusion"]


class TemporalPool(nn.Module):
    """Masked mean over the temporal axis. [B, T, D] x [B, T] -> [B, D]."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, frame_features: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        if mask is None:
            return frame_features.mean(dim=1)
        m = mask.float().unsqueeze(-1)
        return (frame_features * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)


def track_quality_stats(
    valid: torch.Tensor,      # [B, K, T] bool (visibility & in_bounds & padding)
    visibility: torch.Tensor, # [B, K, T]
    in_bounds: torch.Tensor,  # [B, K, T] bool
    any_valid: torch.Tensor,  # [B] bool
) -> torch.Tensor:
    """[B, 4] reliability statistics (all in [0, 1], zero for empty videos)."""
    vf = valid.float()
    valid_track_ratio = (valid.any(dim=-1)).float().mean(dim=-1)  # [B]
    mean_vis = ((visibility * vf).sum(dim=(1, 2)) / vf.sum(dim=(1, 2)).clamp_min(1.0))
    per_track_len = valid.sum(dim=-1).float() / max(valid.shape[-1], 1)  # [B,K]
    mean_len = (per_track_len * valid.any(dim=-1).float()).sum(-1) / valid.any(dim=-1).float().sum(-1).clamp_min(1.0)
    oob_ratio = 1.0 - (
        (in_bounds.float() * vf).sum(dim=(1, 2)) / vf.sum(dim=(1, 2)).clamp_min(1.0)
    )
    q = torch.stack([valid_track_ratio, mean_vis, mean_len, oob_ratio], dim=-1)
    return q * any_valid.unsqueeze(-1).float()


class LateFusion(nn.Module):
    """Pure-image-side adaptive late fusion of two 512-d video views."""

    def __init__(
        self,
        dim: int = 512,
        quality_dim: int = 4,
        fusion_mode: str = "gated_residual",
        gate_init: float = 0.1,
    ) -> None:
        super().__init__()
        if fusion_mode not in ("gated_residual", "concat_linear"):
            raise ValueError(f"unknown fusion_mode {fusion_mode!r}")
        self.fusion_mode = fusion_mode
        self.dim = int(dim)
        self.quality_dim = int(quality_dim)
        if fusion_mode == "gated_residual":
            # The gate should estimate track reliability, not memorize a
            # 1024-dimensional video/trajectory pair.  This is the usual
            # two-stream late-fusion pattern: branch agreement + quality.
            self.gate_linear = nn.Linear(quality_dim + 3, 1)
            nn.init.zeros_(self.gate_linear.weight)
            nn.init.constant_(self.gate_linear.bias, math.log(gate_init / (1.0 - gate_init)))
            self.trajectory_calibration = nn.Linear(dim, dim)
            nn.init.eye_(self.trajectory_calibration.weight)
            nn.init.zeros_(self.trajectory_calibration.bias)
        else:
            self.concat_projection = nn.Linear(dim * 2, dim)

    def forward(
        self,
        video_visual: torch.Tensor,      # [B, 512] normalized
        trajectory_visual: torch.Tensor, # [B, 512] normalized (zero if empty)
        track_quality: torch.Tensor,     # [B, 4]
        any_valid: torch.Tensor,         # [B] bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        any_f = any_valid.float().unsqueeze(-1)
        if self.fusion_mode == "gated_residual":
            trajectory_visual = self.trajectory_calibration(
                trajectory_visual.float()
            ).to(video_visual.dtype)
            cosine = (video_visual.float() * trajectory_visual.float()).sum(
                dim=-1, keepdim=True
            )
            gate_input = torch.cat(
                [track_quality, cosine, cosine.abs(), trajectory_visual.float().norm(
                    dim=-1, keepdim=True
                )], dim=-1
            )
            gate = torch.sigmoid(self.gate_linear(gate_input)).to(video_visual.dtype)
            gate = gate * any_f.to(video_visual.dtype)  # empty video -> gate 0
            final = video_visual + gate * trajectory_visual
        else:
            concat = torch.cat(
                [video_visual, trajectory_visual * any_f], dim=-1
            ).float()
            fused = self.concat_projection(concat).to(video_visual.dtype)
            final = video_visual + any_f.to(video_visual.dtype) * 0.0 + fused
        final = F.normalize(final.float(), dim=-1).to(video_visual.dtype)
        return final, gate if self.fusion_mode == "gated_residual" else torch.zeros_like(final)
