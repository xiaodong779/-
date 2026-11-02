#!/usr/bin/env python3
"""Trokens-style motion-aware Track-ST encoder for TGL v2.

Pipeline (per forward):
    trajectory_appearance [B, K, T, 768]   (gathered, raw CLIP patch feats)
        -> appearance_norm + appearance_proj -> [B, K, T, 192]
    motion input (from POST-augmentation coords):
        normalized x, y / normalized dx, dy / speed / sin(theta), cos(theta)
        / visibility / relative track age                    -> [B, K, T, 9]
        -> motion_mlp -> [B, K, T, 192]
    z = appearance + motion + temporal_embedding, zeroed at invalid positions
    -> 2 decoupled Track-ST blocks:
         block: z = z + temporal_attn(z)   (per track, over T  -> [B*K, T, T])
                z = z + spatial_attn(z)    (per time,  over K  -> [B*T, K, K])
         invalid positions re-zeroed after every sub-layer
         (NO global (K*T)^2 attention, NO fixed-order inter-motion descriptor)
    -> Trajectory CLS: learnable query cross-attends z_flat [B, K*T, 192]
         with key_padding_mask=~valid  -> [B, 192]
    -> trajectory_norm -> trajectory_projection -> [B, 512]

All-invalid video guarantee: zero trajectory_cls, zero trajectory_visual,
no NaN (attention is bypassed for samples without any valid position).

Training-only auxiliary heads (disabled automatically in eval mode):
    * trajectory_aux_head: linear(192 -> num_classes) on trajectory_cls;
    * reconstruction decoder: from MASKED z (appearance+motion zeroed at the
      masked positions before the blocks), reconstruct (i) the stop-gradient
      trajectory-aligned CLIP feature and (ii) normalized (dx, dy); loss is
      computed only on masked & valid positions. The main classification
      branch always uses the UNMASKED trajectory.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["TrackSTEncoder", "build_motion_input"]


def build_motion_input(
    tracks: torch.Tensor,        # [B, K, T, 2] post-augmentation pixel coords
    visibility: torch.Tensor,    # [B, K, T]
    valid: torch.Tensor,         # [B, K, T] bool (visibility & in_bounds & padding)
    image_w: int,
    image_h: int,
    rel_time: Optional[torch.Tensor] = None,  # [B, K, T] in [0,1]; None = t/(T-1)
) -> torch.Tensor:
    """Motion descriptor [B, K, T, 9]:

    (x/W, y/H, dx, dy, speed, sin(theta), cos(theta), visibility, rel_time)
    All quantities derived from post-augmentation coordinates; zeroed at
    invalid positions (and dx/dy/speed/theta zeroed at t=0 - no predecessor).
    """
    B, K, T, _ = tracks.shape
    x = tracks[..., 0].float() / float(image_w)
    y = tracks[..., 1].float() / float(image_h)
    dx = torch.zeros_like(x)
    dy = torch.zeros_like(y)
    if T > 1:
        dx[:, :, 1:] = x[:, :, 1:] - x[:, :, :-1]
        dy[:, :, 1:] = y[:, :, 1:] - y[:, :, :-1]
    speed = torch.sqrt(dx * dx + dy * dy).clamp_min(1e-8)
    theta = torch.atan2(dy, dx)
    sin_t, cos_t = torch.sin(theta), torch.cos(theta)
    if rel_time is None:
        rel_time = (
            torch.arange(T, device=tracks.device).float()[None, None, :] / max(T - 1, 1)
        ).expand(B, K, T).contiguous()
    feat = torch.stack(
        [x, y, dx, dy, speed, sin_t, cos_t, visibility.float(), rel_time], dim=-1
    )  # [B,K,T,9]
    feat = feat * valid.unsqueeze(-1).float()
    if T > 1:
        feat[:, :, 0, :] = 0.0  # no predecessor at t=0: zero the dynamics
    return feat


class _TrackSTBlock(nn.Module):
    """Decoupled temporal-then-spatial attention block (pre-norm residuals)."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm_t = nn.LayerNorm(dim)
        self.attn_t = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_s = nn.LayerNorm(dim)
        self.attn_s = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, z: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        # z: [B, K, T, D]; valid: [B, K, T] bool.
        B, K, T, D = z.shape

        # -- temporal attention: per track over T (batch B*K, seq T) -------
        zt = z.reshape(B * K, T, D)
        vt = valid.reshape(B * K, T)
        h = self.norm_t(zt)
        # nn.MultiheadAttention has no key padding mask here; mask by zeroing
        # keys AND queries: an all-invalid track produces zero query and zero
        # values -> zero attention output (softmax of equal logits = uniform,
        # but values are 0 -> output 0). Additive -inf on invalid KEYS keeps
        # valid rows exact; we implement it via attn_mask.
        key_mask = ~vt  # [B*K, T] True = masked out
        attn_out, _ = self.attn_t(h, h, h, key_padding_mask=key_mask, need_weights=False)
        attn_out = torch.nan_to_num(attn_out, nan=0.0)
        zt = zt + self.dropout(attn_out)
        zt = zt * vt.unsqueeze(-1).float()  # re-zero invalid positions
        zt = zt.reshape(B, K, T, D)  # un-flatten the temporal batch

        # -- spatial attention: per time step over K (batch B*T, seq K) ----
        zs = zt.permute(0, 2, 1, 3).reshape(B * T, K, D)
        vs = valid.permute(0, 2, 1).reshape(B * T, K)
        h = self.norm_s(zs)
        key_mask = ~vs
        attn_out, _ = self.attn_s(h, h, h, key_padding_mask=key_mask, need_weights=False)
        attn_out = torch.nan_to_num(attn_out, nan=0.0)
        zs = zs + self.dropout(attn_out)
        zs = zs * vs.unsqueeze(-1).float()  # re-zero invalid positions

        return zs.reshape(B, T, K, D).permute(0, 2, 1, 3).contiguous()


class TrackSTEncoder(nn.Module):
    """Track-ST encoder + Trajectory CLS + 512-d projection.

    Args mirror the spec: appearance from gathered CLIP patch tokens (768),
    trajectory dim 192, 2 decoupled blocks, 4 heads. `num_classes > 0`
    enables the training-only trajectory auxiliary classifier head.
    """

    def __init__(
        self,
        clip_dim: int = 768,
        track_dim: int = 192,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.0,
        num_classes: int = 0,
        motion_feature_dim: int = 9,
    ) -> None:
        super().__init__()
        self.track_dim = int(track_dim)
        self.appearance_norm = nn.LayerNorm(clip_dim)
        self.appearance_proj = nn.Linear(clip_dim, self.track_dim)
        self.motion_mlp = nn.Sequential(
            nn.Linear(motion_feature_dim, self.track_dim),
            nn.GELU(),
            nn.Linear(self.track_dim, self.track_dim),
        )
        self.temporal_embedding = nn.Parameter(torch.zeros(1, 1, self.track_dim))
        nn.init.trunc_normal_(self.temporal_embedding, std=0.02)
        self.blocks = nn.ModuleList(
            _TrackSTBlock(self.track_dim, num_heads, dropout) for _ in range(int(num_layers))
        )
        self.final_norm = nn.LayerNorm(self.track_dim)
        # Trajectory CLS: learnable query over ALL valid K*T positions.
        self.trajectory_cls_query = nn.Parameter(torch.zeros(1, 1, self.track_dim))
        nn.init.trunc_normal_(self.trajectory_cls_query, std=0.02)
        self.cls_norm_q = nn.LayerNorm(self.track_dim)
        self.cls_attn = nn.MultiheadAttention(self.track_dim, num_heads, batch_first=True)
        self.trajectory_norm = nn.LayerNorm(self.track_dim)
        self.trajectory_projection = nn.Linear(self.track_dim, 512)
        # training-only auxiliary classifier (no text involvement)
        self.trajectory_aux_head = (
            nn.Linear(self.track_dim, int(num_classes)) if int(num_classes) > 0 else None
        )
        # masked reconstruction decoder (training only): z -> (clip_feat, dx, dy)
        self.recon_decoder = nn.Sequential(
            nn.Linear(self.track_dim, self.track_dim * 2),
            nn.GELU(),
            nn.Linear(self.track_dim * 2, 2),
        )
        self.recon_clip_proj = nn.Linear(self.track_dim, 768)

    def forward(
        self,
        trajectory_appearance: torch.Tensor,  # [B, K, T, 768]
        motion: torch.Tensor,                 # [B, K, T, 9]
        valid: torch.Tensor,                  # [B, K, T] bool
        labels: Optional[torch.Tensor] = None,
        recon_mask_ratio: float = 0.0,
        clip_target: Optional[torch.Tensor] = None,  # [B, K, T, 768] stop-grad
        dxdy_target: Optional[torch.Tensor] = None,  # [B, K, T, 2] normalized
    ) -> Dict[str, torch.Tensor]:
        B, K, T, _ = trajectory_appearance.shape
        appearance = self.appearance_proj(self.appearance_norm(trajectory_appearance.float()))
        motion = self.motion_mlp(motion.float())
        z = appearance + motion + self.temporal_embedding
        z = z * valid.unsqueeze(-1).float()
        for blk in self.blocks:
            z = blk(z, valid)
        z = self.final_norm(z) * valid.unsqueeze(-1).float()

        # ---- Trajectory CLS over the flattened valid positions ------------
        z_flat = z.reshape(B, K * T, self.track_dim)
        mask_flat = valid.reshape(B, K * T)
        any_valid = mask_flat.any(dim=-1)  # [B]
        q = self.cls_norm_q(self.trajectory_cls_query).expand(B, -1, -1)
        attn_out, _ = self.cls_attn(
            q, z_flat, z_flat, key_padding_mask=~mask_flat, need_weights=False
        )
        attn_out = torch.nan_to_num(attn_out, nan=0.0)  # all-invalid rows -> 0
        trajectory_cls = attn_out.squeeze(1) * any_valid.unsqueeze(-1).float()
        trajectory_cls = self.trajectory_norm(trajectory_cls) * any_valid.unsqueeze(-1).float()
        trajectory_visual = self.trajectory_projection(trajectory_cls)
        trajectory_visual = F.normalize(trajectory_visual, dim=-1)
        # all-invalid videos: exactly-zero vector (never NaN)
        trajectory_visual = trajectory_visual * any_valid.unsqueeze(-1).float()

        out: Dict[str, torch.Tensor] = {
            "trajectory_cls": trajectory_cls,
            "trajectory_visual": trajectory_visual,
            "any_valid": any_valid,
            "z": z,
        }

        if self.trajectory_aux_head is not None and labels is not None:
            out["trajectory_logits"] = self.trajectory_aux_head(trajectory_cls)

        # ---- masked reconstruction (training only) -------------------------
        # Spec §11: mask SOME valid positions, ZERO their appearance AND
        # motion inputs, re-run the encoder (masked pass), and reconstruct
        # (i) the stop-gradient trajectory-aligned CLIP feature and (ii)
        # normalized (dx, dy) - loss only on masked & valid positions. The
        # main branch above always used the UNMASKED trajectory.
        if self.training and recon_mask_ratio > 0.0 and clip_target is not None:
            rand = torch.rand(B, K, T, device=z.device)
            recon_mask = (rand < recon_mask_ratio) & valid
            if recon_mask.any():
                keep = (~recon_mask) & valid
                app_m = appearance * keep.unsqueeze(-1).float()
                mot_m = motion * keep.unsqueeze(-1).float()
                zm = app_m + mot_m + self.temporal_embedding
                zm = zm * valid.unsqueeze(-1).float()
                for blk in self.blocks:
                    zm = blk(zm, valid)
                zm = self.final_norm(zm) * valid.unsqueeze(-1).float()
                dec = self.recon_decoder(zm.reshape(B, K * T, self.track_dim))
                recon_clip = self.recon_clip_proj(zm.reshape(B, K * T, self.track_dim))
                m = recon_mask.reshape(B, K * T)
                tgt_clip = F.normalize(
                    clip_target.detach().float().reshape(B, K * T, -1), dim=-1
                )
                tgt_dxdy = dxdy_target.reshape(B, K * T, -1).float()
                loss_clip = (1.0 - F.cosine_similarity(recon_clip, tgt_clip, dim=-1)) * m.float()
                loss_dxdy = (dec - tgt_dxdy).norm(dim=-1) * m.float()
                n = m.float().sum().clamp_min(1.0)
                out["loss_recon_clip"] = loss_clip.sum() / n
                out["loss_recon_dxdy"] = loss_dxdy.sum() / n
            else:
                zero = z.new_zeros(())
                out["loss_recon_clip"] = zero
                out["loss_recon_dxdy"] = zero
        return out
