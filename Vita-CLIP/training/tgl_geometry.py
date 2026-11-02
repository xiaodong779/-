#!/usr/bin/env python3
"""Unified trajectory geometry for TGL v2.

Single source of truth for every coordinate operation applied to offline
track caches, so image augmentations and track coordinates always share the
same transform:

    raw coords -> resize -> crop -> horizontal flip -> final 224x224 coords

Conventions (locked by unit tests):
    * track coords are (x, y): x = image width axis, y = image height axis;
    * patches are row-major: patch_index = py * grid_w + px;
    * patch index uses FLOOR binning: px = floor(x * grid_w / W);
    * `in_bounds` is computed BEFORE any clamping - an out-of-bounds point
      can never become valid by clamping;
    * flip: x_new = W - 1 - x_old (pixel convention);
    * crop+resize: x_new = (x_old - crop_x) * out_w / crop_w.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

__all__ = [
    "track_patch_indices",
    "flip_x",
    "crop_resize_coords",
    "flip_affine",
    "crop_resize_affine",
    "apply_affine",
    "reverse_time",
]


def track_patch_indices(
    x: torch.Tensor,
    y: torch.Tensor,
    grid_w: int,
    grid_h: int,
    image_w: int,
    image_h: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Floor-bin track coordinates into row-major patch indices.

    Args:
        x: [B, K, T] (or any shape) track x in [0, image_w) pixel coords.
        y: same for y in [0, image_h).
    Returns:
        (patch_index, in_bounds, px_py_safe):
        patch_index [same shape] long, clamped to the grid (SAFE values for
        gathering placeholders - validity is carried separately by
        `in_bounds`);
        in_bounds [same shape] bool, computed BEFORE clamping;
        (px_safe, py_safe) clamped integer bin indices.
    """
    x = x.float()
    y = y.float()
    in_bounds = (
        (x >= 0) & (x < image_w) & (y >= 0) & (y < image_h)
    )
    px = torch.floor(x * float(grid_w) / float(image_w))
    py = torch.floor(y * float(grid_h) / float(image_h))
    px_safe = px.clamp(0, grid_w - 1).long()
    py_safe = py.clamp(0, grid_h - 1).long()
    patch_index = py_safe * grid_w + px_safe
    return patch_index, in_bounds, (px_safe, py_safe)


def flip_x(x: torch.Tensor, image_w: int) -> torch.Tensor:
    """Horizontal flip of track x coords: x_new = W - 1 - x_old."""
    return float(image_w) - 1.0 - x


def crop_resize_coords(
    x: torch.Tensor,
    y: torch.Tensor,
    crop_x: float,
    crop_y: float,
    crop_w: float,
    crop_h: float,
    out_w: int,
    out_h: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Map raw coords through a crop window into the output canvas."""
    x_new = (x.float() - float(crop_x)) * float(out_w) / float(crop_w)
    y_new = (y.float() - float(crop_y)) * float(out_h) / float(crop_h)
    return x_new, y_new


def flip_affine(image_w: int, image_h: int) -> torch.Tensor:
    """3x3 homogeneous affine for a horizontal flip of width image_w."""
    return torch.tensor(
        [[-1.0, 0.0, image_w - 1.0],
         [0.0, 1.0, 0.0],
         [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )


def crop_resize_affine(
    crop_x: float,
    crop_y: float,
    crop_w: float,
    crop_h: float,
    out_w: int,
    out_h: int,
) -> torch.Tensor:
    """3x3 homogeneous affine: crop window -> out canvas.

    Applies (x - crop_x) * out_w / crop_w, then (y - crop_y) * out_h / crop_h.
    """
    sx = float(out_w) / float(crop_w)
    sy = float(out_h) / float(crop_h)
    return torch.tensor(
        [[sx, 0.0, -float(crop_x) * sx],
         [0.0, sy, -float(crop_y) * sy],
         [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )


def apply_affine(
    x: torch.Tensor,
    y: torch.Tensor,
    affine: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply a 3x3 homogeneous affine to (x, y) coordinate tensors."""
    ones = torch.ones_like(x.float())
    pts = torch.stack([x.float(), y.float(), ones], dim=-1)  # [..., 3]
    out = pts @ affine.to(pts.device).t()
    return out[..., 0], out[..., 1]


def reverse_time(
    frames: Optional[torch.Tensor],
    tracks: torch.Tensor,
    visibility: torch.Tensor,
    frame_indices: torch.Tensor,
    time_dim_frames: int = 1,
    time_dim_tracks: int = 2,
) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reverse temporal order of frames AND tracks TOGETHER.

    Coordinates are NOT touched (the tracked points stay with their frames);
    only the time axis of every tensor is reversed. Enabled only by explicit
    config - default OFF.
    """
    fr = None
    if frames is not None:
        fr = frames.flip(time_dim_frames)
    return (
        fr,
        tracks.flip(time_dim_tracks),
        visibility.flip(time_dim_tracks),
        frame_indices.flip(0),
    )
