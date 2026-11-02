#!/usr/bin/env python3
"""Frozen Grounding-DINO + SAM2 online mask-prompt generator.

This module owns ONLY the mask-generation stage of the Mask-Prompt pipeline:

    video decode -> sample T frames -> shared geometric transforms (done in
    the dataset) -> [this module] frozen Grounding-DINO detection -> frozen
    SAM2 per-frame masks -> expanded-box / interaction masks
    -> [VitaCLIP.forward] spatial-prompt fusion + residual enhancement +
    CLIP normalization -> frozen CLIP image encoder.

The generator is a self-contained interface so the online (DINO+SAM2)
implementation can later be swapped for an offline cached-mask
implementation (``OfflineCachedMaskPromptGenerator``) without touching the
Mask-Prompt enhancement or the CLIP forward path.

Invariants:
- Grounding-DINO and SAM2 are fully frozen: ``eval()`` + ``no_grad`` +
  ``requires_grad=False`` for every parameter.
- On any per-clip failure the generator falls back to identity masks
  (``mask_valid == 0`` everywhere), guaranteeing the enhanced frames are
  bit-identical to the raw RGB input. All-black frames are never produced.
- The masks are returned as binary {0,1} float tensors shaped
  [B, T, 1, H, W]; validity flags are shaped [B, T].
"""

from __future__ import annotations

import dataclasses
import os
import shutil
from typing import Any, List, Optional, Sequence, Tuple

import torch


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class MaskPromptOutput:
    """Masks produced by a mask-prompt generator for one batch.

    Shapes (B = batch, T = frames, H/W = spatial size):
        actor_masks       [B, T, 1, H, W]  {0,1} SAM2 actor segmentation
        expanded_box_masks[B, T, 1, H, W]  {0,1} mask-recomputed box dilated
                                            by ``expand_ratio`` and clipped
        interaction_masks [B, T, 1, H, W]  {0,1} expanded_box * (1 - actor)
        mask_valid        [B, T]           {0,1} per-frame validity; a frame
                                            without a detected main subject
                                            is 0 and must stay un-enhanced
        dino_boxes_vis    [B, T, 3, H, W]  frames with selected DINO boxes
                                            drawn (debug visualization only)
        dino_boxes        list[B][T] of (xyxy tensor or None, score float)
        dino_phrases      list[B][T] of str ("" when no box)
    """

    actor_masks: torch.Tensor
    expanded_box_masks: torch.Tensor
    interaction_masks: torch.Tensor
    mask_valid: torch.Tensor
    dino_boxes_vis: torch.Tensor
    dino_boxes: List[List[Optional[torch.Tensor]]]
    dino_phrases: List[List[str]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _box_iou_xyxy(
    box_a: torch.Tensor,
    box_b: torch.Tensor,
) -> float:
    """IoU between two xyxy pixel boxes (tensors of shape [4])."""
    if box_a is None or box_b is None:
        return 0.0

    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0.0:
        return 0.0

    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(
        0.0, float(box_a[3] - box_a[1])
    )
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(
        0.0, float(box_b[3] - box_b[1])
    )
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def _mask_to_box_xyxy(
    mask: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Tight xyxy pixel box of a binary [H, W] mask; None when empty."""
    if mask is None:
        return None
    flat = mask.reshape(-1)
    if not bool((flat > 0.5).any()):
        return None

    rows = (mask.sum(dim=1) > 0.5).nonzero(as_tuple=False).flatten()
    cols = (mask.sum(dim=0) > 0.5).nonzero(as_tuple=False).flatten()
    if rows.numel() == 0 or cols.numel() == 0:
        return None

    y1 = int(rows.min().item())
    y2 = int(rows.max().item())
    x1 = int(cols.min().item())
    x2 = int(cols.max().item())
    return torch.tensor(
        [float(x1), float(y1), float(x2 + 1), float(y2 + 1)],
        dtype=torch.float32,
        device=mask.device,
    )


def _expand_box_xyxy(
    box: torch.Tensor,
    height: int,
    width: int,
    expand_ratio: float,
) -> torch.Tensor:
    """Expand a xyxy box by ``expand_ratio`` per side, clipped to bounds."""
    x1, y1, x2, y2 = (float(v) for v in box)
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    dx = w * float(expand_ratio) * 0.5
    dy = h * float(expand_ratio) * 0.5
    x1 = max(0.0, x1 - dx)
    y1 = max(0.0, y1 - dy)
    x2 = min(float(width), x2 + dx)
    y2 = min(float(height), y2 + dy)
    return torch.tensor(
        [x1, y1, x2, y2],
        dtype=torch.float32,
        device=box.device,
    )


def _box_to_mask(
    box: Optional[torch.Tensor],
    height: int,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    mask = torch.zeros(1, height, width, device=device)
    if box is None:
        return mask
    x1, y1, x2, y2 = (int(max(0.0, round(float(v)))) for v in box)
    x1 = max(0, min(x1, width))
    x2 = max(x1, min(x2, width))
    y1 = max(0, min(y1, height))
    y2 = max(y1, min(y2, height))
    if x2 > x1 and y2 > y1:
        mask[:, y1:y2, x1:x2] = 1.0
    return mask


def _draw_box_on_frame(
    frame: torch.Tensor,
    box: Optional[torch.Tensor],
    color: Sequence[float] = (1.0, 0.0, 0.0),
    thickness: int = 2,
) -> torch.Tensor:
    """Draw a red rectangle outline on a [3, H, W] frame (in place-safe)."""
    out = frame.clone()
    if box is None:
        return out

    _, height, width = out.shape
    x1, y1, x2, y2 = (int(round(float(v))) for v in box)
    x1 = max(0, min(x1, width - 1))
    x2 = max(0, min(x2, width - 1))
    y1 = max(0, min(y1, height - 1))
    y2 = max(0, min(y2, height - 1))

    color_tensor = torch.tensor(
        color, dtype=out.dtype, device=out.device
    ).view(3, 1, 1)

    t = max(1, int(thickness))
    if y2 >= y1:
        out[:, y1 : y1 + t, x1 : x2 + 1] = color_tensor
        out[:, y2 - t + 1 : y2 + 1, x1 : x2 + 1] = color_tensor
    if x2 >= x1:
        out[:, y1 : y2 + 1, x1 : x1 + t] = color_tensor
        out[:, y1 : y2 + 1, x2 - t + 1 : x2 + 1] = color_tensor
    return out


# ---------------------------------------------------------------------------
# Generator interface
# ---------------------------------------------------------------------------

class BaseMaskPromptGenerator:
    """Interface contract for all mask-prompt generators.

    ``generate`` must be callable under ``torch.no_grad()`` with CUDA frames
    in [0, 1]. Implementations must never return NaN masks and must keep the
    original input untouched when a clip fails (identity fallback).

    Every implementation accumulates lifetime counters in ``self.stats``:
        clips                    processed clips
        total_frames             frames seen
        dino_frames              frames with a detected main subject
        valid_frames             frames whose mask_valid == 1
        identity_fallback_clips  clips degraded to identity after a failure
    used by the training loop to report person_detection_rate /
    valid_mask_rate / identity_fallback_rate.
    """

    def __init__(self) -> None:
        self.stats = {
            "clips": 0,
            "total_frames": 0,
            "dino_frames": 0,
            "valid_frames": 0,
            "identity_fallback_clips": 0,
        }

    def reset_stats(self) -> None:
        self.stats = {
            key: 0 for key in self.stats
        }

    def generate(self, frames_rgb: torch.Tensor) -> MaskPromptOutput:
        raise NotImplementedError

    def train(self) -> None:  # generators are always inference-only
        raise RuntimeError(
            "Mask-prompt generators are frozen inference modules and must "
            "never be placed in training mode."
        )


class IdentityMaskPromptGenerator(BaseMaskPromptGenerator):
    """Degenerate generator: no masks, every frame invalid.

    With ``mask_valid == 0`` the downstream enhancement reduces to the
    identity, so this is also the per-clip failure fallback.
    """

    def generate(self, frames_rgb: torch.Tensor) -> MaskPromptOutput:
        batch, frames, _, height, width = frames_rgb.shape
        device = frames_rgb.device
        self.stats["clips"] += batch
        self.stats["total_frames"] += batch * frames
        zeros = torch.zeros(
            batch, frames, 1, height, width, device=device
        )
        return MaskPromptOutput(
            actor_masks=zeros.clone(),
            expanded_box_masks=zeros.clone(),
            interaction_masks=zeros.clone(),
            mask_valid=torch.zeros(batch, frames, device=device),
            dino_boxes_vis=frames_rgb.clone(),
            dino_boxes=[[None] * frames for _ in range(batch)],
            dino_phrases=[[""] * frames for _ in range(batch)],
        )


class OfflineCachedMaskPromptGenerator(BaseMaskPromptGenerator):
    """Reserved interface for offline pre-computed masks.

    Intended contract (to be wired up when the cache builder lands): the
    training dataset exposes a per-sample key; this generator loads
    ``{cache_dir}/{key}.pt`` containing ``actor_masks`` and optionally
    ``expanded_box_masks`` at the model input resolution and returns them
    with ``mask_valid`` derived from non-empty masks. The downstream
    Mask-Prompt enhancement and CLIP path stay untouched, which is exactly
    why the online/offline split lives behind this interface.
    """

    def __init__(self, cache_dir: str) -> None:
        self.cache_dir = cache_dir
        raise NotImplementedError(
            "Offline cached masks are a reserved interface. Point "
            "--mask_prompt_online (default true) to the online "
            "Grounding-DINO + SAM2 generator for now."
        )


class OnlineMaskPromptGenerator(BaseMaskPromptGenerator):
    """Frozen Grounding-DINO (subject detection) + SAM2 (mask propagation).

    Per clip:
      1. Grounding-DINO detects ``person`` (preferred) / other prompted
         classes on every frame at box/text thresholds.
      2. The main subject is chosen per frame by
         ``conf + w_area * area + w_iou * IoU(prev_box, box)`` among person
         boxes first (fallback: any detected class), keeping identity
         consistent across adjacent frames.
      3. SAM2 box-prompts the selected subject on anchor frames
         (default {0, 4, 7}) and propagates masks over all T frames.
      4. Per frame the tight box of the propagated mask is expanded by
         ``expand_ratio`` (30% default, clipped to frame bounds) giving
         ``expanded_box_masks``; ``interaction = expanded_box * (1-actor)``.
      5. Frames without a detected subject get ``mask_valid = 0`` and empty
         masks (identity downstream).
    """

    def __init__(
        self,
        text_prompt: str = "person . hand .",
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        expand_ratio: float = 0.30,
        sam2_prompt_frames: Sequence[int] = (0, 4, 7),
        fallback_identity: bool = True,
        dino_config: str = "",
        dino_checkpoint: str = "",
        sam2_config: str = "configs/sam2.1/sam2.1_hiera_large.yaml",
        sam2_checkpoint: str = "",
        subject_weight_conf: float = 1.0,
        subject_weight_area: float = 0.5,
        subject_weight_iou: float = 1.0,
        device: str = "cuda",
        max_consecutive_invalid_clips: int = 20,
    ) -> None:
        super().__init__()
        self.text_prompt = str(text_prompt)
        self.box_threshold = float(box_threshold)
        self.text_threshold = float(text_threshold)
        self.expand_ratio = float(expand_ratio)
        self.sam2_prompt_frames = tuple(sam2_prompt_frames)
        self.fallback_identity = bool(fallback_identity)
        self.subject_weight_conf = float(subject_weight_conf)
        self.subject_weight_area = float(subject_weight_area)
        self.subject_weight_iou = float(subject_weight_iou)
        self.device = torch.device(device)
        self.max_consecutive_invalid_clips = int(
            max_consecutive_invalid_clips
        )
        self._consecutive_invalid_clips = 0
        self.stats["failure_types"] = {}

        self._dino = None
        self._sam2 = None
        self._load_models(
            dino_config=dino_config,
            dino_checkpoint=dino_checkpoint,
            sam2_config=sam2_config,
            sam2_checkpoint=sam2_checkpoint,
        )

    # ------------------------------------------------------------------
    # Model loading (lazy imports so enabled=False never touches the deps)
    # ------------------------------------------------------------------

    def _load_models(
        self,
        dino_config: str,
        dino_checkpoint: str,
        sam2_config: str,
        sam2_checkpoint: str,
    ) -> None:
        if not dino_checkpoint:
            raise ValueError(
                "--mask_prompt_dino_checkpoint is required for the online "
                "Grounding-DINO + SAM2 mask generator. Download "
                "groundingdino_swint_ogc (69MB) from IDEA-Research and pass "
                "config+checkpoint paths."
            )
        if not sam2_checkpoint:
            raise ValueError(
                "--mask_prompt_sam2_checkpoint is required for the online "
                "Grounding-DINO + SAM2 mask generator. Download SAM 2.1 "
                "hiera-large from facebookresearch/sam2 and pass "
                "config+checkpoint paths."
            )

        try:
            import groundingdino
            from groundingdino.util.inference import (
                load_model as load_dino_model,
                predict as dino_predict_fn,
            )
        except ModuleNotFoundError as error:
            raise ModuleNotFoundError(
                "The 'groundingdino' package is required by the online mask "
                "prompt generator. Install it with: pip install "
                "git+https://github.com/IDEA-Research/GroundingDINO.git"
            ) from error

        # Keep the predict callable so the per-frame loop does not re-import.
        self._dino_predict = dino_predict_fn

        if not dino_config:
            # Default: the SwinT config shipped inside the pip package.
            dino_config = os.path.join(
                os.path.dirname(groundingdino.__file__),
                "config",
                "GroundingDINO_SwinT_OGC.py",
            )
            if not os.path.isfile(dino_config):
                raise ValueError(
                    "--mask_prompt_dino_config is required (could not "
                    f"auto-locate the packaged config at {dino_config})."
                )
        print(f"[mask_prompt] loading Grounding-DINO: {dino_checkpoint}")
        self._dino = load_dino_model(
            dino_config,
            dino_checkpoint,
            device=self.device.type,
        )
        self._dino.eval()
        for parameter in self._dino.parameters():
            parameter.requires_grad = False

        try:
            from sam2.build_sam import build_sam2_video_predictor
        except ModuleNotFoundError as error:
            raise ModuleNotFoundError(
                "The 'sam2' package is required by the online mask prompt "
                "generator. Install it from "
                "https://github.com/facebookresearch/sam2 and export "
                "SAM2_BUILD_CUDA=1 if building the CUDA operator."
            ) from error

        print(f"[mask_prompt] loading SAM2: {sam2_checkpoint}")
        self._sam2 = build_sam2_video_predictor(
            config_file=sam2_config,
            ckpt_path=sam2_checkpoint,
            device=self.device.type,
        )
        self._sam2.eval()
        for parameter in self._sam2.parameters():
            parameter.requires_grad = False

        total_dino = sum(
            p.numel() for p in self._dino.parameters()
        )
        total_sam2 = sum(
            p.numel() for p in self._sam2.parameters()
        )
        print(
            f"[mask_prompt] frozen parameters: dino={total_dino:,} "
            f"sam2={total_sam2:,} (both eval + no_grad + requires_grad=False)"
        )

    # ------------------------------------------------------------------
    # Grounding-DINO detection + subject selection
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _detect_frame(
        self,
        frame_chw: torch.Tensor,
    ) -> List[Tuple[torch.Tensor, float, str]]:
        """Detect one frame; returns list of (xyxy-pixel box, score, phrase)."""
        boxes_cxcywh, logits, phrases = self._dino_predict(
            model=self._dino,
            image=frame_chw.float().clamp(0.0, 1.0).to(self.device),
            caption=self.text_prompt,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
        )

        height, width = frame_chw.shape[-2], frame_chw.shape[-1]
        results: List[Tuple[torch.Tensor, float, str]] = []
        for box, logit, phrase in zip(boxes_cxcywh, logits, phrases):
            cx, cy, bw, bh = (
                float(box[0]), float(box[1]),
                float(box[2]), float(box[3]),
            )
            x1 = (cx - bw / 2.0) * width
            y1 = (cy - bh / 2.0) * height
            x2 = (cx + bw / 2.0) * width
            y2 = (cy + bh / 2.0) * height
            results.append(
                (
                    torch.tensor(
                        [x1, y1, x2, y2],
                        dtype=torch.float32,
                        device=frame_chw.device,
                    ),
                    float(logit),
                    str(phrase).strip().lower(),
                )
            )
        return results

    def _select_main_subject(
        self,
        per_frame_detections: List[List[Tuple[torch.Tensor, float, str]]],
        height: int,
        width: int,
    ) -> List[Optional[Tuple[torch.Tensor, float, str]]]:
        """Pick one subject box per frame.

        Person boxes are strictly preferred. Among candidates the score is
        ``w_conf * conf + w_area * area_ratio + w_iou * IoU(prev, box)`` so a
        confident, reasonably large box that overlaps the previous frame's
        selection wins, suppressing identity jumps in multi-person clips.
        """
        frame_area = float(height * width)
        selected: List[Optional[Tuple[torch.Tensor, float, str]]] = []
        prev_box: Optional[torch.Tensor] = None

        for detections in per_frame_detections:
            person = [
                item for item in detections
                if "person" in item[2]
            ]
            candidates = person if person else detections
            if not candidates:
                selected.append(None)
                continue

            best_item = None
            best_score = float("-inf")
            for box, conf, phrase in candidates:
                area = (
                    max(0.0, float(box[2] - box[0]))
                    * max(0.0, float(box[3] - box[1]))
                ) / frame_area
                iou = _box_iou_xyxy(box, prev_box)
                score = (
                    self.subject_weight_conf * conf
                    + self.subject_weight_area * min(1.0, area)
                    + self.subject_weight_iou * iou
                )
                if score > best_score:
                    best_score = score
                    best_item = (box, conf, phrase)

            selected.append(best_item)
            prev_box = best_item[0]

        return selected

    # ------------------------------------------------------------------
    # SAM2 video propagation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _sam2_segment_clip(
        self,
        frames_chw: torch.Tensor,
        selected_boxes: List[Optional[torch.Tensor]],
    ) -> Optional[List[Optional[torch.Tensor]]]:
        """Run SAM2 video prediction for one clip.

        ``frames_chw``: [T, 3, H, W] float in [0, 1]. Returns a list of
        [H, W] binary masks per frame (None when a frame has no propagated
        mask), or ``None`` when SAM2 failed for the whole clip.
        """
        num_frames, _, height, width = frames_chw.shape
        anchors = [
            t for t in self.sam2_prompt_frames
            if 0 <= t < num_frames and selected_boxes[t] is not None
        ]
        if not anchors:
            return None

        video_np = (
            (frames_chw.clamp(0.0, 1.0) * 255.0)
            .round()
            .to(torch.uint8)
            .permute(0, 2, 3, 1)  # [T, H, W, 3]
            .cpu()
            .numpy()
            .copy()
        )

        cleanup_dir: Optional[str] = None
        try:
            # The released SAM2 `init_state` only accepts a JPEG frame
            # directory via ``video_path``. Newer builds/forks also accept an
            # in-memory ``video`` array; try that first and fall back to
            # writing the frames to a temp directory.
            try:
                state = self._sam2.init_state(video=video_np)
            except TypeError:
                import tempfile

                from PIL import Image

                cleanup_dir = tempfile.mkdtemp(
                    prefix="sam2_frames_"
                )
                for frame_idx, frame_arr in enumerate(video_np):
                    Image.fromarray(frame_arr).save(
                        os.path.join(
                            cleanup_dir, f"{frame_idx:05d}.jpg"
                        )
                    )
                state = self._sam2.init_state(video_path=cleanup_dir)

            for frame_idx in anchors:
                box = selected_boxes[frame_idx]
                # Box prompt plus a positive point at the box center in a
                # single call: SAM2 accepts box and points together, and it
                # forbids a second prompt call on the same (frame, object).
                # points: [P, 2] pixel xy; labels: [P] (1 = positive point).
                center = torch.tensor(
                    [
                        (float(box[0]) + float(box[2])) / 2.0,
                        (float(box[1]) + float(box[3])) / 2.0,
                    ],
                    dtype=torch.float32,
                    device=self.device,
                )
                self._sam2.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=frame_idx,
                    obj_id=1,
                    points=center.unsqueeze(0),  # [1, 2]
                    labels=torch.tensor(
                        [1], device=self.device
                    ),
                    box=box.to(self.device),
                )

            propagated: List[Optional[torch.Tensor]] = [
                None
            ] * num_frames
            for (
                out_frame_idx,
                _obj_ids,
                out_mask_logits,
            ) in self._sam2.propagate_in_video(state):
                # Single tracked object (obj_id=1): logits [1, 1, H, W].
                mask = (
                    out_mask_logits[0] > 0.0
                )  # [1, H, W] bool
                if bool(mask.any()):
                    propagated[int(out_frame_idx)] = (
                        mask[0].to(torch.float32).cpu()
                    )
            self._sam2.reset_state(state)
            return propagated
        except Exception as error:  # noqa: BLE001 - SAM2 failure is data-level
            if not self.fallback_identity:
                raise
            print(
                f"[mask_prompt] SAM2 failed on one clip "
                f"({type(error).__name__}: {error}); falling back to "
                f"identity masks for this clip."
            )
            return None
        finally:
            if cleanup_dir is not None:
                shutil.rmtree(cleanup_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(self, frames_rgb: torch.Tensor) -> MaskPromptOutput:
        """Generate masks for a batch of clips.

        Args:
            frames_rgb: [B, T, 3, H, W] float tensors in [0, 1] (post
                geometric-transform, pre-CLIP-normalization RGB).
        """
        if frames_rgb.ndim != 5 or frames_rgb.shape[2] != 3:
            raise ValueError(
                "frames_rgb must be [B, T, 3, H, W], got "
                f"{tuple(frames_rgb.shape)}"
            )

        batch, num_frames, _, height, width = frames_rgb.shape
        device = frames_rgb.device

        actor_masks = torch.zeros(
            batch, num_frames, 1, height, width, device=device
        )
        expanded_masks = torch.zeros(
            batch, num_frames, 1, height, width, device=device
        )
        interaction_masks = torch.zeros(
            batch, num_frames, 1, height, width, device=device
        )
        mask_valid = torch.zeros(batch, num_frames, device=device)
        boxes_vis = frames_rgb.clone()
        all_boxes: List[List[Optional[torch.Tensor]]] = []
        all_phrases: List[List[str]] = []

        for b in range(batch):
            self.stats["clips"] += 1
            self.stats["total_frames"] += num_frames
            try:
                clip = frames_rgb[b]  # [T, 3, H, W]
                detections = [
                    self._detect_frame(clip[t])
                    for t in range(num_frames)
                ]
                selected = self._select_main_subject(
                    detections, height, width
                )
                self.stats["dino_frames"] += sum(
                    1 for item in selected if item is not None
                )

                # _sam2_segment_clip expects plain xyxy box tensors, not the
                # (box, conf, phrase) selection tuples.
                propagated = self._sam2_segment_clip(
                    clip,
                    [
                        item[0] if item is not None else None
                        for item in selected
                    ],
                )

                clip_boxes: List[Optional[torch.Tensor]] = []
                clip_phrases: List[str] = []
                for t in range(num_frames):
                    item = selected[t]
                    box = item[0] if item is not None else None
                    score = item[1] if item is not None else 0.0
                    phrase = item[2] if item is not None else ""

                    actor = None
                    if propagated is not None:
                        actor = propagated[t]
                    if actor is None and box is not None:
                        # No propagated mask on this frame (e.g. occlusion):
                        # keep the DINO box itself as a coarse actor mask so
                        # the frame can still be enhanced.
                        actor = _box_to_mask(
                            box, height, width, device
                        )[0]

                    if actor is None or box is None:
                        # No person detected on this frame: mask_valid = 0
                        # and every mask stays empty (identity downstream).
                        clip_boxes.append(None)
                        clip_phrases.append("")
                        boxes_vis[b, t] = _draw_box_on_frame(
                            clip[t], None
                        )
                        continue

                    actor = actor.to(device=device, dtype=torch.float32)
                    if actor.ndim == 2:
                        actor = actor.unsqueeze(0)  # [1, H, W]

                    # Recompute the tight box from the mask (per spec), then
                    # expand by expand_ratio and clip to frame bounds.
                    tight = _mask_to_box_xyxy(actor[0])
                    base_box = tight if tight is not None else box
                    expanded = _expand_box_xyxy(
                        base_box, height, width, self.expand_ratio
                    )
                    expanded_mask = _box_to_mask(
                        expanded, height, width, device
                    )

                    actor_masks[b, t] = actor
                    expanded_masks[b, t] = expanded_mask
                    interaction_masks[b, t] = (
                        expanded_mask * (1.0 - actor)
                    )
                    mask_valid[b, t] = 1.0

                    clip_boxes.append(box.detach().cpu())
                    clip_phrases.append(f"{phrase}:{score:.2f}")
                    boxes_vis[b, t] = _draw_box_on_frame(clip[t], box)

                all_boxes.append(clip_boxes)
                all_phrases.append(clip_phrases)
                self.stats["valid_frames"] += int(
                    mask_valid[b].sum().item()
                )
            except Exception as error:  # noqa: BLE001 - per-clip fallback
                if not self.fallback_identity:
                    raise
                self.stats["identity_fallback_clips"] += 1
                failure_type = type(error).__name__
                self.stats["failure_types"][failure_type] = (
                    self.stats["failure_types"].get(failure_type, 0) + 1
                )
                print(
                    f"[mask_prompt] clip {b} failed "
                    f"({type(error).__name__}: {error}); identity fallback."
                )
                # Reset the whole clip to identity (all-zero masks). A
                # mid-clip failure may have partially written masks, so
                # explicitly zero them instead of relying on initialization.
                actor_masks[b] = 0.0
                expanded_masks[b] = 0.0
                interaction_masks[b] = 0.0
                mask_valid[b] = 0.0
                boxes_vis[b] = frames_rgb[b]
                all_boxes.append([None] * num_frames)
                all_phrases.append([""] * num_frames)
                self._update_consecutive_invalid(0)

        return MaskPromptOutput(
            actor_masks=actor_masks,
            expanded_box_masks=expanded_masks,
            interaction_masks=interaction_masks,
            mask_valid=mask_valid,
            dino_boxes_vis=boxes_vis,
            dino_boxes=all_boxes,
            dino_phrases=all_phrases,
        )

    def _update_consecutive_invalid(self, clip_valid_frames: int) -> None:
        """Track clips without any valid mask; abort after N consecutive."""
        if clip_valid_frames > 0:
            self._consecutive_invalid_clips = 0
            return
        self._consecutive_invalid_clips += 1
        if (
            self.max_consecutive_invalid_clips > 0
            and self._consecutive_invalid_clips
            >= self.max_consecutive_invalid_clips
        ):
            raise RuntimeError(
                f"[mask_prompt] {self._consecutive_invalid_clips} "
                "consecutive clips produced no valid mask "
                f"(failure types so far: {self.stats['failure_types']}, "
                f"valid_frame_rate="
                f"{self.stats['valid_frames']}/"
                f"{self.stats['total_frames']}). Stopping training: the "
                "mask generator is effectively broken on this data."
            )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_mask_prompt_generator(
    args: Any,
    device: str = "cuda",
) -> Optional[BaseMaskPromptGenerator]:
    """Build the configured generator, or ``None`` when disabled.

    ``args`` only needs the ``mask_prompt_*`` attributes (the training
    script's argparse namespace satisfies this).
    """
    if not getattr(args, "mask_prompt_enabled", False):
        return None

    if not getattr(args, "mask_prompt_online", True):
        return OfflineCachedMaskPromptGenerator(
            cache_dir=getattr(args, "mask_prompt_cache_dir", ""),
        )

    prompt_frames = sorted(
        {
            int(x)
            for x in str(
                getattr(args, "mask_prompt_sam2_prompt_frames", "0,4,7")
            ).split(",")
            if str(x).strip()
        }
    )

    try:
        return OnlineMaskPromptGenerator(
            text_prompt=getattr(args, "mask_prompt_text_prompt", "person ."),
            box_threshold=getattr(args, "mask_prompt_box_threshold", 0.35),
            text_threshold=getattr(args, "mask_prompt_text_threshold", 0.25),
            expand_ratio=getattr(args, "mask_prompt_expand_ratio", 0.30),
            sam2_prompt_frames=prompt_frames,
            fallback_identity=getattr(args, "mask_prompt_fallback_identity", True),
            dino_config=getattr(args, "mask_prompt_dino_config", ""),
            dino_checkpoint=getattr(args, "mask_prompt_dino_checkpoint", ""),
            sam2_config=getattr(
                args,
                "mask_prompt_sam2_config",
                "configs/sam2.1/sam2.1_hiera_t.yaml",
            ),
            sam2_checkpoint=getattr(args, "mask_prompt_sam2_checkpoint", ""),
            device=device,
            max_consecutive_invalid_clips=int(
                getattr(
                    args,
                    "mask_prompt_max_consecutive_invalid_clips",
                    20,
                )
            ),
        )
    except Exception as error:
        # Startup failures (missing dependency, bad config, unreadable
        # checkpoint) must abort training with a clear RuntimeError; silently
        # degrading every clip to identity is not allowed.
        raise RuntimeError(
            "--mask_prompt_enabled=true but the online Grounding-DINO + "
            "SAM2 mask generator failed to initialize "
            f"({type(error).__name__}: {error}). Fix the dependency or "
            "checkpoint before training; silent identity degradation is "
            "not permitted."
        ) from error
