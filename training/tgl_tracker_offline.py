"""Offline CoTracker3 trajectory preprocessing for TGL-Adapter.

Run BEFORE training on every video; the tracker is never placed inside the
training graph and never run dynamically at train/test time. For each video
this script saves a ``.pt`` file with exactly the contract the online module
(``TGL_adapter.py``) expects:

    {
      "tracks":        FloatTensor [K, T, 2]   # (x, y) in preprocessed image coords
      "visibility":    FloatTensor [K, T]      # in [0, 1]
      "birth_frame":   LongTensor  [K]         # trajectory exists for t >= birth
      "frame_indices": LongTensor  [T]         # which source frames were sampled
      "image_size":    [H, W]                  # e.g. [224, 224]
      "preprocess_config": {...}               # resize/crop/fps info
    }

Pipeline (TAT-inspired, multi-frame birth)
------------------------------------------
1. Load the video and apply the SAME preprocessing as train/test (resize to
   224x224 by default), so tracker coordinates live in the coordinate system
   the ViT sees.
2. Initialize query points on MULTIPLE frames (not only frame 0): every
   ``query_frame_stride`` frames (T=8, stride=2 -> births at 0, 2, 4, 6).
   On each birth frame, one query per ViT patch center (14x14 = 196 points
   for ViT-B/16), so every patch gets a chance to be tracked from the frame
   where it first appears.
3. Track all queries with CoTracker3 OFFLINE over all sampled frames.
4. For later-born trajectories, positions before ``birth_frame`` are marked
   invisible (the online module additionally masks them out of attention).
5. Deduplicate: two trajectories whose average distance over co-visible
   frames is < ``delta`` px AND whose mean motion directions are similar
   (cosine > ``dir_thresh``) are redundant; keep the earlier-born one (tie:
   higher mean visibility).
6. Keep at most K trajectories (default 64/128):
   - sampler "tat_random": uniform random choice after dedup (TAT-style,
     seeded) - the DEFAULT first version.
   - sampler "improved": score = motion_score + 0.5 * visibility_score
     + 0.3 * semantic_score, top-K. The semantic score (optional) is the
     cosine similarity between the last-layer CLIP/ViT patch token at the
     trajectory position and that frame's CLS token. It is ONLY used to
     choose query points - it never touches the tracker and never enters the
     online module.

Usage
-----
    python training/tgl_tracker_offline.py --input /path/video.mp4 \
        --out outputs/tracks/video.pt --num-frames 8 --max-tracks 64

The CoTracker3 model is loaded lazily via ``torch.hub`` (``cotracker3_offline``);
pass ``--hub-repo`` to point at a local checkout, or call ``build_tracker``/
``process_video`` programmatically with a pre-built tracker to stay fully
offline. Nothing in this file is imported by the training graph.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

__all__ = [
    "build_tracker",
    "load_frames",
    "make_query_points",
    "run_tracker",
    "dedup_trajectories",
    "motion_scores",
    "visibility_scores",
    "semantic_scores",
    "select_trajectories",
    "process_video",
    "track_cache_path",
    "TGL_PRESETS",
    "resolve_tgl_preset",
]


# --------------------------------------------------------------------------- #
#  Stable cache paths + TGL presets                                            #
# --------------------------------------------------------------------------- #
def track_cache_path(track_root: str, rel_path: str) -> str:
    """Stable per-video cache path: sha1 of the normalized relative path.

    ``rel_path`` is normalized (posix separators, stripped) before hashing, so
    the same video always maps to the same cache file regardless of how the
    list line was written. The mapping is injective for practical purposes
    and avoids filename collisions between datasets.
    """
    key = os.path.normpath(str(rel_path).strip().replace("\\", "/")).lstrip("/")
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return os.path.join(track_root, f"{digest}.pt")


# TGL presets (requirement: tat_like first). "gather"/"scatter" name the
# grid alignment mode of the ONLINE adapter; "sampler"/"score_sampler"/
# "semantic_sampler" control the OFFLINE query-point selection; "bilinear"
# and "local_tube_size" mirror the online TGLAdapter arguments.
TGL_PRESETS: Dict[str, Dict[str, object]] = {
    "tat_like": {
        "gather": "nearest",
        "scatter": "nearest",
        "sampler": "random",
        "local_tube_size": 1,
        "bilinear": False,
        "score_sampler": False,
        "semantic_sampler": False,
    },
    "local3": {
        "gather": "nearest",
        "scatter": "nearest",
        "sampler": "random",
        "local_tube_size": 3,
        "bilinear": False,
        "score_sampler": False,
        "semantic_sampler": False,
    },
    "improved": {
        "gather": "bilinear",
        "scatter": "bilinear",
        "sampler": "score",
        "local_tube_size": 3,
        "bilinear": True,
        "score_sampler": True,
        "semantic_sampler": True,
    },
}


def resolve_tgl_preset(preset: str) -> Dict[str, object]:
    """Return a validated copy of the named preset."""
    if preset not in TGL_PRESETS:
        raise ValueError(
            f"Unknown TGL preset {preset!r}; available: {sorted(TGL_PRESETS)}"
        )
    resolved = dict(TGL_PRESETS[preset])
    # Online adapter mode: bilinear=True forces bilinear gather/scatter.
    if bool(resolved["bilinear"]):
        resolved["gather"] = "bilinear"
        resolved["scatter"] = "bilinear"
    # Offline sampler: score_sampler wins over the plain random sampler.
    resolved["offline_sampler"] = (
        "improved" if bool(resolved["score_sampler"]) else "tat_random"
    )
    return resolved


# --------------------------------------------------------------------------- #
#  Tracker / video loading                                                     #
# --------------------------------------------------------------------------- #
def build_tracker(
    device: str = "cuda",
    version: str = "cotracker3_offline",
    hub_repo: str = "facebookresearch/co-tracker",
) -> torch.nn.Module:
    """Load the (frozen) CoTracker3 offline tracker.

    ``hub_repo`` may be a local path to a checkout of co-tracker (then
    ``torch.hub.load(..., source="local")`` is used). The tracker is used in
    eval mode with ``requires_grad_(False)``; it must NEVER be trained.
    """
    if os.path.isdir(hub_repo):
        model: torch.nn.Module = torch.hub.load(  # type: ignore[assignment]
            hub_repo, version, source="local", trust_repo=True
        )
    else:
        model = torch.hub.load(hub_repo, version, trust_repo=True)  # type: ignore[assignment]
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _evenly_spaced_indices(n_total: int, num_frames: int) -> List[int]:
    if num_frames >= n_total:
        return list(range(n_total))
    # Inclusive endpoints, uniform spacing (matches common train/eval sampling).
    return [round(i * (n_total - 1) / max(1, num_frames - 1)) for i in range(num_frames)]


def load_frames(
    path: str,
    num_frames: int = 8,
    resize: Tuple[int, int] = (224, 224),
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, object]]:
    """Load + preprocess a video file or a frame directory.

    Returns (frames [T, 3, H, W] float32 in [0,1], frame_indices [T] long,
    preprocess_config dict).
    """
    resize = (int(resize[0]), int(resize[1]))
    if os.path.isdir(path):
        from PIL import Image

        names = sorted(
            f for f in os.listdir(path)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        )
        if not names:
            raise FileNotFoundError(f"no frames found in {path}")
        idx = _evenly_spaced_indices(len(names), num_frames)
        sel = [names[i] for i in idx]
        imgs = []
        for n in sel:
            with Image.open(os.path.join(path, n)) as im:
                imgs.append(im.convert("RGB"))
        frames = torch.stack(
            [torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0) for im in imgs]
        )  # [T, H, W, 3]
        frames = frames.permute(0, 3, 1, 2)  # [T, 3, H, W]
        src = "frames_dir"
        fps = 0.0
    else:
        from torchvision.io import read_video

        v, _, info = read_video(path, pts_unit="sec")  # [T, H, W, C] uint8
        idx = _evenly_spaced_indices(int(v.shape[0]), num_frames)
        frames = v[idx].permute(0, 3, 1, 2).float() / 255.0  # [T, 3, H, W]
        src = "video_file"
        fps = float(info.get("video_fps", 0.0))

    if tuple(frames.shape[-2:]) != resize:
        frames = F.interpolate(
            frames, size=resize, mode="bilinear", align_corners=False
        )
    config = {
        "source_type": src,
        "source_path": os.path.abspath(path),
        "resize": list(resize),
        "resized": True,  # pure resize (no crop), train/test-consistent coords
        "source_fps": fps,
        "num_frames": int(frames.shape[0]),
    }
    return frames, torch.tensor(idx, dtype=torch.long), config


# --------------------------------------------------------------------------- #
#  Query initialization (TAT-style multi-frame births on patch centers)        #
# --------------------------------------------------------------------------- #
def make_query_points(
    num_frames: int,
    patch_grid: int = 14,
    image_size: Tuple[int, int] = (224, 224),
    query_frame_stride: int = 2,
) -> torch.Tensor:
    """One query per patch center on every ``query_frame_stride``-th frame.

    Returns queries [S, 3] with rows (frame_idx, x, y) in image coords.
    T=8, stride=2 -> births at frames 0, 2, 4, 6; each contributes
    patch_grid^2 points (196 for ViT-B/16), so S = 4 * 196 = 784.
    """
    H, W = int(image_size[0]), int(image_size[1])
    cell_y, cell_x = H / float(patch_grid), W / float(patch_grid)
    ys = (torch.arange(patch_grid, dtype=torch.float32) + 0.5) * cell_y
    xs = (torch.arange(patch_grid, dtype=torch.float32) + 0.5) * cell_x
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # [G, G]
    centers = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)  # [G*G, 2]

    birth_frames = list(range(0, num_frames, max(1, int(query_frame_stride))))
    qs = []
    for f in birth_frames:
        q = torch.cat(
            [torch.full((centers.shape[0], 1), float(f)), centers], dim=-1
        )  # [G*G, 3]
        qs.append(q)
    return torch.cat(qs, dim=0)  # [S, 3]


def run_tracker(
    tracker: torch.nn.Module,
    frames: torch.Tensor,
    queries: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Track queries over all frames (no grad - tracker is never trained).

    Input frames are [T, 3, H, W] floats in [0, 1]; the CoTracker3 model
    internally normalizes with ``2 * (video / 255) - 1``, so it expects the
    raw [0, 255] scale - we rescale here.

    CoTracker3 OFFLINE returns frame-major tensors: coords [B, T, N, 2] and
    visibility [B, T, N]. We transpose to query-major ([S, T, ...]) which is
    the layout every downstream step (birth masking, dedup, sampling) uses.

    Returns (tracks [S, T, 2], visibility [S, T]) in the preprocessed image
    coordinate system. Tracker runs under torch.no_grad() and eval mode.
    """
    device = next(tracker.parameters()).device
    video = (frames * 255.0).unsqueeze(0).to(device)  # [1, T, 3, H, W] in [0, 255]
    q = queries.to(device).float().unsqueeze(0)  # [1, S, 3]
    with torch.no_grad():
        tracks, visibility = tracker(video, queries=q)
    # [B, T, N, 2] -> [S, T, 2]; [B, T, N] -> [S, T]
    tracks = tracks.permute(0, 2, 1, 3)[0].float().cpu()
    visibility = visibility.permute(0, 2, 1)[0].float().cpu()
    return tracks, visibility


# --------------------------------------------------------------------------- #
#  Deduplication                                                               #
# --------------------------------------------------------------------------- #
def _pairwise_stats(
    tracks: torch.Tensor,
    visibility: torch.Tensor,
    vis_thr: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized pairwise statistics for all S x S trajectory pairs.

    Returns:
        avg_dist [S, S]: average L2 distance over co-visible frames
            (inf where a pair is never co-visible);
        cos [S, S]: cosine between the per-track MEAN DISPLACEMENT
            directions (each track's mean runs over its own visible
            consecutive frame pairs - equivalent to the original pairwise
            mean when both tracks are visible throughout, and a faithful
            approximation otherwise; the direction gate only ever applies
            when the distance gate already passed);
        dir_ok [S, S]: both tracks have at least one visible displacement.
    """
    S, T, _ = tracks.shape
    vis_b = visibility > vis_thr  # [S, T]
    # Pairwise per-frame distances: [S, S, T].
    diff = tracks[:, None, :, :] - tracks[None, :, :, :]
    d = torch.sqrt((diff * diff).sum(-1).clamp_min(1e-12))
    both = vis_b[:, None, :] & vis_b[None, :, :]  # [S, S, T]
    cnt = both.sum(-1)
    avg_dist = torch.where(
        cnt > 0,
        (d * both).sum(-1) / cnt.clamp_min(1).float(),
        torch.full_like(d[..., 0], float("inf")),
    )
    # Per-track mean displacement direction over visible consecutive pairs.
    disp = tracks[:, 1:] - tracks[:, :-1]  # [S, T-1, 2]
    w = (vis_b[:, 1:] & vis_b[:, :-1]).float()  # [S, T-1]
    mean_dir = (disp * w.unsqueeze(-1)).sum(1) / w.sum(-1).clamp_min(1).unsqueeze(-1)
    unit = mean_dir / mean_dir.norm(dim=-1).clamp_min(1e-8).unsqueeze(-1)
    cos = unit @ unit.t()  # [S, S]
    dir_ok = (w.sum(-1) > 0)
    dir_ok = dir_ok[:, None] & dir_ok[None, :]
    return avg_dist, cos, dir_ok


def dedup_trajectories(
    tracks: torch.Tensor,
    visibility: torch.Tensor,
    birth_frame: torch.Tensor,
    delta: float = 8.0,
    dir_thresh: float = 0.9,
    vis_thr: float = 0.5,
) -> torch.Tensor:
    """Remove redundant trajectories; keep earlier birth (tie: higher mean vis).

    Redundant = average distance over co-visible frames < ``delta`` px AND
    mean motion directions similar (cosine > ``dir_thresh``; pairs with no
    visible displacement on either side are treated as direction-similar).

    Fully vectorized: the S x S statistics are computed with tensor ops and
    the greedy priority loop checks each candidate against ALL kept
    trajectories with a single masked reduction (the naive per-pair Python
    loop took ~45 s for S=784; this version is ~0.2 s).

    Returns LongTensor [K_kept] of kept indices (original order).
    """
    S = tracks.shape[0]
    if S <= 1:
        return torch.arange(S, dtype=torch.long)
    avg_dist, cos, dir_ok = _pairwise_stats(tracks, visibility, vis_thr)
    mean_vis = visibility.mean(dim=1)  # [S]
    # Process early births first so survivors are the earlier-born ones.
    order = sorted(range(S), key=lambda i: (int(birth_frame[i]), -float(mean_vis[i])))
    kept_ids: List[int] = []
    for i in order:
        if kept_ids:
            k = torch.tensor(kept_ids, dtype=torch.long)
            close = avg_dist[i, k] < delta
            similar = (~dir_ok[i, k]) | (cos[i, k] > dir_thresh)
            if bool((close & similar).any()):
                continue
        kept_ids.append(i)
    return torch.tensor(sorted(kept_ids), dtype=torch.long)


# --------------------------------------------------------------------------- #
#  Samplers                                                                    #
# --------------------------------------------------------------------------- #
def motion_scores(tracks: torch.Tensor, visibility: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor:
    """Normalized mean speed per trajectory [S], min-max scaled to [0, 1]."""
    H, W = int(image_size[0]), int(image_size[1])
    diag = float(np.hypot(H, W))
    disp = (tracks[:, 1:] - tracks[:, :-1]).norm(dim=-1)  # [S, T-1]
    vis_pair = torch.sqrt(visibility[:, 1:] * visibility[:, :-1]).clamp_min(1e-6)
    speed = (disp / diag / vis_pair).sum(dim=1) / vis_pair.sum(dim=1).clamp_min(1e-6)
    lo, hi = float(speed.min()), float(speed.max())
    return (speed - lo) / (hi - lo) if hi > lo else torch.zeros_like(speed)


def visibility_scores(visibility: torch.Tensor) -> torch.Tensor:
    """Mean visibility per trajectory [S], min-max scaled to [0, 1]."""
    m = visibility.mean(dim=1)
    lo, hi = float(m.min()), float(m.max())
    return (m - lo) / (hi - lo) if hi > lo else torch.zeros_like(m)


def semantic_scores(
    tracks: torch.Tensor,
    visibility: torch.Tensor,
    semantic_map_fn: Callable[[torch.Tensor], torch.Tensor],
    frames: torch.Tensor,
    image_size: Tuple[int, int],
) -> torch.Tensor:
    """Optional semantic score [S] in [0, 1]: mean over valid frames of the
    cosine similarity between the CLIP patch token under the trajectory point
    and that frame's CLS token.

    ``semantic_map_fn(frames [T,3,H,W]) -> [T, G, G]`` similarity map (e.g.
    from the last-layer CLIP/ViT patch tokens vs the CLS token). It is used
    ONLY here, to rank query points - the tracker never sees it and the online
    module never receives it.
    """
    with torch.no_grad():
        sim = semantic_map_fn(frames).float()  # [T, G, G]
    T, G, _ = sim.shape
    H, W = int(image_size[0]), int(image_size[1])
    gx = ((tracks[..., 0].clamp(0, W - 1)) / W * G).long().clamp(0, G - 1)
    gy = ((tracks[..., 1].clamp(0, H - 1)) / H * G).long().clamp(0, G - 1)
    vals = sim.permute(1, 2, 0)[gy, gx, torch.arange(T)[None, :]]  # hmm, see below
    # vals: [S, T] via advanced indexing: sim[t, gy[s,t], gx[s,t]]
    s = (vals * visibility).sum(dim=1) / visibility.sum(dim=1).clamp_min(1e-6)
    lo, hi = float(s.min()), float(s.max())
    return (s - lo) / (hi - lo) if hi > lo else torch.zeros_like(s)


def select_trajectories(
    tracks: torch.Tensor,
    visibility: torch.Tensor,
    birth_frame: torch.Tensor,
    max_tracks: int = 64,
    sampler: str = "tat_random",
    image_size: Tuple[int, int] = (224, 224),
    seed: int = 0,
    semantic_map_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    frames: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Cap the deduplicated trajectory set to ``max_tracks``.

    sampler="tat_random": seeded uniform random choice (TAT-style, DEFAULT).
    sampler="improved": top-K by motion + 0.5 * visibility + 0.3 * semantic.

    Returns LongTensor [<= max_tracks] of selected indices.
    """
    S = tracks.shape[0]
    if S <= max_tracks:
        return torch.arange(S, dtype=torch.long)
    if sampler == "tat_random":
        g = torch.Generator().manual_seed(int(seed))
        perm = torch.randperm(S, generator=g)[:max_tracks]
        return perm.sort().values
    if sampler == "improved":
        score = motion_scores(tracks, visibility, image_size)
        score = score + 0.5 * visibility_scores(visibility)
        if semantic_map_fn is not None and frames is not None:
            score = score + 0.3 * semantic_scores(
                tracks, visibility, semantic_map_fn, frames, image_size
            )
        return score.topk(max_tracks).indices.sort().values
    raise ValueError(f"unknown sampler {sampler!r} (use 'tat_random'|'improved')")


# --------------------------------------------------------------------------- #
#  Full per-video pipeline                                                     #
# --------------------------------------------------------------------------- #
def process_video(
    input_path: str,
    out_path: str,
    tracker: Optional[torch.nn.Module] = None,
    num_frames: int = 8,
    resize: Tuple[int, int] = (224, 224),
    patch_grid: int = 14,
    query_frame_stride: int = 2,
    delta: float = 8.0,
    dir_thresh: float = 0.9,
    max_tracks: int = 64,
    sampler: str = "tat_random",
    seed: int = 0,
    semantic_map_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, object]:
    """Extract, dedup, select and save trajectories for one video."""
    frames, frame_indices, preprocess_config = load_frames(
        input_path, num_frames=num_frames, resize=resize
    )
    T = frames.shape[0]

    if tracker is None:
        tracker = build_tracker(device)
    queries = make_query_points(
        T, patch_grid=patch_grid, image_size=resize, query_frame_stride=query_frame_stride
    )
    trk, vis = run_tracker(tracker, frames, queries)  # [S,T,2], [S,T]

    # Birth frame per query = the frame the query was initialized on; mark
    # pre-birth positions invisible (later-born; masked online anyway).
    birth_frame = queries[:, 0].long()
    pre = torch.arange(T)[None, :] < birth_frame[:, None]
    vis = vis.clone()
    vis[pre] = 0.0

    keep = dedup_trajectories(trk, vis, birth_frame, delta=delta, dir_thresh=dir_thresh)
    trk, vis, birth_frame = trk[keep], vis[keep], birth_frame[keep]

    sel = select_trajectories(
        trk, vis, birth_frame,
        max_tracks=max_tracks,
        sampler=sampler,
        image_size=resize,
        seed=seed,
        semantic_map_fn=semantic_map_fn,
        frames=frames,
    )
    payload = {
        "tracks": trk[sel].float(),                 # [K, T, 2]
        "visibility": vis[sel].float(),             # [K, T]
        "birth_frame": birth_frame[sel].long(),     # [K]
        "frame_indices": frame_indices.long(),      # [T]
        "image_size": list(resize),
        "preprocess_config": {
            **preprocess_config,
            "patch_grid": int(patch_grid),
            "query_frame_stride": int(query_frame_stride),
            "delta_px": float(delta),
            "dir_thresh": float(dir_thresh),
            "max_tracks": int(max_tracks),
            "sampler": str(sampler),
            "seed": int(seed),
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    torch.save(payload, out_path)
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description="Offline CoTracker3 trajectories for TGL-Adapter")
    ap.add_argument("--input", required=True, help="video file OR frames directory")
    ap.add_argument("--out", required=True, help="output .pt path")
    ap.add_argument("--num-frames", type=int, default=8)
    ap.add_argument("--resize", type=int, nargs=2, default=[224, 224])
    ap.add_argument("--patch-grid", type=int, default=14, help="ViT patch grid (14 for ViT-B/16)")
    ap.add_argument("--query-frame-stride", type=int, default=2, help="birth frames every k frames")
    ap.add_argument("--delta", type=float, default=8.0, help="dedup distance threshold (px)")
    ap.add_argument("--dir-thresh", type=float, default=0.9, help="dedup direction cosine threshold")
    ap.add_argument("--max-tracks", type=int, default=64)
    ap.add_argument("--sampler", choices=["tat_random", "improved"], default="tat_random")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--hub-repo", default="facebookresearch/co-tracker",
                    help="torch.hub repo or local co-tracker checkout path")
    ap.add_argument("--tracker-version", default="cotracker3_offline")
    args = ap.parse_args()

    tracker = build_tracker(args.device, version=args.tracker_version, hub_repo=args.hub_repo)
    payload = process_video(
        args.input, args.out, tracker=tracker,
        num_frames=args.num_frames, resize=tuple(args.resize),
        patch_grid=args.patch_grid, query_frame_stride=args.query_frame_stride,
        delta=args.delta, dir_thresh=args.dir_thresh,
        max_tracks=args.max_tracks, sampler=args.sampler,
        seed=args.seed, device=args.device,
    )
    print(
        f"saved {payload['tracks'].shape[0]} tracks -> {args.out} "
        f"(T={payload['tracks'].shape[1]}, image_size={payload['image_size']})"
    )


if __name__ == "__main__":
    main()
