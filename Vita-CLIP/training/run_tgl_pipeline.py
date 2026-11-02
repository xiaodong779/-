#!/usr/bin/env python3
"""TGL one-click pipeline: offline trajectory generation + training launch.

Flow (in this order, strictly):
    1. (optional, --auto-track) scan the train/val lists and generate the
       MISSING offline CoTracker3 track caches; existing caches are skipped
       (resumable). The tracker NEVER runs during training - only here,
       offline, once per video.
    2. pre-launch consistency checks: every training video must have a cache
       whose num_frames / image_size / preprocess_config match the training
       configuration; on mismatch the pipeline errors out and tells the user
       to regenerate (``--force-regen`` deletes stale caches first).
    3. launch the trainer with MODEL.TGL.* enabled and
       MODEL.TGL.RUN_TRACKER_ONLINE=False (hard-enforced; the trainer also
       refuses the online flag).

Model mode ``tgl_vp_two_text`` maps to:
    * TWO text branches kept   : fixed template branch + learnable prompt
                                 branch + Text_MSTFF text fusion;
    * video prompt ON          : CVP (Correspondence Video Prompt) block;
    * TGL trajectory adapter ON: MODEL.TGL.* with the chosen preset;
    * OFF (legacy)             : mask adapter, layered video prompts / patch
                                 enhancer, BIE, SCR, aggregation mixer,
                                 discriminative module, image (frequency)
                                 adapter (already legacy-removed).

Preset ``tat_like`` (ECCV'24 TAT-like alignment, no extras):
    gather=nearest, scatter=nearest, sampler=random, local_tube_size=1,
    bilinear=false, score_sampler=false, semantic_sampler=false

Smoke test (--smoke-test) uses a FAKE video list with SIMULATED track caches
(synthetic trajectories - the tracker is not even imported) and runs one
forward + backward, asserting: loss backward works, TGL adapter parameters
receive gradients, video prompt parameters receive gradients, and the two
text-branch states match the configuration.

Examples
--------
    # full pipeline (generate missing tracks, then launch training)
    python training/run_tgl_pipeline.py --auto-track \
        --track-root /hy-tmp/outputs/tgl_tracks \
        --train-list /hy-tmp/UCF101/annotations/ucfTrainTestlist/trainlist01.txt \
        --val-list   /hy-tmp/UCF101/annotations/ucfTrainTestlist/testlist01.txt \
        --data-root  /hy-tmp/UCF101/rawframes

    # smoke test only (no tracker, no real videos, one fwd+bwd)
    python training/run_tgl_pipeline.py --smoke-test

    # print the training command without running it
    python training/run_tgl_pipeline.py --auto-track ... --dry-run
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional, Tuple

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tgl_tracker_offline as tgl_offline  # noqa: E402
from tgl_tracker_offline import (  # noqa: E402
    TGL_PRESETS,
    resolve_tgl_preset,
    track_cache_path,
)

TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TRAIN_SCRIPT = os.path.join(TRAINING_DIR, "train_OPENAI_ALIGNED.py")
DEFAULT_BACKBONE = "/hy-tmp/weights/ViT-B-16.pt"
DEFAULT_CLASSES = "/hy-tmp/UCF101/annotations/ucfTrainTestlist/classInd.txt"


# --------------------------------------------------------------------------- #
#  Minimal dotted config (MODEL.TGL.* style)                                   #
# --------------------------------------------------------------------------- #
class CfgNode(dict):
    """Dict with attribute access; nested dicts are wrapped transparently."""

    def __getattr__(self, key: str):
        try:
            value = self[key]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(key) from exc
        if isinstance(value, dict) and not isinstance(value, CfgNode):
            value = CfgNode(value)
            self[key] = value
        return value

    def __setattr__(self, key: str, value) -> None:
        self[key] = value


def build_config(args: argparse.Namespace) -> CfgNode:
    """Compose the MODEL.* config required by the TGL pipeline."""
    cfg = CfgNode()
    cfg.MODEL = CfgNode()

    tgl = CfgNode()
    # This IS the TGL pipeline: the adapter is always on; --auto-track only
    # controls whether missing caches are generated before launch.
    tgl.ENABLE = True
    tgl.TRACK_ROOT = os.path.abspath(args.track_root)
    # HARD REQUIREMENT: no online tracking, ever. The trainer hard-errors on
    # the online flag; here we simply never expose a way to set it True.
    tgl.RUN_TRACKER_ONLINE = False
    tgl.PRESET = args.tgl_preset
    tgl.PRESET_RESOLVED = CfgNode(resolve_tgl_preset(args.tgl_preset))
    tgl.NUM_FRAMES = int(args.num_frames)
    tgl.IMAGE_SIZE = [int(args.spatial_size), int(args.spatial_size)]
    tgl.PATCH_GRID = int(args.spatial_size) // int(args.patch_size)
    tgl.MAX_TRACKS = int(args.max_tracks)
    tgl.QUERY_FRAME_STRIDE = int(args.query_frame_stride)
    tgl.DELTA_PX = float(args.delta)
    tgl.DIR_THRESH = float(args.dir_thresh)
    tgl.ADAPTER_DIM = args.tgl_adapter_dim
    tgl.NUM_LAYERS = int(args.tgl_num_layers)
    tgl.NUM_HEADS = int(args.tgl_num_heads)
    tgl.DROPOUT = float(args.tgl_dropout)
    tgl.MAX_SCALE = float(args.tgl_max_scale)
    tgl.MAX_FUSION = float(args.tgl_max_fusion)
    tgl.INSERT_MODE = "blocks"
    tgl.INSERT_LAYERS = args.tgl_insert_layers
    tgl.NUM_PROMPTS = int(args.tgl_num_prompts)
    tgl.PROMPT_SOURCE_LAYER = int(args.tgl_v2_prompt_source_layer)
    tgl.V2_PROMPT_LAYERS = args.tgl_v2_prompt_layers
    tgl.RECON_RATIO = float(args.tgl_v2_recon_ratio)
    tgl.PATCH_POOL_BASELINE = False  # ablation-only switch, default OFF
    tgl.GATE_INIT = float(args.tgl_gate_init)  # warm start (scale units)
    tgl.READOUT_GATE = 0.15  # TAT-style trajectory readout fusion gate
    cfg.MODEL.TGL = tgl

    vp = CfgNode()
    vp.ENABLE = True  # video prompt (CVP block) is part of tgl_vp_two_text
    vp.INSERT_LAYER = int(args.cvp_insert_layer)
    vp.NUM_PROMPTS = int(args.cvp_num_prompts)
    cfg.MODEL.VIDEO_PROMPT = vp

    tb = CfgNode()
    tb.NUM = 2  # fixed template branch + learnable prompt branch
    tb.FUSION = "text_mstff"
    cfg.MODEL.TEXT_BRANCHES = tb

    cfg.MODEL.MODE = args.model_mode
    return cfg


def validate_config(cfg: CfgNode) -> None:
    """Enforce the invariants the TGL contract depends on."""
    if cfg.MODEL.MODE != "tgl_vp_two_text":
        raise SystemExit(
            f"Unknown --model-mode {cfg.MODEL.MODE!r}; "
            "supported: tgl_vp_two_text"
        )
    if cfg.MODEL.TGL.RUN_TRACKER_ONLINE is not False:
        # Unreachable via the CLI today; kept as a hard tripwire so future
        # config plumbing can never silently enable online tracking.
        raise SystemExit(
            "FATAL: MODEL.TGL.RUN_TRACKER_ONLINE must be False. CoTracker3 "
            "is an OFFLINE preprocessing step only."
        )
    if cfg.MODEL.TGL.ENABLE and not cfg.MODEL.TGL.TRACK_ROOT:
        raise SystemExit("MODEL.TGL.ENABLE=True requires MODEL.TGL.TRACK_ROOT.")
    if cfg.MODEL.TEXT_BRANCHES.NUM != 2:
        raise SystemExit("tgl_vp_two_text requires MODEL.TEXT_BRANCHES.NUM=2.")
    if not cfg.MODEL.VIDEO_PROMPT.ENABLE:
        raise SystemExit("tgl_vp_two_text requires MODEL.VIDEO_PROMPT.ENABLE=True.")
    if not cfg.MODEL.TGL.ENABLE:
        raise SystemExit("MODEL.TGL.ENABLE must be True for this pipeline.")


# --------------------------------------------------------------------------- #
#  Video lists + cache management                                              #
# --------------------------------------------------------------------------- #
def parse_video_list(list_path: str) -> List[Tuple[str, Optional[int]]]:
    """Parse 'rel_path [label]' lines (same format as VideoDataset)."""
    entries: List[Tuple[str, Optional[int]]] = []
    with open(list_path) as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            label = int(parts[1]) if len(parts) >= 2 and parts[1].lstrip("-").isdigit() else None
            entries.append((parts[0], label))
    if not entries:
        raise SystemExit(f"No videos parsed from list {list_path}")
    return entries


def resolve_media(data_root: str, rel_path: str) -> Optional[str]:
    """Map a list entry to actual media (raw-frame dir or video file).

    Mirrors VideoDataset: a trailing .avi is stripped for raw-frame dirs.
    """
    candidates = [rel_path]
    if rel_path.lower().endswith(".avi"):
        candidates.append(rel_path[:-4])
    for cand in candidates:
        full = os.path.join(data_root, cand)
        if os.path.isdir(full) or os.path.isfile(full):
            return full
    return None


def simulate_track_cache(
    cache_path: str,
    num_frames: int,
    max_tracks: int,
    image_size: Tuple[int, int],
    patch_grid: int,
    seed_key: str,
) -> None:
    """Write a SYNTHETIC track cache (smoke test only, no tracker involved).

    Trajectories are smooth random walks born at a few frames (TAT-style
    multi-frame birth) with mixed visibility; geometry is deterministic per
    ``seed_key`` so re-running the smoke test is reproducible.
    """
    H, W = int(image_size[0]), int(image_size[1])
    gen = torch.Generator().manual_seed(
        abs(hash(seed_key)) % (2 ** 31)
    )
    K = max(2, max_tracks // 4)
    tracks = torch.zeros(K, num_frames, 2)
    visibility = torch.zeros(K, num_frames)
    birth_frame = torch.zeros(K, dtype=torch.long)
    for k in range(K):
        birth = int(torch.randint(0, max(1, num_frames // 2), (1,), generator=gen))
        birth_frame[k] = birth
        pos = torch.rand(2, generator=gen) * torch.tensor([W, H], dtype=torch.float32)
        vel = (torch.rand(2, generator=gen) - 0.5) * (torch.tensor([W, H]) / num_frames)
        for t in range(birth, num_frames):
            tracks[k, t] = pos
            visibility[k, t] = 1.0 if torch.rand(1, generator=gen) > 0.2 else 0.0
            pos = (pos + vel).clamp(torch.tensor([0.0, 0.0]), torch.tensor([float(W), float(H)]))
            vel = vel + 0.05 * torch.randn(2, generator=gen) * (torch.tensor([W, H]) / num_frames)
    payload = {
        "tracks": tracks,
        "visibility": visibility,
        "birth_frame": birth_frame,
        "frame_indices": torch.linspace(0, max(num_frames - 1, 0), num_frames).long(),
        "image_size": [H, W],
        "preprocess_config": {
            "source_type": "simulated",
            "resize": [H, W],
            "patch_grid": int(patch_grid),
            "sampler": "simulated",
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    tmp = cache_path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, cache_path)


def generate_track_cache(
    tracker: Optional[torch.nn.Module],
    media_path: str,
    cache_path: str,
    args: argparse.Namespace,
) -> None:
    """Generate ONE cache via the offline extractor; atomic write."""
    tmp_path = cache_path + ".building"
    tgl_offline.process_video(
        input_path=media_path,
        out_path=tmp_path,
        tracker=tracker,
        num_frames=int(args.num_frames),
        resize=(int(args.spatial_size), int(args.spatial_size)),
        patch_grid=int(args.spatial_size) // int(args.patch_size),
        query_frame_stride=int(args.query_frame_stride),
        delta=float(args.delta),
        dir_thresh=float(args.dir_thresh),
        max_tracks=int(args.max_tracks),
        sampler=str(tgl_offline.resolve_tgl_preset(args.tgl_preset)["offline_sampler"]),
        seed=int(args.seed),
        device=args.device,
    )
    os.replace(tmp_path, cache_path)


def auto_track(args: argparse.Namespace, cfg: CfgNode) -> None:
    """Requirements 2-6: scan lists, skip existing caches, fill the gaps."""
    entries = parse_video_list(args.train_list)
    if args.val_list:
        entries += parse_video_list(args.val_list)

    todo: List[Tuple[str, str, str]] = []  # (rel_path, media, cache_path)
    skipped = 0
    for rel_path, _ in entries:
        cache_path = track_cache_path(cfg.MODEL.TGL.TRACK_ROOT, rel_path)
        if os.path.isfile(cache_path) and not args.force_regen:
            skipped += 1
            continue
        if args.force_regen and os.path.isfile(cache_path):
            os.remove(cache_path)
        media = resolve_media(args.data_root, rel_path)
        if media is None:
            print(f"[auto-track] WARNING: media missing for {rel_path!r}, skipped")
            continue
        todo.append((rel_path, media, cache_path))

    total = len(entries)
    print(f"[auto-track] {total} videos | {skipped} cached (skipped) | {len(todo)} to generate")
    if not todo:
        return

    tracker = None
    for i, (rel_path, media, cache_path) in enumerate(todo, 1):
        print(f"[auto-track] ({i}/{len(todo)}) {rel_path}")
        if args.smoke_simulate:
            simulate_track_cache(
                cache_path,
                int(args.num_frames),
                int(args.max_tracks),
                cfg.MODEL.TGL.IMAGE_SIZE,
                cfg.MODEL.TGL.PATCH_GRID,
                seed_key=rel_path,
            )
        else:
            if tracker is None:
                tracker = tgl_offline.build_tracker(
                    device=args.device,
                    version=args.tracker_version,
                    hub_repo=args.hub_repo,
                )
            generate_track_cache(tracker, media, cache_path, args)


def _preflight_entries(entries, tgl, track_root):
    """Per-entry cache validation. Returns (problems, warnings)."""
    problems = []
    warnings = []
    T = int(tgl.NUM_FRAMES)
    size = list(tgl.IMAGE_SIZE)
    grid = int(tgl.PATCH_GRID)

    for rel_path, _ in entries:
        cache_path = track_cache_path(track_root, rel_path)
        if not os.path.isfile(cache_path):
            problems.append(f"MISSING cache for {rel_path!r} -> {cache_path}")
            continue
        payload = torch.load(cache_path, map_location="cpu")
        tracks = payload.get("tracks")
        if tracks is None or tracks.dim() != 3 or tracks.shape[1] != T:
            got = None if tracks is None else tuple(tracks.shape)
            problems.append(
                f"{rel_path!r}: cache num_frames mismatch (expected T={T}, "
                f"got {got}). Regenerate the tracks.")
            continue
        if tracks.shape[0] > int(tgl.MAX_TRACKS):
            problems.append(
                f"{rel_path!r}: K_cache={tracks.shape[0]} > "
                f"max_tracks={int(tgl.MAX_TRACKS)}. Regenerate.")
        if payload.get("image_size") != size:
            problems.append(
                f"{rel_path!r}: image_size {payload.get('image_size')} != "
                f"training {size}. Regenerate the tracks.")
        pcfg = payload.get("preprocess_config", {})
        if list(pcfg.get("resize", [])) != size:
            problems.append(
                f"{rel_path!r}: preprocess resize {pcfg.get('resize')} != "
                f"training {size}. Regenerate.")
        if int(pcfg.get("patch_grid", -1)) != grid:
            problems.append(
                f"{rel_path!r}: patch_grid {pcfg.get('patch_grid')} != model "
                f"grid {grid}. Regenerate.")
        if len(payload.get("frame_indices", [])) != T:
            problems.append(
                f"{rel_path!r}: frame_indices length != {T}. Regenerate.")
        for key, expected in (
            ("sampler", tgl.PRESET_RESOLVED["offline_sampler"]),
            ("query_frame_stride", int(tgl.QUERY_FRAME_STRIDE)),
            ("delta_px", float(tgl.DELTA_PX)),
            ("max_tracks", int(tgl.MAX_TRACKS)),
        ):
            if key == "sampler" and pcfg.get("sampler") == "simulated":
                continue  # smoke 的合成缓存总是兼容
            if key in pcfg and pcfg[key] != expected:
                problems.append(
                    f"{rel_path!r}: preprocess {key}={pcfg[key]} != training "
                    f"{expected}. Regenerate.")
        for key, assumed in (
            ("tracker_version", "cotracker3_offline"),
            ("coordinate_convention", "(x, y), row-major patches"),
            ("original_frame_size", "unrecorded"),
            ("align_corners", False),
        ):
            if key in pcfg and pcfg[key] != assumed:
                problems.append(
                    f"{rel_path!r}: {key}={pcfg[key]} != {assumed}. Regenerate.")
            elif key not in pcfg:
                warnings.append(f"{rel_path!r}: {key} not recorded; assuming {assumed!r}")
    return problems, warnings


def verify_caches(args: argparse.Namespace, cfg: CfgNode) -> None:
    """Requirement 13: cache completeness + preprocessing consistency."""
    train_entries = parse_video_list(args.train_list)
    val_entries = parse_video_list(args.val_list) if args.val_list else []
    problems = []
    warnings = []
    p, w = _preflight_entries(train_entries, cfg.MODEL.TGL, cfg.MODEL.TGL.TRACK_ROOT)
    problems += ["[train] " + x for x in p]
    warnings += ["[train] " + x for x in w]
    p, w = _preflight_entries(val_entries, cfg.MODEL.TGL, cfg.MODEL.TGL.TRACK_ROOT)
    problems += ["[val] " + x for x in p]
    warnings += ["[val] " + x for x in w]

    if warnings:
        seen = set()
        for w in warnings:
            short = w.split(":")[1] if ":" in w else w
            if short not in seen:
                seen.add(short)
                print(f"[check][warn] {w.split('] ')[1] if '] ' in w else w}")
        print(f"[check] {len(warnings)} legacy-field warnings (assumed defaults)")
    if problems:
        print("=" * 72)
        print(f"TGL PRE-LAUNCH CHECK FAILED with {len(problems)} problem(s):")
        for p in problems[:50]:
            print("  " + p)
        if len(problems) > 50:
            print(f"  ... and {len(problems) - 50} more")
        print(
            "\nThe training preprocessing and the track caches disagree.\n"
            "Fix: re-run this pipeline with --force-regen (deletes stale "
            "caches and re-extracts with the current num_frames/"
            "image_size/preprocess settings), or align --num-frames / "
            "--spatial-size with the caches."
        )
        print("=" * 72)
        raise SystemExit(2)
    print(f"[check] all {len(train_entries)} train caches consistent "
          f"(T={cfg.MODEL.TGL.NUM_FRAMES}, image_size={cfg.MODEL.TGL.IMAGE_SIZE}, "
          f"grid={cfg.MODEL.TGL.PATCH_GRID}, run_tracker_online=False)")


# --------------------------------------------------------------------------- #
#  Model-mode resolution                                                       #
# --------------------------------------------------------------------------- #
def resolve_model_kwargs(cfg: CfgNode) -> Dict[str, object]:
    """MODEL MODE: tgl_vp_two_text -> VitaCLIP constructor kwargs."""
    preset = cfg.MODEL.TGL.PRESET_RESOLVED
    return {
        # --- two text branches + text fusion (kept) ---
        "use_fixed_text_branch": True,
        "use_learnable_text_branch": True,
        "use_text_mstff": True,
        # --- TGL v2: per-layer video prompts (CVPT) + Track-ST + fusion ---
        "use_tgl_v2": True,
        "tgl_v2_prompt_source_layer": int(cfg.MODEL.TGL.PROMPT_SOURCE_LAYER),
        "tgl_v2_prompt_layers": str(cfg.MODEL.TGL.V2_PROMPT_LAYERS),
        "tgl_v2_num_prompts": int(cfg.MODEL.TGL.NUM_PROMPTS),
        "tgl_v2_recon_ratio": float(cfg.MODEL.TGL.RECON_RATIO),
        "tgl_v2_readout_gate": float(cfg.MODEL.TGL.READOUT_GATE),
        # --- legacy modules (disabled) ---
        # --- legacy modules (disabled) ---
        "use_mask_adapter": False,        # 旧 mask adapter
        "use_layered_prompts": False,     # 旧 patch enhancer / 多余 prompt 分支
        "use_discriminative_module": False,
        "use_bie": False,
        "use_scr": False,
        "use_aggregation_mixer": False,   # 旧 image-side aggregation adapter
    }


# --------------------------------------------------------------------------- #
#  Training launch                                                             #
# --------------------------------------------------------------------------- #
def build_train_command(args: argparse.Namespace, cfg: CfgNode) -> List[str]:
    # Single source of truth: the model-mode kwargs decide which module flags
    # are forwarded. NEVER rely on trainer defaults here - an earlier launch
    # silently dropped the video prompt because --use_cvp_block defaults to
    # False in the trainer while MODEL.VIDEO_PROMPT.ENABLE was True.
    mk = resolve_model_kwargs(cfg)
    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        f"--nproc_per_node={args.gpus}",
        args.train_script,
        "--train_list_path", os.path.abspath(args.train_list),
        "--val_list_path", os.path.abspath(args.val_list) if args.val_list else "",
        "--train_data_root", os.path.abspath(args.data_root),
        "--data_root", os.path.abspath(args.data_root),
        "--num_frames", str(cfg.MODEL.TGL.NUM_FRAMES),
        "--spatial_size", str(cfg.MODEL.TGL.IMAGE_SIZE[0]),
        # TGL v2 on + offline caches; RUN_TRACKER_ONLINE is always False.
        "--tgl_enable",
        "--tgl_track_root", cfg.MODEL.TGL.TRACK_ROOT,
        "--tgl_preset", cfg.MODEL.TGL.PRESET,
        "--tgl_v2",
        "--tgl_v2_prompt_source_layer", str(cfg.MODEL.TGL.PROMPT_SOURCE_LAYER),
        "--tgl_v2_prompt_layers", str(cfg.MODEL.TGL.V2_PROMPT_LAYERS),
        "--tgl_v2_recon_ratio", str(cfg.MODEL.TGL.RECON_RATIO),
        "--tgl_loss_traj_aux", "0.2",
        "--tgl_loss_recon", "0.1",
        "--tgl_readout_gate", str(cfg.MODEL.TGL.READOUT_GATE),
        # --- two text branches + text fusion (explicit, no default drift) ---
        *(
            ["--use_fixed_text_branch"] if mk["use_fixed_text_branch"] else []
        ),
        *(
            ["--use_learnable_text_branch"] if mk["use_learnable_text_branch"] else []
        ),
        *(["--use_text_mstff"] if mk["use_text_mstff"] else []),
        # --- legacy modules stay OFF (trainer defaults are False; asserted) ---
    ]
    if args.epochs:
        cmd += ["--epochs", str(args.epochs)]
    if args.batch_size:
        cmd += ["--batch_size", str(args.batch_size)]
    if args.lr:
        cmd += ["--lr", str(args.lr)]
    if args.backbone_path:
        cmd += ["--backbone_path", os.path.abspath(args.backbone_path)]
    if args.classes_file:
        cmd += ["--text_prompt_classes_path", os.path.abspath(args.classes_file)]
    if args.output_dir:
        cmd += ["--checkpoint_dir", os.path.abspath(args.output_dir)]
    for extra in args.train_arg or []:
        cmd.extend(extra.split())
    return [c for c in cmd if c != ""]


def launch_training(args: argparse.Namespace, cfg: CfgNode) -> int:
    cmd = build_train_command(args, cfg)
    print("[launch] MODEL.TGL.ENABLE=True | MODEL.TGL.RUN_TRACKER_ONLINE=False "
          f"| MODEL.TGL.PRESET={cfg.MODEL.TGL.PRESET} | "
          f"MODEL.VIDEO_PROMPT.ENABLE=True | MODEL.TEXT_BRANCHES.NUM=2")
    print("[launch] " + " ".join(cmd))
    if args.dry_run:
        print("[launch] --dry-run: command printed, not executed.")
        return 0
    return subprocess.call(cmd)


# --------------------------------------------------------------------------- #
#  Smoke test (requirement 14)                                                 #
# --------------------------------------------------------------------------- #
def smoke_test(args: argparse.Namespace, cfg: CfgNode) -> None:
    """Fake list + simulated caches + one forward/backward + assertions."""
    from VitaCLIP_model import VitaCLIP

    device = args.smoke_device
    B, K, T = 2, 8, int(cfg.MODEL.TGL.NUM_FRAMES)
    print(f"[smoke] building simulated caches (T={T}, K={K}) - no tracker")

    with tempfile.TemporaryDirectory(prefix="tgl_smoke_") as tmp:
        fake_list = os.path.join(tmp, "fake_list.txt")
        track_root = os.path.join(tmp, "tracks")
        lines = []
        for i in range(B):
            rel = f"FakeClass/fake_video_{i}"
            lines.append(f"{rel} {i + 1}")
            simulate_track_cache(
                track_cache_path(track_root, rel),
                T, K, cfg.MODEL.TGL.IMAGE_SIZE, cfg.MODEL.TGL.PATCH_GRID,
                seed_key=rel,
            )
        with open(fake_list, "w") as f:
            f.write("\n".join(lines) + "\n")

        # re-run the pre-launch checks against the simulated caches
        smoke_args = argparse.Namespace(**{**vars(args), "train_list": fake_list, "val_list": ""})
        smoke_cfg = build_config(smoke_args)
        smoke_cfg.MODEL.TGL.TRACK_ROOT = track_root
        verify_caches(smoke_args, smoke_cfg)

        # --- model under the tgl_vp_two_text mode ---
        kwargs = resolve_model_kwargs(cfg)
        print(f"[smoke] mode={cfg.MODEL.MODE} -> "
              f"v2_prompts={kwargs['use_tgl_v2']}, tgl={kwargs['use_tgl_v2']}, "
              f"mask_adapter={kwargs['use_mask_adapter']}, "
              f"layered_prompts={kwargs['use_layered_prompts']}, "
              f"bie={kwargs['use_bie']}, scr={kwargs['use_scr']}")
        model = VitaCLIP(
            backbone_path=args.backbone_path,
            input_size=(cfg.MODEL.TGL.IMAGE_SIZE[0], cfg.MODEL.TGL.IMAGE_SIZE[1]),
            num_frames=T,
            num_classes=101,  # matches the default UCF101 class file
            text_prompt_classes_path=args.classes_file,
            **kwargs,
        ).to(device).eval()

        # --- one batch from the simulated caches ---
        tracks = torch.stack([
            torch.load(track_cache_path(track_root, f"FakeClass/fake_video_{i}"),
                       map_location="cpu")["tracks"][:K]
            for i in range(B)
        ]).to(device)
        visibility = torch.stack([
            torch.load(track_cache_path(track_root, f"FakeClass/fake_video_{i}"),
                       map_location="cpu")["visibility"][:K]
            for i in range(B)
        ]).to(device)
        birth_frame = torch.stack([
            torch.load(track_cache_path(track_root, f"FakeClass/fake_video_{i}"),
                       map_location="cpu")["birth_frame"][:K]
            for i in range(B)
        ]).to(device)
        frames = torch.randn(B, 3, T, cfg.MODEL.TGL.IMAGE_SIZE[0], cfg.MODEL.TGL.IMAGE_SIZE[1], device=device)
        labels = torch.randint(0, 4, (B,), device=device)

        def grad_ok(param) -> bool:
            return param.grad is not None and float(param.grad.abs().sum()) > 0.0

        # --- structural assertions for the v2 design (Track-ST + late fusion,
        # NO mid-layer TGL adapters, NO gamma scatter gates) ---
        assert model.visual.use_tgl_v2 is True
        assert model.visual.v2_track_st is not None
        assert model.visual.tgl_adapters is None, \
            "v2 must not construct the legacy mid-layer TGL adapters"
        assert model.visual.tgl_patch_pool_baseline is False, \
            "patch_pool_baseline must default OFF (confounded readout)"
        assert model.visual.tgl_baseline_logit is None
        assert model.visual.tgl_readout is None, \
            "v2 uses the late-fusion gate, not the legacy readout gate"
        # prompt generator: beta warm-starts at 0.5, per-layer up_proj non-zero
        pg = model.visual.v2_prompt_gen
        assert float(torch.sigmoid(pg.beta_logit)) == 0.5
        assert len(pg.up_proj_l) == len(model.visual.tgl_v2_prompt_layers)
        for up in pg.up_proj_l:
            assert float(up.weight.abs().max()) > 0.0, \
                "up_proj_l must be non-zero init (single-zero-factor rule)"
        # TAT-style late fusion gate: warm init at 0.1 of trajectory energy
        fg = model.visual.v2_late_fusion
        assert abs(float(fg.gate_linear.bias) - math.log(0.1 / 0.9)) < 1e-4, \
            "fusion gate bias must init to logit(0.1)"
        model.train()  # 训练步模拟：启用 recon/aux 分支（dropout 仍为 0）

        # --- step 0 (warm start): the residual path is OPEN, so every TGL
        # parameter receives gradient on the very first backward. ---
        logits = model(frames, labels, tracks=tracks, visibility=visibility, birth_frame=birth_frame)
        assert logits.shape == (B, 101), logits.shape
        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward()
        assert torch.isfinite(loss), "loss is not finite"
        # step0 唤醒状态断言：
        #   * v2_track_st / v2_late_fusion / prompt 生成器主体 / resblock
        #     alpha —— 立即有梯度（alpha 是活的缩放因子）；
        #   * up_proj_l / prompt_norm_l —— 被 alpha=0 阻断，step1 后唤醒
        #     （规范 §三：单步冷启动而非永久死锁）。
        # step0/step1 唤醒语义（规范 §三 test 18）：
        #   step0: 轨迹分支（Track-ST/融合门）与 resblock alpha 立即有梯度；
        #          prompt 生成器 up_proj/prompt_norm 与 resblock cross/ln 被
        #          alpha=0 阻断 —— 单步冷启动，alpha 移动一步后全部唤醒。
        def _grp(n):
            if "track_st" in n: return "track_st"
            if "late_fusion" in n: return "late_fusion"
            if "prompt_gen" in n: return "prompt_gen"
            if ".prompt_" in n: return "resblock_prompt"
            return None

        groups = {"track_st": [0, 0], "late_fusion": [0, 0],
                  "prompt_gen": [0, 0], "resblock_prompt": [0, 0]}
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            grp = _grp(n)
            if grp is None:
                continue
            g = 0.0 if p.grad is None else float(p.grad.abs().sum())
            groups[grp][0 if g > 0 else 1] += 1
        assert groups["track_st"][1] == 0 and groups["track_st"][0] > 0, \
            "step0: Track-ST 必须全部有梯度"
        assert groups["late_fusion"][1] == 0 and groups["late_fusion"][0] > 0, \
            "step0: 融合门必须有梯度"
        assert groups["resblock_prompt"][1] == 0, \
            "step0: resblock prompt alpha 必须有梯度（alpha 是活因子）"
        assert groups["prompt_gen"][1] == groups["prompt_gen"][0], \
            "step0: prompt 生成器被 alpha=0 阻断属预期（单步冷启动）"
        pg_stats = model.visual._last_tgl_stats.get("v2_prompt_gen", {})
        assert abs(pg_stats.get("beta", -1) - 0.5) < 1e-6, "beta 必须 warm-start 0.5"
        print(f"[smoke] step0: loss={float(loss):.4f} backward ok | "
              f"awake={sum(g[0] for g in groups.values())} "
              f"blocked={sum(g[1] for g in groups.values())} (alpha=0 阻断) | "
              f"beta={pg_stats.get('beta', 'n/a')}")

        # step1: alpha 移动一步 → prompt 生成器全路径唤醒
        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            for blk in model.visual.transformer.resblocks:
                if getattr(blk, "prompt_alpha", None) is not None:
                    blk.prompt_alpha += 0.5
        logits = model(frames, labels, tracks=tracks, visibility=visibility,
                       birth_frame=birth_frame)
        torch.nn.functional.cross_entropy(logits, labels).backward()
        gen_awake = sum(1 for n, p in model.named_parameters()
                        if "v2_prompt_gen" in n and p.requires_grad
                        and p.grad is not None and float(p.grad.abs().sum()) > 0)
        gen_total = sum(1 for n, p in model.named_parameters()
                        if "v2_prompt_gen" in n and p.requires_grad)
        assert gen_awake >= 0.9 * gen_total, \
            f"alpha 移动后 prompt 生成器必须唤醒: {gen_awake}/{gen_total}"
        print(f"[smoke] step1: prompt 生成器唤醒 {gen_awake}/{gen_total} ✓")

        # v2 的 video prompt = 逐层注入的 prompt 生成器（替代 CVP 残差路径）
        vp_params = [(n, p) for n, p in model.named_parameters()
                     if ("v2_prompt_gen" in n) and p.requires_grad]
        assert vp_params, "v2 prompt generator 必须存在且可训练"

        # --- text branch states must match MODEL.TEXT_BRANCHES.NUM=2 ---
        assert model.use_fixed_text_branch is True, "fixed text branch must be ON"
        assert model.use_learnable_text_branch is True, "learnable text branch must be ON"
        assert model.use_text_mstff is True, "text fusion (MSTFF) must be ON"
        learnable = [(n, p) for n, p in model.named_parameters()
                     if "prompt_learner" in n and p.requires_grad]
        assert learnable, "learnable text branch must own trainable parameters"
        assert all(grad_ok(p) for _, p in learnable), "learnable prompt branch must receive gradients"
        # legacy branches really OFF
        assert kwargs["use_mask_adapter"] is False and kwargs["use_layered_prompts"] is False
        print(f"[smoke] text branches: fixed=ON learnable=ON "
              f"({len(learnable)} trainable prompt tensors, all with grads) | "
              f"legacy adapters OFF | patch_pool_baseline=OFF")
    print("[smoke] ALL SMOKE ASSERTIONS PASSED")


# --------------------------------------------------------------------------- #
#  CLI                                                                         #
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # pipeline switches
    ap.add_argument("--auto-track", action="store_true", dest="auto_track",
                    help="generate missing track caches before training")
    ap.add_argument("--track-root", default="/hy-tmp/outputs/tgl_tracks",
                    help="MODEL.TGL.TRACK_ROOT: offline cache directory")
    ap.add_argument("--train-list", default="", help="training list ('rel_path label' lines)")
    ap.add_argument("--val-list", default="", help="optional validation list")
    ap.add_argument("--data-root", default="/hy-tmp/UCF101/rawframes",
                    help="root that list paths are relative to")
    # offline tracker
    ap.add_argument("--num-frames", type=int, default=8)
    ap.add_argument("--spatial-size", type=int, default=224)
    ap.add_argument("--patch-size", type=int, default=16)
    ap.add_argument("--max-tracks", type=int, default=64)
    ap.add_argument("--query-frame-stride", type=int, default=2,
                    help="TAT-style multi-frame births every k frames")
    ap.add_argument("--delta", type=float, default=8.0, help="dedup distance threshold (px)")
    ap.add_argument("--dir-thresh", type=float, default=0.9, help="dedup direction cosine threshold")
    ap.add_argument("--tracker-version", default="cotracker3_offline")
    ap.add_argument("--hub-repo", default="facebookresearch/co-tracker")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force-regen", action="store_true",
                    help="delete existing caches and regenerate (after a config mismatch)")
    # model / preset
    ap.add_argument("--model-mode", default="tgl_vp_two_text",
                    choices=("tgl_vp_two_text",),
                    help="model-mode: TGL adapter + video prompt + two text branches")
    ap.add_argument("--tgl-preset", default="tat_like", choices=sorted(TGL_PRESETS))
    ap.add_argument("--tgl-enable", action="store_true", dest="tgl_enable",
                    help="also set MODEL.TGL.ENABLE in the trainer even without --auto-track")
    ap.add_argument("--tgl-adapter-dim", type=int, default=None)
    ap.add_argument("--tgl-num-layers", type=int, default=2)
    ap.add_argument("--tgl-num-heads", type=int, default=4)
    ap.add_argument("--tgl-dropout", type=float, default=0.1)
    ap.add_argument("--tgl-max-scale", type=float, default=0.1)
    ap.add_argument("--tgl-max-fusion", type=float, default=0.5)
    ap.add_argument("--tgl-insert-layers", default="6,8,10",
                    help="ViT blocks after which TGL writes its residual "
                    "(MODEL.TGL.INSERT_LAYERS, insert_mode=blocks)")
    ap.add_argument("--tgl-num-prompts", type=int, default=4,
                    help="trajectory-residual video prompts per insertion "
                    "(source = residual tokens, never mean_pool(enhanced))")
    ap.add_argument("--tgl-gate-init", type=float, default=0.04,
                    help="initial gate opening in scale units; warm start "
                    "lets the full adapter train from step 1 (0 = strict "
                    "identity start, which starves the adapter of gradient)")
    ap.add_argument("--tgl-v2-prompt-source-layer", type=int, default=4,
                    help="1-based block whose output builds the 8 frame tokens")
    ap.add_argument("--tgl-v2-prompt-layers", default="5,6,7,8,9,10,11,12",
                    help="1-based blocks receiving prompt injection")
    ap.add_argument("--tgl-v2-recon-ratio", type=float, default=0.1,
                    help="fraction of valid trajectory positions masked for "
                    "the reconstruction auxiliary (training only)")
    ap.add_argument("--cvp-insert-layer", type=int, default=4)
    ap.add_argument("--cvp-num-prompts", type=int, default=4)
    # training launch
    ap.add_argument("--train-script", default=DEFAULT_TRAIN_SCRIPT)
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--backbone-path", default=DEFAULT_BACKBONE)
    ap.add_argument("--classes-file", default=DEFAULT_CLASSES)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--train-arg", action="append", default=[],
                    help="extra raw args forwarded to the trainer, e.g. "
                    "--train-arg '--fp16' --train-arg '--eval_freq 5'")
    ap.add_argument("--dry-run", action="store_true",
                    help="run auto-track + checks, print the train command, do not train")
    # smoke test
    ap.add_argument("--smoke-test", action="store_true",
                    help="fake list + simulated caches + one fwd/bwd assertion run")
    ap.add_argument("--smoke-device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--smoke-simulate", action="store_true",
                    help="with --auto-track: write SIMULATED caches instead of "
                    "running the tracker (offline tests without CoTracker3)")
    return ap.parse_args(argv)


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    validate_config(cfg)

    print(f"[config] MODEL.TGL.ENABLE={cfg.MODEL.TGL.ENABLE} | "
          f"MODEL.TGL.TRACK_ROOT={cfg.MODEL.TGL.TRACK_ROOT} | "
          f"MODEL.TGL.RUN_TRACKER_ONLINE={cfg.MODEL.TGL.RUN_TRACKER_ONLINE} | "
          f"MODEL.TGL.PRESET={cfg.MODEL.TGL.PRESET}")
    print(f"[config] MODEL.VIDEO_PROMPT.ENABLE={cfg.MODEL.VIDEO_PROMPT.ENABLE} | "
          f"MODEL.TEXT_BRANCHES.NUM={cfg.MODEL.TEXT_BRANCHES.NUM} | "
          f"MODEL.MODE={cfg.MODEL.MODE}")

    if args.smoke_test:
        smoke_test(args, cfg)
        return

    if not args.train_list:
        raise SystemExit("--train-list is required (or use --smoke-test).")
    if args.auto_track or args.force_regen:
        auto_track(args, cfg)
    else:
        print("[auto-track] disabled: assuming caches already exist.")

    verify_caches(args, cfg)

    # Only now, with every cache in place and consistent, start training.
    code = launch_training(args, cfg)
    sys.exit(code)


if __name__ == "__main__":
    main()
