#!/usr/bin/env python
"""DataLoader configuration for supervised UCF101 raw-frame training."""

from __future__ import annotations

import argparse
import math
import random
from typing import Dict

import numpy as np
import torch
import torch.distributed as dist

from .dataset import DummyDataset, VideoDataset


UCF101_ROOT = "/hy-tmp/UCF101"
UCF101_ANNOTATION_ROOT = (
    f"{UCF101_ROOT}/annotations/ucfTrainTestlist"
)
CLIP_MEAN = (
    0.48145466,
    0.45782750,
    0.40821073,
)
CLIP_STD = (
    0.26862954,
    0.26130258,
    0.27577711,
)


def setup_arg_parser(
    parser: argparse.ArgumentParser,
) -> None:
    # UCF101 defaults. They can still be overridden for another dataset.
    parser.add_argument(
        "--train_list_path",
        type=str,
        default=(
            f"{UCF101_ANNOTATION_ROOT}/trainlist01.txt"
        ),
        help="training list path",
    )
    parser.add_argument(
        "--val_list_path",
        type=str,
        default=(
            f"{UCF101_ANNOTATION_ROOT}/testlist01.txt"
        ),
        help="validation list path",
    )
    parser.add_argument(
        "--train_data_root",
        type=str,
        default="",
        help="optional training root overriding --data_root",
    )
    parser.add_argument(
        "--val_data_root",
        type=str,
        default="",
        help="optional validation root overriding --data_root",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=f"{UCF101_ROOT}/rawframes",
        help="video or raw-frame root",
    )
    parser.add_argument(
        "--class_index_path",
        type=str,
        default=(
            f"{UCF101_ANNOTATION_ROOT}/classInd.txt"
        ),
        help=(
            "official class index, required when validation list "
            "contains paths without numeric labels"
        ),
    )
    parser.add_argument(
        "--label_offset",
        type=int,
        default=1,
        help=(
            "subtract this value from list/classInd labels; "
            "official UCF101 labels use 1"
        ),
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=101,
    )

    # Original Vita-CLIP interprets batch_size as the total batch over GPUs.
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="global training batch size across all GPUs",
    )

    parser.add_argument(
        "--num_spatial_views",
        type=int,
        default=1,
        choices=(1, 3),
        help="validation spatial crops",
    )
    parser.add_argument(
        "--num_temporal_views",
        type=int,
        default=1,
        help="validation temporal views",
    )
    parser.add_argument(
        "--val_batch_size",
        type=int,
        default=1,
        help=(
            "number of validation videos per batch. Each video already "
            "carries all its views, so a larger value increases GPU "
            "throughput during evaluation."
        ),
    )
    parser.add_argument(
        "--eval_frame_shuffle",
        action="store_true",
        dest="eval_frame_shuffle",
        help="temporal-sensitive evaluation: randomly permute the T frames "
        "of every view before inference. A model that only exploits single-"
        "frame semantics keeps high accuracy after shuffling; a drop in "
        "accuracy shows genuine use of temporal structure.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=8,
        help="frames per video view",
    )
    parser.add_argument(
        "--sampling_rate",
        type=int,
        default=16,
        help="fixed temporal stride when TSN sampling is disabled",
    )
    parser.add_argument(
        "--tsn_sampling",
        action="store_true",
        dest="tsn_sampling",
        help="sample frames across the complete video",
    )
    parser.add_argument(
        "--no_tsn_sampling",
        action="store_false",
        dest="tsn_sampling",
        help="use fixed-stride temporal sampling",
    )
    parser.set_defaults(
        tsn_sampling=True
    )

    parser.add_argument(
        "--spatial_size",
        type=int,
        default=224,
    )

    parser.add_argument(
        "--mean",
        type=float,
        nargs="+",
        default=list(CLIP_MEAN),
        help="RGB normalization mean",
    )
    parser.add_argument(
        "--std",
        type=float,
        nargs="+",
        default=list(CLIP_STD),
        help="RGB normalization std",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--no_pin_memory",
        action="store_false",
        dest="pin_memory",
    )
    parser.set_defaults(
        pin_memory=True
    )

    parser.add_argument(
        "--dummy_dataset",
        action="store_true",
        help="use zero tensors for speed testing only",
    )
    parser.add_argument(
        "--auto_augment",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--interpolation",
        type=str,
        default="bicubic",
    )
    parser.add_argument(
        "--no_mirror",
        action="store_false",
        dest="mirror",
    )
    parser.set_defaults(
        mirror=True
    )
    parser.add_argument(
        "--temporal_flip",
        action="store_true",
        help="randomly reverse the frame order of a training clip with "
        "probability 0.5 (temporal regulariser for action recognition)",
    )
    parser.add_argument(
        "--train_rrc_scale",
        type=str,
        default="0.08,1.0",
        help="lower,upper area-scale range of the training RandomResizedCrop "
        "as a comma-separated pair, e.g. '0.5,1.0'. (0.08,1.0) is the "
        "ImageNet-style default but is overly aggressive for action "
        "recognition: it frequently crops out the action subject. "
        "(default: 0.08,1.0)",
    )
    parser.add_argument(
        "--tgl_track_root",
        type=str,
        default="",
        help="root of offline TGL track caches (.pt per video). When set, "
        "the dataset loads precomputed CoTracker3 trajectories and switches "
        "to the deterministic TGL geometric pipeline. The tracker is NEVER "
        "run here.",
    )
    parser.add_argument(
        "--tgl_max_tracks",
        type=int,
        default=64,
        help="fixed trajectory count per sample (truncate/pad)",
    )


def _parse_train_rrc_scale(
    args: argparse.Namespace,
) -> tuple:
    raw = getattr(args, "train_rrc_scale", "0.08,1.0")
    parts = [float(x.strip()) for x in str(raw).split(",")]
    if len(parts) != 2 or not (0.0 < parts[0] <= parts[1] <= 1.0):
        raise ValueError(
            f"--train_rrc_scale must be 'low,high' with 0<low<=high<=1, got {raw!r}"
        )
    return (parts[0], parts[1])


def _parse_mean_and_std(
    args: argparse.Namespace,
) -> Dict[str, torch.Tensor]:

    def parse(
        values,
        default_values,
    ) -> torch.Tensor:
        if values is None:
            return torch.tensor(
                default_values,
                dtype=torch.float32,
            )
        if len(values) == 1:
            return torch.tensor(
                values * 3,
                dtype=torch.float32,
            )
        if len(values) == 3:
            return torch.tensor(
                values,
                dtype=torch.float32,
            )
        raise ValueError(
            "mean/std must contain one or three values."
        )

    return {
        "mean": parse(
            args.mean,
            CLIP_MEAN,
        ),
        "std": parse(
            args.std,
            CLIP_STD,
        ),
    }


def _dataset_kwargs(
    args: argparse.Namespace,
) -> Dict[str, object]:
    kwargs: Dict[str, object] = {
        "class_index_path": args.class_index_path,
        "label_offset": args.label_offset,
        "num_classes": args.num_classes,
    }
    track_root = getattr(args, "tgl_track_root", "") or None
    if track_root is not None:
        kwargs["track_root"] = track_root
        kwargs["tgl_max_tracks"] = int(
            getattr(args, "tgl_max_tracks", 64)
        )
    return kwargs


def create_train_dataset(
    args: argparse.Namespace,
) -> torch.utils.data.Dataset:
    if args.dummy_dataset:
        return DummyDataset(
            list_path=args.train_list_path,
            num_frames=args.num_frames,
            num_views=1,
            spatial_size=args.spatial_size,
        )

    return VideoDataset(
        list_path=args.train_list_path,
        data_root=(
            args.train_data_root
            or args.data_root
        ),
        num_spatial_views=1,
        num_temporal_views=1,
        random_sample=True,
        auto_augment=args.auto_augment,
        interpolation=args.interpolation,
        mirror=args.mirror,
        temporal_flip=args.temporal_flip,
        num_frames=args.num_frames,
        sampling_rate=(
            -1
            if args.tsn_sampling
            else args.sampling_rate
        ),
        spatial_size=args.spatial_size,
        train_scale=_parse_train_rrc_scale(args),
        **_parse_mean_and_std(args),
        **_dataset_kwargs(args),
    )


def _distributed_context():
    if dist.is_initialized():
        return (
            dist.get_rank(),
            dist.get_world_size(),
        )
    return 0, 1


def _seed_worker(
    worker_id: int,
) -> None:
    worker_seed = (
        torch.initial_seed()
        % (2 ** 32)
    )
    np.random.seed(
        worker_seed
    )
    random.seed(
        worker_seed
    )


def create_train_loader(
    args: argparse.Namespace,
    resume_step: int = 0,
) -> torch.utils.data.DataLoader:
    dataset = create_train_dataset(args)
    rank, world_size = _distributed_context()

    if args.batch_size <= 0:
        raise ValueError(
            "batch_size must be positive."
        )
    if args.batch_size % world_size != 0:
        raise ValueError(
            f"Global batch size {args.batch_size} must be divisible by "
            f"world_size={world_size}."
        )
    if args.num_steps is None or args.num_steps <= 0:
        raise ValueError(
            "args.num_steps must be resolved before creating the loader."
        )
    if not 0 <= resume_step <= args.num_steps:
        raise ValueError(
            f"Invalid resume_step={resume_step} for "
            f"num_steps={args.num_steps}."
        )

    batch_size_per_gpu = (
        args.batch_size
        // world_size
    )
    total_global_samples = (
        args.num_steps
        * args.batch_size
    )
    repetitions = math.ceil(
        total_global_samples
        / len(dataset)
    )

    shuffled_cycles = []
    for cycle_index in range(
        repetitions
    ):
        generator = torch.Generator()
        generator.manual_seed(
            cycle_index
        )
        shuffled_cycles.append(
            torch.randperm(
                len(dataset),
                generator=generator,
            )
        )

    global_indices = torch.cat(
        shuffled_cycles,
        dim=0,
    )[:total_global_samples]

    global_indices = global_indices.view(
        args.num_steps,
        args.batch_size,
    )

    rank_start = (
        rank * batch_size_per_gpu
    )
    rank_end = (
        rank_start
        + batch_size_per_gpu
    )
    local_indices = global_indices[
        resume_step:,
        rank_start:rank_end,
    ].reshape(-1).tolist()

    return torch.utils.data.DataLoader(
        dataset,
        sampler=local_indices,
        batch_size=batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=(
            args.num_workers > 0
        ),
        worker_init_fn=_seed_worker,
        drop_last=True,
    )


def create_val_dataset(
    args: argparse.Namespace,
) -> torch.utils.data.Dataset:
    if args.dummy_dataset:
        return DummyDataset(
            list_path=args.val_list_path,
            num_frames=args.num_frames,
            num_views=(
                args.num_spatial_views
                * args.num_temporal_views
            ),
            spatial_size=args.spatial_size,
        )

    return VideoDataset(
        list_path=args.val_list_path,
        data_root=(
            args.val_data_root
            or args.data_root
        ),
        num_spatial_views=args.num_spatial_views,
        num_temporal_views=args.num_temporal_views,
        random_sample=False,
        num_frames=args.num_frames,
        sampling_rate=(
            -1
            if args.tsn_sampling
            else args.sampling_rate
        ),
        spatial_size=args.spatial_size,
        **_parse_mean_and_std(args),
        **_dataset_kwargs(args),
    )


def create_val_loader(
    args: argparse.Namespace,
) -> torch.utils.data.DataLoader:
    dataset = create_val_dataset(args)
    rank, world_size = _distributed_context()

    # Each validation video is evaluated exactly once across all ranks.
    sampler = list(
        range(
            rank,
            len(dataset),
            world_size,
        )
    )

    return torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=getattr(args, "val_batch_size", 1),
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=(
            args.num_workers > 0
        ),
        worker_init_fn=_seed_worker,
        drop_last=False,
    )