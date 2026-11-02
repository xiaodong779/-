#!/usr/bin/env python
"""Fully supervised Vita-CLIP training for UCF101.

Recommended first-run schedule for the current parameter-efficient model:

- 50 epochs
- 8 input frames
- validation every 10 epochs
- checkpoint every 5 epochs
- AdamW
- cosine learning-rate decay
- 3 warm-up epochs
- prompt LR: 5e-4
- adapter/Text_MSTFF LR: 2e-4
- temporal embedding LR: 1e-4
- fusion/logit-scale LR: 1e-4
- weight decay: 5e-4
- label smoothing: 0.1
- gradient clipping: 1.0

The original CLIP visual/text backbones stay frozen. The optimizer receives
only parameters whose ``requires_grad`` flag is True.
"""

from __future__ import annotations

import argparse
import builtins
import math
import os
import random
import sys
from collections import OrderedDict, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist

sys.path.append("./")

import checkpoint

import video_dataset
from tgl_tracker_offline import TGL_PRESETS, resolve_tgl_preset
from VitaCLIP_model import VitaCLIP


# ---------------------------------------------------------------------------
# Distributed logging
# ---------------------------------------------------------------------------

def setup_print(is_master: bool) -> None:
    """Disable normal printing on non-master distributed processes."""
    builtin_print = builtins.print

    def distributed_print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            now = datetime.now().time()
            builtin_print(f"[{now}] ", end="")
            builtin_print(*args, **kwargs)

    builtins.print = distributed_print


def set_random_seed(seed: int, rank: int) -> None:
    """Set independent but reproducible random seeds on every rank."""
    process_seed = int(seed) + int(rank)
    random.seed(process_seed)
    np.random.seed(process_seed)
    torch.manual_seed(process_seed)
    torch.cuda.manual_seed_all(process_seed)


# ---------------------------------------------------------------------------
# Training-length resolution
# ---------------------------------------------------------------------------

def _count_nonempty_lines(path: str) -> int:
    list_path = Path(path)
    if not list_path.is_file():
        raise FileNotFoundError(f"Training list not found: {list_path}")

    with list_path.open("r", encoding="utf-8") as file:
        return sum(1 for line in file if line.strip())


def _resolve_global_batch_size(args: argparse.Namespace) -> Optional[int]:
    """Return the original Vita-CLIP global batch size.

    ``video_dataset/dataloader.py`` interprets ``args.batch_size`` as the
    total batch across all distributed processes and divides it by world size
    when constructing each local DataLoader.
    """
    value = getattr(args, "batch_size", None)
    if value is None:
        return None

    value = int(value)
    if value <= 0:
        raise ValueError(
            f"batch_size must be positive, got {value}."
        )
    return value


def resolve_training_schedule(
    args: argparse.Namespace,
    world_size: int,
) -> Tuple[int, Optional[int], int]:
    """Resolve total steps, steps per epoch and warm-up steps.

    ``num_steps`` can still be supplied explicitly for backward compatibility.
    When it is omitted, the code derives it from:

        ceil(num_train_samples / global_batch_size) * epochs
    """
    train_list_path = getattr(args, "train_list_path", None)
    global_batch_size = _resolve_global_batch_size(args)

    steps_per_epoch: Optional[int] = None
    num_train_samples: Optional[int] = None

    if train_list_path and global_batch_size is not None:
        num_train_samples = _count_nonempty_lines(train_list_path)

        if global_batch_size % world_size != 0:
            raise ValueError(
                f"Global batch size {global_batch_size} must be divisible "
                f"by world_size={world_size}."
            )

        steps_per_epoch = math.ceil(
            num_train_samples / global_batch_size
        )

    if args.num_steps is None:
        if steps_per_epoch is None:
            raise ValueError(
                "Cannot derive num_steps automatically. Provide --num_steps, "
                "or make sure video_dataset exposes --batch_size and "
                "--train_list_path."
            )
        args.num_steps = int(steps_per_epoch * args.epochs)
    else:
        args.num_steps = int(args.num_steps)

    if args.num_steps <= 0:
        raise ValueError(
            f"num_steps must be positive, got {args.num_steps}."
        )

    if args.warmup_steps is not None:
        warmup_steps = int(args.warmup_steps)
    elif steps_per_epoch is not None:
        warmup_steps = int(
            round(args.warmup_epochs * steps_per_epoch)
        )
    else:
        warmup_steps = int(
            round(args.warmup_ratio * args.num_steps)
        )

    warmup_steps = max(0, min(warmup_steps, args.num_steps - 1))

    if args.eval_freq <= 0:
        if steps_per_epoch is not None:
            args.eval_freq = max(
                1,
                int(round(args.eval_every_epochs * steps_per_epoch)),
            )
        else:
            args.eval_freq = max(1, args.num_steps // 30)

    if args.save_freq <= 0:
        if steps_per_epoch is not None:
            args.save_freq = max(
                1,
                int(round(args.save_every_epochs * steps_per_epoch)),
            )
        else:
            args.save_freq = max(1, args.num_steps // 6)

    if num_train_samples is not None:
        print(
            "Training schedule: "
            f"samples={num_train_samples}, "
            f"world_size={world_size}, "
            f"global_batch_size={global_batch_size}, "
            f"batch_per_process={global_batch_size // world_size}, "
            f"steps_per_epoch={steps_per_epoch}, "
            f"epochs={args.epochs}, "
            f"num_steps={args.num_steps}, "
            f"warmup_steps={warmup_steps}, "
            f"eval_freq={args.eval_freq}, "
            f"save_freq={args.save_freq}"
        )
    else:
        print(
            "Training schedule: "
            f"num_steps={args.num_steps}, "
            f"warmup_steps={warmup_steps}, "
            f"eval_freq={args.eval_freq}, "
            f"save_freq={args.save_freq}"
        )

    return args.num_steps, steps_per_epoch, warmup_steps


# ---------------------------------------------------------------------------
# Optimizer and scheduler
# ---------------------------------------------------------------------------

def _parameter_family(name: str) -> str:
    """Map every trainable tensor to one explicit optimizer family.

    Scalar controls are checked before adapter-module names. This is important
    because parameters such as ``bie.gate_t.logit`` and
    ``text_mstff.fusion_logits`` contain an adapter name but should use the
    conservative scalar learning rate.
    """
    lower_name = name.lower()

    # TGL v2 scalar controls (per-layer injection alpha, prompt-gen beta).
    if "prompt_alpha" in lower_name or "beta_logit" in lower_name:
        return "scalar"

    # CVP Block family (raw_gamma/raw_beta and all correspondence modules).
    # The layered v3 modules (mask adapter / per-layer video prompts) share
    # the same conservative family so they can be ablated side by side.
    # TGL v2 recurrent four-slot prompt generator joins this conservative family.
    if (
        "cvp" in lower_name
        or "mask_adapter" in lower_name
        or "layered_prompts" in lower_name
        or "video_prompts" in lower_name
        or "v2_prompt" in lower_name
    ):
        return "cvp"

    scalar_markers = (
        "logit_scale",
        "text_fusion_logits",
        "fusion_logits",
        "text_mstff_gate",
        "residual_scale",
        "patch_mix_logit",
        "raw_c1",
        "raw_c2",
        "bie.gate_t",
        "bie.gate_v",
        "aggregation_mixer.scale",
        "tgl_fusion_logit",
    )
    if any(marker in lower_name for marker in scalar_markers):
        return "scalar"

    # BIE transform MLPs and dynamic-gate MLP: separate, conservative LR
    # family (the gate scalars above already route to "scalar").
    if "bie." in lower_name:
        return "bie"

    if "time_embed" in lower_name:
        return "time_embed"

    if "prompt_learner" in lower_name:
        return "prompt"

    if (
        "discriminative_module" in lower_name
        or "rwkv" in lower_name
        or "text_mstff" in lower_name
        or "aggregation_mixer" in lower_name
        or "tgl_adapter" in lower_name
        or "tgl_readout" in lower_name
        or "v2_track_st" in lower_name
        or "v2_frame_tokens" in lower_name
        or "v2_late_fusion" in lower_name
        or "temporal_pool" in lower_name
    ):
        return "adapter"

    return "base"


def _apply_weight_decay(name: str, parameter: torch.nn.Parameter) -> bool:
    """Apply decay only to adapter matrix weights.

    Prompt embeddings, temporal embeddings, scalar controls, biases and
    normalization parameters are intentionally excluded from weight decay.
    """
    lower_name = name.lower()
    family = _parameter_family(name)

    if family in {"prompt", "time_embed", "scalar"}:
        return False
    if parameter.ndim <= 1:
        return False
    if lower_name.endswith(".bias"):
        return False
    if (
        "norm" in lower_name
        or ".ln_" in lower_name
        or "layernorm" in lower_name
    ):
        return False

    return True


def build_optimizer(
    model: torch.nn.Module,
    args: argparse.Namespace,
) -> torch.optim.AdamW:
    """Build audited AdamW groups from all and only trainable parameters."""
    family_lrs = {
        "base": float(args.lr),
        "prompt": float(args.prompt_lr),
        "adapter": float(args.adapter_lr),
        "time_embed": float(args.time_embed_lr),
        "scalar": float(args.scalar_lr),
        "bie": float(args.bie_lr),
        "cvp": float(getattr(args, "cvp_lr", 2.0e-4)),
    }

    grouped_parameters: Dict[
        Tuple[str, bool],
        List[torch.nn.Parameter],
    ] = defaultdict(list)
    grouped_names: Dict[
        Tuple[str, bool],
        List[str],
    ] = defaultdict(list)

    trainable_by_id: Dict[int, str] = {}
    unexpected_base_names: List[str] = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        parameter_id = id(parameter)
        if parameter_id in trainable_by_id:
            raise RuntimeError(
                "A trainable parameter is registered under multiple names: "
                f"{trainable_by_id[parameter_id]!r} and {name!r}."
            )
        trainable_by_id[parameter_id] = name

        family = _parameter_family(name)
        if family == "base":
            unexpected_base_names.append(name)

        use_decay = _apply_weight_decay(name, parameter)
        key = (family, use_decay)
        grouped_parameters[key].append(parameter)
        grouped_names[key].append(name)

    if not grouped_parameters:
        raise RuntimeError(
            "No trainable parameters were found. Check the model freezing "
            "configuration before constructing the optimizer."
        )

    if unexpected_base_names:
        raise RuntimeError(
            "Uncategorized trainable parameters were found. Refusing to start "
            "training because this may accidentally optimize the frozen CLIP "
            "backbone:\n  "
            + "\n  ".join(unexpected_base_names)
        )

    optimizer_groups = []
    sorted_group_items = sorted(grouped_parameters.items())

    for (family, use_decay), parameters in sorted_group_items:
        optimizer_groups.append(
            {
                "params": parameters,
                "lr": family_lrs[family],
                "weight_decay": (
                    float(args.weight_decay)
                    if use_decay
                    else 0.0
                ),
                "group_name": (
                    f"{family}_{'decay' if use_decay else 'no_decay'}"
                ),
            }
        )

    optimizer = torch.optim.AdamW(
        optimizer_groups,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
    )

    optimizer_parameter_ids: List[int] = [
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    if len(optimizer_parameter_ids) != len(set(optimizer_parameter_ids)):
        raise RuntimeError(
            "The optimizer contains at least one parameter more than once."
        )

    trainable_ids = set(trainable_by_id)
    optimizer_ids = set(optimizer_parameter_ids)
    missing_ids = trainable_ids - optimizer_ids
    extra_ids = optimizer_ids - trainable_ids

    if missing_ids or extra_ids:
        missing_names = [
            trainable_by_id[parameter_id]
            for parameter_id in missing_ids
        ]
        raise RuntimeError(
            "Optimizer coverage audit failed. "
            f"missing={missing_names}, extra_count={len(extra_ids)}."
        )

    trainable_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(f"Total trainable parameters: {trainable_count:,}")
    print(
        "[optimizer audit] all trainable tensors are covered exactly once; "
        "no original CLIP tensor is categorized as trainable."
    )
    print("Optimizer parameter groups:")

    for group, ((family, use_decay), names) in zip(
        optimizer.param_groups,
        sorted(grouped_names.items()),
    ):
        parameter_count = sum(
            parameter.numel()
            for parameter in group["params"]
        )
        print(
            f"  {group['group_name']}: "
            f"lr={group['lr']:.3e}, "
            f"weight_decay={group['weight_decay']:.3e}, "
            f"parameters={parameter_count:,}, "
            f"tensors={len(group['params'])}"
        )
        for name in names:
            print(f"    - {name}")

    return optimizer

def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warm-up followed by cosine decay."""
    if total_steps <= 0:
        raise ValueError("total_steps must be positive.")
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError(
            f"min_lr_ratio must be in [0, 1], got {min_lr_ratio}."
        )

    def lr_multiplier(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)

        cosine_steps = max(1, total_steps - warmup_steps)
        progress = (
            float(step - warmup_steps)
            / float(cosine_steps)
        )
        progress = min(max(progress, 0.0), 1.0)

        cosine_value = 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )
        return (
            min_lr_ratio
            + (1.0 - min_lr_ratio) * cosine_value
        )

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lr_multiplier,
    )


def clamp_logit_scale(
    model: torch.nn.Module,
    max_scale: float = 100.0,
) -> None:
    """Keep CLIP's learned similarity scale in its standard stable range."""
    unwrapped_model = (
        model.module
        if hasattr(model, "module")
        else model
    )
    logit_scale = getattr(unwrapped_model, "logit_scale", None)
    if logit_scale is None:
        return

    with torch.no_grad():
        logit_scale.clamp_(
            min=0.0,
            max=math.log(float(max_scale)),
        )


# ---------------------------------------------------------------------------
# Checkpoint compatibility
# ---------------------------------------------------------------------------

def load_optional_model_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: Optional[str],
) -> None:
    if not checkpoint_path:
        return

    print(f"Loading checkpoint: {checkpoint_path}")
    loaded = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    if isinstance(loaded, dict):
        state_dict = loaded.get(
            "model",
            loaded.get(
                "state_dict",
                loaded.get("model_state_dict", loaded),
            ),
        )
    else:
        state_dict = loaded

    if not isinstance(state_dict, dict):
        raise TypeError(
            "checkpoint_path must contain a model state dictionary."
        )

    renamed_state_dict = OrderedDict()

    for key, value in state_dict.items():
        if not torch.is_tensor(value):
            continue

        clean_key = (
            key[len("module.") :]
            if key.startswith("module.")
            else key
        )
        renamed_state_dict[clean_key] = value

    incompatible = model.load_state_dict(
        renamed_state_dict,
        strict=False,
    )

    if incompatible.missing_keys:
        print(
            f"[checkpoint] missing_keys "
            f"({len(incompatible.missing_keys)}):"
        )
        for key in incompatible.missing_keys:
            print(f"  - {key}")

    if incompatible.unexpected_keys:
        print(
            f"[checkpoint] unexpected_keys "
            f"({len(incompatible.unexpected_keys)}):"
        )
        for key in incompatible.unexpected_keys:
            print(f"  + {key}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()

    video_dataset.setup_arg_parser(parser)
    checkpoint.setup_arg_parser(parser)

    # ------------------------------------------------------------------
    # Training schedule
    # ------------------------------------------------------------------
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help=(
            "number of UCF101 training epochs when --num_steps is omitted"
        ),
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=None,
        help=(
            "explicit number of optimizer steps; overrides automatic "
            "epochs-to-steps conversion"
        ),
    )
    parser.add_argument(
        "--warmup_epochs",
        type=float,
        default=3.0,
        help="linear warm-up duration in epochs",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=None,
        help="explicit warm-up steps; overrides --warmup_epochs",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.10,
        help=(
            "fallback warm-up ratio when steps per epoch cannot be derived"
        ),
    )
    parser.add_argument(
        "--min_lr_ratio",
        type=float,
        default=0.01,
        help="final LR as a fraction of each parameter group's base LR",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="run evaluation only",
    )
    parser.add_argument(
        "--eval_freq",
        type=int,
        default=0,
        help=(
            "evaluate every N steps; 0 selects --eval_every_epochs"
        ),
    )
    parser.add_argument(
        "--save_freq",
        type=int,
        default=0,
        help=(
            "save every N steps; 0 selects --save_every_epochs"
        ),
    )
    parser.add_argument(
        "--eval_every_epochs",
        type=float,
        default=10.0,
        help="automatic validation frequency in epochs",
    )
    parser.add_argument(
        "--eval_start_epoch",
        type=float,
        default=0.0,
        help="only run validation once the training epoch is >= this value. "
        "Saves time in early epochs when the model is far from a useful "
        "checkpoint (default: 0.0 = validate from the start).",
    )
    parser.add_argument(
        "--save_every_epochs",
        type=float,
        default=5.0,
        help="automatic checkpoint frequency in epochs",
    )
    parser.add_argument(
        "--print_freq",
        type=int,
        default=10,
        help="print log message every N optimizer steps",
    )

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="base learning rate for uncategorized trainable parameters",
    )
    parser.add_argument(
        "--prompt_lr",
        type=float,
        default=5e-4,
        help="learning rate for the learnable text prompt",
    )
    parser.add_argument(
        "--adapter_lr",
        type=float,
        default=2e-4,
        help="learning rate for adapter modules and Text_MSTFF",
    )
    parser.add_argument(
        "--time_embed_lr",
        type=float,
        default=1e-4,
        help="learning rate for temporal position embeddings",
    )
    parser.add_argument(
        "--scalar_lr",
        type=float,
        default=1e-4,
        help="learning rate for fusion, temperature and adapter scalars",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=5e-4,
        help="AdamW decay for non-bias, non-normalization matrix weights",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
    )
    parser.add_argument(
        "--adam_eps",
        type=float,
        default=1e-8,
    )
    parser.add_argument(
        "--grad_clip_norm",
        type=float,
        default=1.0,
        help="global gradient-norm clipping; <=0 disables clipping",
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.1,
        help="CrossEntropyLoss label smoothing",
    )
    parser.add_argument(
        "--batch_split",
        type=int,
        default=1,
        help=(
            "split each loaded batch into smaller sequential micro-batches"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    # ------------------------------------------------------------------
    # Backbone and checkpoint paths
    # ------------------------------------------------------------------
    parser.add_argument(
        "--backbone_path",
        type=str,
        default="",
        help="path to pretrained CLIP backbone weights",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="optional old Vita-CLIP/model-only checkpoint",
    )

    # ------------------------------------------------------------------
    # Model parameters
    # ------------------------------------------------------------------
    parser.add_argument(
        "--patch_size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--feature_dim",
        type=int,
        default=768,
    )
    parser.add_argument(
        "--embed_dim",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--mlp_factor",
        type=float,
        default=4.0,
    )
    parser.add_argument(
        "--cls_dropout",
        type=float,
        default=0.5,
        help=(
            "retained for compatibility; current text-similarity classifier "
            "does not use a separate linear classification head"
        ),
    )

    # ------------------------------------------------------------------
    # Optional zero-shot evaluation
    # ------------------------------------------------------------------
    parser.add_argument(
        "--zeroshot_evaluation",
        action="store_true",
        dest="zeroshot_evaluation",
    )
    parser.add_argument(
        "--zeroshot_text_features_path",
        type=str,
        default="./ucf101_text_features_B16/class-only.pth",
    )
    parser.add_argument(
        "--use_bie",
        action="store_true",
        dest="use_bie",
        help="enable Bidirectional Interpolation Enhancement (supervised only)",
    )
    parser.add_argument(
        "--bie_p_temperature",
        type=float,
        default=10.0,
        help="dedicated temperature of the classification softmax that drives "
        "the BIE interpolation weights. The shared logit_scale (~100) "
        "saturates the softmax into one-hot and starves the p path of "
        "gradients; 10.0 keeps p soft while preserving class separability. "
        "(default: 10.0)",
    )
    parser.add_argument(
        "--bie_max_gate",
        type=float,
        default=0.5,
        help="BIE injection gate upper bound for both directions "
        "(identity start: gates are zero at init; default: 0.5)",
    )
    parser.add_argument(
        "--bie_gate_warmup_epochs",
        type=float,
        default=0.0,
        help="number of epochs over which BIE injection gates ramp from 0 to "
        "1 (linear warmup; 0 disables warmup)",
    )
    parser.add_argument(
        "--bie_lr",
        type=float,
        default=1e-4,
        help="learning rate for the BIE transform MLPs and dynamic gate MLP "
        "(default: 1e-4)",
    )
    parser.add_argument(
        "--no_bie_transform",
        action="store_false",
        dest="bie_use_transform",
        default=True,
        help="disable the per-direction residual transform MLPs, leaving pure "
        "interpolation (default: enabled)",
    )
    parser.add_argument(
        "--bie_transform_bottleneck",
        type=int,
        default=128,
        help="hidden dim of the BIE transform MLPs (default: 128)",
    )
    parser.add_argument(
        "--bie_gate_bottleneck",
        type=int,
        default=128,
        help="hidden dim of the BIE dynamic video gate MLP (default: 128)",
    )
    parser.add_argument(
        "--bie_enh_thr",
        type=float,
        default=0.0,
        help="confidence threshold for text-side enhancement: only classes with "
        "soft classification probability p > bie_enh_thr get their text pulled "
        "toward the video; others keep their prototype text. 0 disables the "
        "filter (default). A reasonable value is ~1.0/num_classes.",
    )
    parser.add_argument(
        "--bie_max_dist",
        type=float,
        default=0.05,
        help="BIE video-side consistency threshold: maximum allowed L2 distance "
        "between the enhanced and original video feature before the consistency "
        "regularizer penalizes (default: 0.05)",
    )
    parser.add_argument(
        "--bie_max_dist_t",
        type=float,
        default=0.05,
        help="BIE text-side consistency threshold: maximum allowed L2 distance "
        "between an enhanced class text and its prototype before the "
        "consistency regularizer penalizes (default: 0.05)",
    )
    parser.add_argument(
        "--bie_alpha_aux_lambda",
        type=float,
        default=0.1,
        help="weight of the auxiliary classification loss that trains the BIE "
        "independent alpha_head classifier (default: 0.1)",
    )
    parser.add_argument(
        "--use_scr",
        action="store_true",
        dest="use_scr",
        help="apply Semantic Class-Relation Refinement: push each class text "
        "prototype away from its most confusable neighbours (identity start, "
        "fp32, AMP-safe)",
    )
    parser.add_argument(
        "--scr_top_k",
        type=int,
        default=5,
        help="number of most-confusable neighbours each class is pushed away "
        "from (default: 5)",
    )
    parser.add_argument(
        "--scr_max_scale",
        type=float,
        default=0.5,
        help="upper bound of the SCR pushing strength lambda (default: 0.5)",
    )
    parser.add_argument(
        "--use_aggregation_mixer",
        action="store_true",
        dest="use_aggregation_mixer",
        help="apply a depthwise temporal Conv1d between the frame reshape and "
        "the mean aggregation (identity start, fp32, AMP-safe)",
    )
    parser.add_argument(
        "--aggregation_mixer_kernel",
        type=int,
        default=3,
        help="kernel size of the aggregation temporal mixer (default: 3)",
    )
    parser.add_argument(
        "--aggregation_mixer_type",
        type=str,
        default="transformer",
        choices=["conv", "transformer"],
        help="temporal mixer implementation: 'conv' (depthwise Conv1d) or "
        "'transformer' (lightweight temporal self-attention over frames)",
    )
    parser.add_argument(
        "--aggregation_mixer_heads",
        type=int,
        default=8,
        help="number of attention heads for the 'transformer' mixer",
    )
    parser.add_argument(
        "--aggregation_mixer_dropout",
        type=float,
        default=0.0,
        help="attention dropout for the 'transformer' mixer",
    )
    # ------------------------------------------------------------------
    # Correspondence-Guided Video Prompt Block (CVP)
    # ------------------------------------------------------------------
    parser.add_argument(
        "--use_cvp_block",
        action="store_true",
        dest="use_cvp_block",
        help="insert the CVP Block after frozen CLIP Block `cvp_insert_layer` "
        "(supervised only; mutually exclusive with --use_discriminative_module)",
    )
    parser.add_argument(
        "--cvp_insert_layer", type=int, default=4,
        help="CVP runs after resblocks[this-1] / before resblocks[this] "
        "(1-based CLIP Block `this`); supports 2/4/6/8 ablations (default: 4)",
    )
    parser.add_argument("--cvp_topk", type=int, default=64,
        help="patches selected per frame (32/64/96 ablations; default: 64)")
    parser.add_argument("--cvp_window_size", type=int, default=7,
        help="local correspondence window side on the 14x14 grid "
        "(5/7/14 ablations; 14 = whole frame; default: 7)")
    parser.add_argument("--cvp_num_prompts", type=int, default=4,
        help="number of dynamic video prompts (default: 4)")
    parser.add_argument("--cvp_bottleneck_dim", type=int, default=192,
        help="CVP attention/adapter bottleneck width (default: 192)")
    parser.add_argument("--cvp_num_heads", type=int, default=4,
        help="CVP attention heads (default: 4)")
    parser.add_argument("--cvp_gamma_min", type=float, default=0.1)
    parser.add_argument("--cvp_gamma_max", type=float, default=0.5)
    parser.add_argument("--cvp_gamma_init", type=float, default=0.2,
        help="initial residual strength; bounded [gamma_min, gamma_max] "
        "via sigmoid(raw_gamma) (default: 0.2)")
    parser.add_argument("--cvp_beta_min", type=float, default=0.5)
    parser.add_argument("--cvp_beta_max", type=float, default=1.5)
    parser.add_argument("--cvp_beta_init", type=float, default=1.0,
        help="initial prompt content strength; bounded [beta_min, beta_max] "
        "(default: 1.0)")
    parser.add_argument("--cvp_corr_loss_weight", type=float, default=0.1,
        help="weight of the correspondence reconstruction loss on selected "
        "patches (default: 0.1)")
    parser.add_argument(
        "--cvp_selection_method",
        type=str,
        default="adaptive_attention_motion",
        choices=("attention_only", "motion_only", "fixed_attention_motion",
                 "adaptive_attention_motion", "random_topk"),
        help="patch selection scoring (default: adaptive_attention_motion)",
    )
    parser.add_argument("--cvp_motion_max_weight", type=float, default=0.6,
        help="max motion weight: wD = this * reliability (default: 0.6)")
    parser.add_argument(
        "--cvp_current_context_attention",
        action="store_true",
        dest="cvp_current_context_attention",
        default=True,
        help="two-stage mask query: read current-frame unselected input "
        "patches before querying past/future (default: true)",
    )
    parser.add_argument(
        "--cvp_no_current_context_attention",
        action="store_false",
        dest="cvp_current_context_attention",
        help="disable the current-frame context stage (v1 behavior)",
    )
    parser.add_argument(
        "--cvp_corr_scope", type=str, default="full",
        choices=("full", "local"),
        help="correspondence scope: full frame (default) or hard local "
        "window (ablation only)",
    )
    parser.add_argument("--cvp_spatial_bias_max", type=float, default=4.0,
        help="lambda_sp = this * sigmoid(raw) (0 disables spatial bias; "
        "default: 4.0)")
    parser.add_argument("--cvp_spatial_bias_init", type=float, default=1.0,
        help="initial lambda_sp (default: 1.0)")
    parser.add_argument(
        "--cvp_lr", type=float, default=2.0e-4,
        help="learning rate of the cvp parameter family (default: 2e-4)",
    )
    # ------------------------------------------------------------------
    # TGL trajectory adapter (offline CoTracker3 caches; the tracker is
    # NEVER run during training - MODEL.TGL.RUN_TRACKER_ONLINE=False).
    # ------------------------------------------------------------------
    parser.add_argument(
        "--tgl_enable", action="store_true", dest="tgl_enable",
        help="enable the Trajectory-Guided Local Spatio-Temporal Adapter "
        "(MODEL.TGL.ENABLE=True)",
    )
    parser.add_argument(
        "--tgl_preset", type=str, default="tat_like",
        choices=sorted(TGL_PRESETS.keys()),
        help="TGL preset (MODEL.TGL.PRESET): tat_like = nearest gather/"
        "scatter, random sampler, no tube, no bilinear, no score/semantic "
        "samplers",
    )
    parser.add_argument(
        "--tgl_run_tracker_online", action="store_true",
        dest="tgl_run_tracker_online",
        help="FORBIDDEN. MODEL.TGL.RUN_TRACKER_ONLINE must stay False: the "
        "trainer hard-errors when this flag is set.",
    )
    parser.add_argument("--tgl_adapter_dim", type=int, default=None,
        help="TGL bottleneck d (default: C//4)")
    parser.add_argument("--tgl_num_layers", type=int, default=2,
        help="TGL factorized space-time blocks (default: 2)")
    parser.add_argument("--tgl_num_heads", type=int, default=4,
        help="TGL attention heads (default: 4)")
    parser.add_argument("--tgl_dropout", type=float, default=0.1,
        help="TGL dropout (default: 0.1)")
    parser.add_argument("--tgl_max_scale", type=float, default=0.1,
        help="TGL residual cap: scale = max_scale * tanh(gamma) (default: 0.1)")
    parser.add_argument("--tgl_max_fusion", type=float, default=0.5,
        help="cap of the zero-init CLS/patch fusion gate (default: 0.5)")
    parser.add_argument("--tgl_override_gather", type=str, default=None,
        choices=("nearest", "bilinear"), help=None,
    )
    parser.add_argument("--tgl_insert_layers", type=str, default="6,8,10",
        help="comma-separated ViT block indices AFTER which the TGL adapter "
        "writes its residual (insert before block L+1); e.g. '6,8,10'",
    )
    parser.add_argument("--tgl_num_prompts", type=int, default=4,
        help="trajectory-residual video prompts emitted per insertion "
        "(source = residual tokens via visibility-weighted pool; "
        "0 disables)")
    parser.add_argument("--tgl_patch_pool_baseline", action="store_true",
        dest="tgl_patch_pool_baseline",
        help="ABLATION ONLY: also add the mean_pool(enhanced_patches) "
        "readout behind its own zero-init gate (default OFF - it is the "
        "confounded branch that cannot attribute gains to trajectories)")
    parser.add_argument("--tgl_gate_init", type=float, default=0.04,
        dest="tgl_gate_init",
        help="initial gate opening in SCALE units (scale = max_scale * "
        "tanh(gamma) starts here). 0 = strict identity start, but in "
        "practice the zero-gate cold-start starves the adapter of gradient. "
        "Warm start 0.04 = ~4%% patch perturbation immediately, bounded by "
        "max_scale=0.1.")
    parser.add_argument("--tgl_readout_gate", type=float, default=0.15,
        dest="tgl_readout_gate",
        help="initial fusion weight of the TAT-style trajectory readout "
        "(learnable-query attention pooling of the trajectory tokens, fused "
        "into the video feature; warm init, trains from step 1)")
    parser.add_argument("--tgl_v2", action="store_true", dest="tgl_v2",
        help="use the TGL v2 image-side architecture: per-layer video "
        "prompts (CVPT-style injection) + Track-ST trajectory branch + "
        "Trajectory CLS + pure-image adaptive late fusion; the legacy CVP "
        "residual path is bypassed")
    parser.add_argument("--tgl_v2_prompt_source_layer", type=int, default=4,
        help="1-based block whose output builds the 8 frame tokens "
        "(prompt_source_layer; must precede every prompt layer)")
    parser.add_argument("--tgl_v2_prompt_layers", type=str,
        default="5,6,7,8,9,10,11,12",
        help="1-based blocks receiving prompt injection")
    parser.add_argument("--tgl_v2_recon_ratio", type=float, default=0.1,
        help="fraction of valid trajectory positions masked for the "
        "reconstruction auxiliary (training only)")
    parser.add_argument("--tgl_loss_traj_aux", type=float, default=0.2,
        help="weight of the trajectory auxiliary classifier loss")
    parser.add_argument("--tgl_loss_recon", type=float, default=0.1,
        help="weight of the masked reconstruction auxiliary loss")
    # ------------------------------------------------------------------
    # Decoupled layered video adaptation (v3): mask adapter + layered
    # per-layer video prompts. Mutually exclusive with --use_cvp_block.
    # References: VPT-Deep (ECCV'22, per-layer independent prompts),
    # AIM (ICCV'23) / ST-Adapter (NeurIPS'22, per-layer residual adapters).
    # ------------------------------------------------------------------
    parser.add_argument(
        "--use_mask_adapter",
        action="store_true",
        dest="use_mask_adapter",
        help="enable the per-layer mask patch adapter (residual, "
        "zero-init gate; selection re-scored per insertion layer)",
    )
    parser.add_argument(
        "--mask_adapter_layers", type=str, default="2,4,6,8",
        help="1-based frozen blocks AFTER which the mask adapter runs "
        "(default: 2,4,6,8 -- shallow layers per ST-Adapter)",
    )
    parser.add_argument(
        "--mask_adapter_topk", type=int, default=64,
        help="patches selected per frame per insertion layer (default: 64)",
    )
    parser.add_argument(
        "--mask_adapter_bottleneck", type=int, default=192,
        help="mask adapter bottleneck width (default: 192)",
    )
    parser.add_argument(
        "--mask_adapter_heads", type=int, default=4,
        help="mask adapter temporal attention heads (default: 4)",
    )
    parser.add_argument(
        "--mask_adapter_motion_max_weight", type=float, default=0.6,
        help="max motion weight in the adapter's mask scoring (default: 0.6)",
    )
    parser.add_argument(
        "--use_layered_prompts",
        action="store_true",
        dest="use_layered_prompts",
        help="enable VPT-Deep-style per-layer video prompts (independent "
        "per-layer embeddings + video conditioning)",
    )
    parser.add_argument(
        "--num_video_prompts", type=int, default=4,
        help="prompts injected at each layer (default: 4)",
    )
    parser.add_argument(
        "--prompt_cond_rank", type=int, default=192,
        help="low-rank width of the per-layer conditioning head (default: 192)",
    )
    parser.add_argument(
        "--prompt_insert_layers", type=str, default="all",
        help="'all' or comma-separated 1-based blocks receiving prompts "
        "(default: all)",
    )
    parser.add_argument(
        "--cvp_debug_visualization",
        action="store_true",
        dest="cvp_debug_visualization",
        help="save the first-batch CVP visualizations (attention/diff/"
        "selection/residual heatmaps + correspondence positions)",
    )
    parser.add_argument(
        "--cvp_debug_dir", type=str, default="training/cvp_vis",
        help="output directory for CVP debug visualizations",
    )

    parser.add_argument(
        "--grad_accum",
        type=int,
        default=1,
        help="number of data steps to accumulate gradients over before one "
        "optimizer step; effective batch = batch_size * grad_accum. "
        "Warmup/cosine scheduling is rescaled accordingly. (default: 1)",
    )
    parser.add_argument(
        "--use_model_ema",
        action="store_true",
        help="maintain an exponential-moving-average copy of the trainable "
        "parameters and evaluate/save the EMA weights. Strongest remedy for "
        "the train-acc=100%% / val-plateau gap (ml-training-recipes standard "
        "practice). EMA shadow is rebuilt from the resumed weights, so it is "
        "not persisted in checkpoints.",
    )
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.999,
        help="EMA decay factor (default: 0.999)",
    )

    # ------------------------------------------------------------------
    # Mixed precision
    # ------------------------------------------------------------------
    parser.add_argument(
        "--use_fp16",
        action="store_true",
        dest="fp16",
        help="enable CUDA AMP mixed precision",
    )
    parser.set_defaults(fp16=False)

    # ------------------------------------------------------------------
    # Original Vita-CLIP visual prompts
    # ------------------------------------------------------------------
    parser.add_argument(
        "--use_summary_token",
        action="store_true",
        dest="use_summary_token",
    )
    parser.add_argument(
        "--use_local_prompts",
        action="store_true",
        dest="use_local_prompts",
    )
    parser.add_argument(
        "--use_global_prompts",
        action="store_true",
        dest="use_global_prompts",
    )
    parser.add_argument(
        "--num_global_prompts",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--use_original_visual_prompts",
        action="store_true",
        dest="use_original_visual_prompts",
        help="restore original global/local/summary visual prompts",
    )
    parser.set_defaults(
        use_original_visual_prompts=False,
        use_summary_token=False,
        use_local_prompts=False,
        use_global_prompts=False,
    )

    # ------------------------------------------------------------------
    # Discriminative region + RWKV temporal module
    # ------------------------------------------------------------------
    parser.add_argument(
        "--use_discriminative_module",
        action="store_true",
        dest="use_discriminative_module",
        help=(
            "enable the discriminative region + RWKV temporal module "
            "(STOP-style top-k selection + shared-weight RWKV)"
        ),
    )
    parser.add_argument(
        "--no_discriminative_module",
        action="store_false",
        dest="use_discriminative_module",
    )
    parser.add_argument(
        "--dr_layers",
        type=str,
        default="3,5,7,9",
        help="zero-based transformer layers where the module runs",
    )
    parser.add_argument(
        "--share_discriminative_module",
        action="store_true",
        dest="share_discriminative_module",
        help="share one RWKV module across all dr layers",
    )
    parser.add_argument(
        "--dr_arch",
        type=str,
        default="rwkv",
        choices=("rwkv", "dual", "rap"),
        help="discriminative temporal module: 'rwkv' (original, default) "
        "or 'dual' (conv3d + full Mamba with gating fusion)",
    )
    parser.add_argument(
        "--dr_top_k",
        type=int,
        default=12,
        help="number of discriminative patches kept per frame (STOP N_s)",
    )
    parser.add_argument(
        "--dr_alpha",
        type=float,
        default=0.4,
        help="fusion weight: W = alpha * attention + (1-alpha) * motion",
    )
    parser.add_argument(
        "--dr_bottleneck",
        type=int,
        default=192,
        help="RWKV bottleneck width",
    )
    parser.add_argument(
        "--dr_rwkv_layers",
        type=int,
        default=1,
        help="number of lightweight RWKV blocks",
    )
    parser.add_argument(
        "--dr_residual_scale_init",
        type=float,
        default=0.01,
        help="initial raw residual gate for the discriminative module",
    )
    parser.add_argument(
        "--dr_topk_temperature",
        type=float,
        default=0.5,
        help="softmax temperature of the soft top-k CLS aggregation; "
        "lower = sharper/adapt faster, higher = more uniform selection",
    )
    parser.add_argument(
        "--dr_mamba_scope",
        type=str,
        default="full",
        choices=("full", "union"),
        help="Mamba scan scope: 'full' scans all spatial positions "
        "spatial-first (cross-position flow), 'union' scans only the "
        "selected union positions per-frame (no cross-position flow)",
    )
    parser.add_argument(
        "--no_dr_conv3d",
        action="store_true",
        dest="no_dr_conv3d",
        help="disable the conv3d branch of the dual module (single-branch "
        "attribution: Mamba only)",
    )
    parser.add_argument(
        "--no_dr_mamba",
        action="store_true",
        dest="no_dr_mamba",
        help="disable the Mamba branch of the dual module (single-branch "
        "attribution: conv3d only)",
    )
    parser.add_argument(
        "--dr_mamba_bidir",
        action="store_true",
        dest="dr_mamba_bidir",
        help="run the Mamba selective scan both forward and backward and "
        "sum the outputs",
    )
    parser.add_argument(
        "--dr_d_state",
        type=int,
        default=16,
        help="selective-scan state size of the Mamba branch",
    )
    parser.add_argument(
        "--dr_d_conv",
        type=int,
        default=4,
        help="causal conv kernel size inside the Mamba branch",
    )
    parser.add_argument(
        "--dr_use_asa",
        action="store_true",
        dest="dr_use_asa",
        default=True,
        help="enable the RAP sparse-ASA branch of the rap module",
    )
    parser.add_argument(
        "--no_dr_asa",
        action="store_false",
        dest="dr_use_asa",
        help="disable the RAP sparse-ASA branch of the rap module "
        "(single-branch attribution: conv3d only)",
    )
    parser.add_argument(
        "--dr_asa_heads",
        type=int,
        default=8,
        help="number of attention heads of the RAP sparse-ASA branch "
        "(dim must be divisible by heads)",
    )
    parser.add_argument(
        "--dr_asa_time_offset_max",
        type=float,
        default=3.0,
        help="max |temporal offset| (frames) predicted per query in the RAP "
        "sparse-ASA branch; 0 -> identity sampling at start",
    )
    parser.add_argument(
        "--dr_asa_space_offset_max",
        type=float,
        default=4.0,
        help="max |spatial offset| (patch columns) predicted per query in "
        "the RAP sparse-ASA branch; 0 -> identity sampling at start",
    )
    parser.add_argument(
        "--dr_asa_window_size",
        type=int,
        default=5,
        help="side length of the square sampling window around each query "
        "offset anchor in the RAP sparse-ASA branch; the number of "
        "cross-frame candidate keys/values per query is window_size^2 "
        "(5 -> 25 candidates, a reduced 7x7 RAP paper window). Must be odd.",
    )
    parser.set_defaults(
        use_discriminative_module=False,
        share_discriminative_module=False,
        no_dr_conv3d=False,
        no_dr_mamba=False,
        dr_mamba_bidir=False,
        dr_use_asa=True,
    )

    # ------------------------------------------------------------------
    # Learnable text prompt
    # ------------------------------------------------------------------
    parser.add_argument(
        "--use_text_prompt_learning",
        action="store_true",
        dest="use_text_prompt_learning",
    )
    parser.add_argument(
        "--no_text_prompt_learning",
        action="store_false",
        dest="use_text_prompt_learning",
    )
    parser.add_argument(
        "--text_context_length",
        type=int,
        default=77,
    )
    parser.add_argument(
        "--text_vocab_size",
        type=int,
        default=49408,
    )
    parser.add_argument(
        "--text_transformer_width",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--text_transformer_heads",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--text_transformer_layers",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--text_num_prompts",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--text_prompt_pos",
        type=str,
        default="end",
        choices=("end", "middle", "front"),
    )
    parser.add_argument(
        "--text_prompt_init",
        type=str,
        default="",
    )
    parser.add_argument(
        "--use_text_prompt_CSC",
        action="store_true",
        dest="text_prompt_CSC",
    )
    parser.add_argument(
        "--text_prompt_classes_path",
        type=str,
        default=(
            "/hy-tmp/UCF101/annotations/"
            "ucfTrainTestlist/classInd.txt"
        ),
    )
    parser.set_defaults(
        use_text_prompt_learning=True,
        text_prompt_CSC=False,
    )

    # ------------------------------------------------------------------
    # Two text branches
    # ------------------------------------------------------------------
    parser.add_argument(
        "--use_fixed_text_branch",
        action="store_true",
        dest="use_fixed_text_branch",
    )
    parser.add_argument(
        "--no_fixed_text_branch",
        action="store_false",
        dest="use_fixed_text_branch",
    )
    parser.add_argument(
        "--use_learnable_text_branch",
        action="store_true",
        dest="use_learnable_text_branch",
    )
    parser.add_argument(
        "--no_learnable_text_branch",
        action="store_false",
        dest="use_learnable_text_branch",
    )
    parser.add_argument(
        "--use_text_mstff",
        action="store_true",
        dest="use_text_mstff",
    )
    parser.add_argument(
        "--no_text_mstff",
        action="store_false",
        dest="use_text_mstff",
    )
    parser.add_argument(
        "--fixed_text_template",
        type=str,
        default="a video of a person {}",
    )
    parser.add_argument(
        "--text_mstff_gate_init",
        type=float,
        default=0.01,
        help="initial residual gate for Text_MSTFF",
    )
    parser.add_argument(
        "--text_fusion_fixed_logit",
        type=float,
        default=3.0,
        help="initial fusion logit for the fixed text branch",
    )
    parser.add_argument(
        "--text_fusion_learnable_logit",
        type=float,
        default=0.0,
        help="initial fusion logit for the learnable text branch",
    )
    parser.add_argument(
        "--naturalize_class_names",
        action="store_true",
        dest="naturalize_class_names",
    )
    parser.add_argument(
        "--no_naturalize_class_names",
        action="store_false",
        dest="naturalize_class_names",
    )
    parser.set_defaults(
        use_fixed_text_branch=True,
        use_learnable_text_branch=True,
        use_text_mstff=True,
        naturalize_class_names=True,
    )

    args = parser.parse_args()

    if args.epochs <= 0:
        raise ValueError(
            f"epochs must be positive, got {args.epochs}."
        )
    if args.batch_split <= 0:
        raise ValueError(
            f"batch_split must be positive, got {args.batch_split}."
        )
    if not 0.0 <= args.label_smoothing < 1.0:
        raise ValueError(
            "label_smoothing must be in [0, 1)."
        )
    # Layered v3 modules are mutually exclusive with the single-shot CVP
    # block (both rewrite the token stream between frozen blocks).
    if args.use_cvp_block and (args.use_mask_adapter or args.use_layered_prompts):
        raise ValueError(
            "--use_cvp_block is mutually exclusive with "
            "--use_mask_adapter / --use_layered_prompts. Enable exactly "
            "one visual adaptation design."
        )
    if (args.use_mask_adapter or args.use_layered_prompts) and (
        args.use_discriminative_module
    ):
        raise ValueError(
            "--use_mask_adapter / --use_layered_prompts are mutually "
            "exclusive with --use_discriminative_module."
        )
    # ------------------------------------------------------------------
    # TGL validation (MODEL.TGL.*). Online tracking is FORBIDDEN: the
    # tracker is run exclusively by the offline preprocessing pipeline
    # (run_tgl_pipeline.py -> tgl_tracker_offline.py) before training.
    # ------------------------------------------------------------------
    if args.tgl_run_tracker_online:
        raise SystemExit(
            "FATAL: MODEL.TGL.RUN_TRACKER_ONLINE=True is forbidden. The "
            "CoTracker3 tracker must never run inside the dataloader or the "
            "model forward. Precompute all trajectories offline with "
            "training/run_tgl_pipeline.py --auto-track and pass "
            "--tgl_track_root."
        )
    if args.tgl_enable and not args.tgl_track_root:
        raise ValueError(
            "--tgl_enable requires --tgl_track_root (offline track caches)."
        )
    if args.tgl_enable:
        args.tgl_preset_resolved = resolve_tgl_preset(args.tgl_preset)
        if args.tgl_override_gather is not None:
            args.tgl_preset_resolved["gather"] = args.tgl_override_gather
            args.tgl_preset_resolved["scatter"] = args.tgl_override_gather
        print(
            f"[TGL] enabled | preset={args.tgl_preset} -> "
            f"gather={args.tgl_preset_resolved['gather']} "
            f"sampler={args.tgl_preset_resolved['offline_sampler']} "
            f"tube={args.tgl_preset_resolved['local_tube_size']} | "
            f"track_root={args.tgl_track_root} | "
            f"run_tracker_online=False"
        )
    else:
        args.tgl_preset_resolved = None
    # The original project is launched with torchrun/distributed launch.
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    setup_print(rank == 0)
    set_random_seed(args.seed, rank)

    cuda_device_id = rank % torch.cuda.device_count()
    torch.cuda.set_device(cuda_device_id)

    total_steps, steps_per_epoch, warmup_steps = (
        resolve_training_schedule(
            args,
            world_size=world_size,
        )
    )

    # BIE gate warmup: convert (fractional) epochs to global steps. During
    # the first gate_warmup_steps the injection is scaled linearly from 0,
    # so early training is exactly the baseline behavior.
    bie_gate_warmup_steps = 0
    if args.bie_gate_warmup_epochs > 0:
        if steps_per_epoch is not None:
            bie_gate_warmup_steps = int(
                round(args.bie_gate_warmup_epochs * steps_per_epoch)
            )
        else:
            bie_gate_warmup_steps = int(
                round(
                    args.bie_gate_warmup_epochs
                    / max(1.0, float(args.epochs))
                    * float(total_steps)
                )
            )

    model = VitaCLIP(
        backbone_path=args.backbone_path,
        input_size=(
            args.spatial_size,
            args.spatial_size,
        ),
        num_frames=args.num_frames,
        num_classes=args.num_classes,
        feature_dim=args.feature_dim,
        patch_size=(
            args.patch_size,
            args.patch_size,
        ),
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        mlp_factor=args.mlp_factor,
        embed_dim=args.embed_dim,
        use_original_visual_prompts=(
            args.use_original_visual_prompts
        ),
        use_summary_token=args.use_summary_token,
        use_local_prompts=args.use_local_prompts,
        use_global_prompts=args.use_global_prompts,
        num_global_prompts=args.num_global_prompts,
        use_discriminative_module=(
            args.use_discriminative_module
        ),
        dr_layers=(
            tuple(
                int(value)
                for value in args.dr_layers.split(",")
                if value.strip()
            )
            if args.use_discriminative_module
            else ()
        ),
        share_discriminative_module=(
            args.share_discriminative_module
        ),
        dr_arch=(
            args.dr_arch if args.use_discriminative_module else "rwkv"
        ),
        dr_top_k=args.dr_top_k,
        dr_alpha=args.dr_alpha,
        dr_bottleneck=args.dr_bottleneck,
        dr_rwkv_layers=args.dr_rwkv_layers,
        dr_residual_scale_init=(
            args.dr_residual_scale_init
        ),
        dr_topk_temperature=args.dr_topk_temperature,
        dr_mamba_scope=(
            args.dr_mamba_scope if args.use_discriminative_module else "full"
        ),
        dr_use_conv3d=(
            args.use_discriminative_module and not args.no_dr_conv3d
        ),
        dr_use_mamba=(
            args.use_discriminative_module and not args.no_dr_mamba
        ),
        dr_use_asa=(
            args.use_discriminative_module and args.dr_use_asa
        ),
        dr_asa_heads=args.dr_asa_heads,
        dr_asa_time_offset_max=args.dr_asa_time_offset_max,
        dr_asa_space_offset_max=args.dr_asa_space_offset_max,
        dr_asa_window_size=args.dr_asa_window_size,
        dr_mamba_bidir=(
            args.dr_mamba_bidir if args.use_discriminative_module else False
        ),
        dr_d_state=args.dr_d_state,
        dr_d_conv=args.dr_d_conv,
        use_text_prompt_learning=(
            args.use_text_prompt_learning
        ),
        text_context_length=args.text_context_length,
        text_vocab_size=args.text_vocab_size,
        text_transformer_width=(
            args.text_transformer_width
        ),
        text_transformer_heads=(
            args.text_transformer_heads
        ),
        text_transformer_layers=(
            args.text_transformer_layers
        ),
        text_num_prompts=args.text_num_prompts,
        text_prompt_pos=args.text_prompt_pos,
        text_prompt_init=args.text_prompt_init,
        text_prompt_CSC=args.text_prompt_CSC,
        text_prompt_classes_path=(
            args.text_prompt_classes_path
        ),
        use_fixed_text_branch=(
            args.use_fixed_text_branch
        ),
        use_learnable_text_branch=(
            args.use_learnable_text_branch
        ),
        use_text_mstff=args.use_text_mstff,
        fixed_text_template=args.fixed_text_template,
        naturalize_class_names=args.naturalize_class_names,
        text_mstff_gate_init=args.text_mstff_gate_init,
        text_fusion_fixed_logit=(
            args.text_fusion_fixed_logit
        ),
        text_fusion_learnable_logit=(
            args.text_fusion_learnable_logit
        ),
        zeroshot_evaluation=args.zeroshot_evaluation,
        zeroshot_text_features_path=(
            args.zeroshot_text_features_path
        ),
        use_bie=args.use_bie,
        bie_p_temperature=args.bie_p_temperature,
        bie_max_gate=args.bie_max_gate,
        bie_gate_warmup_steps=bie_gate_warmup_steps,
        bie_use_transform=args.bie_use_transform,
        bie_transform_bottleneck=args.bie_transform_bottleneck,
        bie_gate_bottleneck=args.bie_gate_bottleneck,
        bie_enh_thr=args.bie_enh_thr,
        bie_max_dist=args.bie_max_dist,
        bie_max_dist_t=args.bie_max_dist_t,
        bie_alpha_aux_lambda=args.bie_alpha_aux_lambda,
        use_scr=args.use_scr,
        scr_top_k=args.scr_top_k,
        scr_max_scale=args.scr_max_scale,
        use_aggregation_mixer=args.use_aggregation_mixer,
        aggregation_mixer_kernel=(
            args.aggregation_mixer_kernel
        ),
        aggregation_mixer_type=args.aggregation_mixer_type,
        aggregation_mixer_heads=args.aggregation_mixer_heads,
        aggregation_mixer_dropout=args.aggregation_mixer_dropout,
        use_cvp_block=args.use_cvp_block,
        cvp_insert_layer=args.cvp_insert_layer,
        cvp_topk=args.cvp_topk,
        cvp_window_size=args.cvp_window_size,
        cvp_num_prompts=args.cvp_num_prompts,
        cvp_bottleneck_dim=args.cvp_bottleneck_dim,
        cvp_num_heads=args.cvp_num_heads,
        cvp_gamma_min=args.cvp_gamma_min,
        cvp_gamma_max=args.cvp_gamma_max,
        cvp_gamma_init=args.cvp_gamma_init,
        cvp_beta_min=args.cvp_beta_min,
        cvp_beta_max=args.cvp_beta_max,
        cvp_beta_init=args.cvp_beta_init,
        cvp_corr_loss_weight=args.cvp_corr_loss_weight,
        cvp_selection_method=args.cvp_selection_method,
        cvp_motion_max_weight=args.cvp_motion_max_weight,
        cvp_current_context_attention=args.cvp_current_context_attention,
        cvp_corr_scope=args.cvp_corr_scope,
        cvp_spatial_bias_max=args.cvp_spatial_bias_max,
        cvp_spatial_bias_init=args.cvp_spatial_bias_init,
        use_mask_adapter=args.use_mask_adapter,
        mask_adapter_layers=args.mask_adapter_layers,
        mask_adapter_topk=args.mask_adapter_topk,
        mask_adapter_bottleneck=args.mask_adapter_bottleneck,
        mask_adapter_heads=args.mask_adapter_heads,
        mask_adapter_motion_max_weight=args.mask_adapter_motion_max_weight,
        use_layered_prompts=args.use_layered_prompts,
        num_video_prompts=args.num_video_prompts,
        prompt_cond_rank=args.prompt_cond_rank,
        prompt_insert_layers=args.prompt_insert_layers,
        use_tgl_adapter=args.tgl_enable and not args.tgl_v2,
        tgl_adapter_dim=args.tgl_adapter_dim,
        tgl_num_layers=args.tgl_num_layers,
        tgl_num_heads=args.tgl_num_heads,
        tgl_dropout=args.tgl_dropout,
        tgl_gather_mode=(
            str(args.tgl_preset_resolved["gather"]) if args.tgl_enable else "nearest"
        ),
        tgl_local_tube_size=(
            int(args.tgl_preset_resolved["local_tube_size"])
            if args.tgl_enable
            else 1
        ),
        tgl_visibility_threshold=0.5,
        tgl_max_scale=args.tgl_max_scale,
        tgl_max_fusion=args.tgl_max_fusion,
        tgl_insert_layers=args.tgl_insert_layers,
        tgl_num_prompts=args.tgl_num_prompts,
        tgl_patch_pool_baseline=args.tgl_patch_pool_baseline,
        tgl_gate_init=args.tgl_gate_init,
        tgl_readout=True,
        use_tgl_v2=args.tgl_v2,
        tgl_v2_prompt_source_layer=args.tgl_v2_prompt_source_layer,
        tgl_v2_prompt_layers=args.tgl_v2_prompt_layers,
        tgl_v2_num_prompts=args.tgl_num_prompts,
        tgl_v2_num_classes=args.num_classes,
        tgl_v2_recon_ratio=args.tgl_v2_recon_ratio,
        tgl_v2_readout_gate=args.tgl_readout_gate,
        tgl_readout_gate=args.tgl_readout_gate,
        )

    load_optional_model_checkpoint(
        model,
        args.checkpoint_path,
    )

    print(model)
    print("----------------------------------------------------")
    print("Trainable Parameters")
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            print(name)
    print("----------------------------------------------------")

    model.cuda()

    # Build optimizer before DDP so group names stay readable. Parameter
    # references remain valid after wrapping in DistributedDataParallel.
    optimizer = build_optimizer(
        model,
        args,
    )
    # The learning-rate scheduler is advanced once per *optimizer* step
    # (see the do_optimizer_step guard in the training loop), but
    # ``total_steps`` / ``warmup_steps`` are expressed in *data* steps.
    # Without this correction the cosine schedule only traverses
    # 1 / grad_accum of its full range and never anneals to min_lr_ratio,
    # which leaves the LR high at the end of training and causes the
    # late-training validation oscillation / post-peak collapse.
    optimizer_steps = max(
        1,
        total_steps // args.grad_accum,
    )
    optimizer_warmup_steps = max(
        0,
        min(warmup_steps // args.grad_accum, optimizer_steps - 1),
    )
    lr_sched = build_warmup_cosine_scheduler(
        optimizer,
        total_steps=optimizer_steps,
        warmup_steps=optimizer_warmup_steps,
        min_lr_ratio=args.min_lr_ratio,
    )

    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[cuda_device_id],
        output_device=cuda_device_id,
        # True: conditional branches (e.g. disabled BIE transform, warmup
        # gates at zero) may leave some params unused per iteration.
        # Single-GPU DDP adds no communication overhead for this.
        find_unused_parameters=True,
        # static_graph=True resolves the DDP "marked as ready twice" error
        # caused by the BIE alpha_aux_loss (a second learnable classifier path
        # that shares the alpha_head parameters with the enhancement graph)
        # under gradient accumulation. The model's compute graph is fixed
        # across iterations (gate warmup only changes scalar values, not
        # structure), so a static graph is valid here.
        static_graph=True,
    )

    loss_scaler = torch.cuda.amp.GradScaler(
        enabled=args.fp16
    )
    criterion = torch.nn.CrossEntropyLoss(
        label_smoothing=args.label_smoothing
    )

    resume_step = checkpoint.resume_from_checkpoint(
        model,
        optimizer,
        lr_sched,
        loss_scaler,
        args,
    )

    # ---- Optional model EMA (evaluated weights, not persisted) ----
    ema_shadow = None
    ema_trainable = None
    ema_backup = None
    if args.use_model_ema:
        if not 0.0 < args.ema_decay < 1.0:
            raise ValueError("--ema_decay must be in (0, 1).")
        ema_trainable = [
            (name, param)
            for name, param in model.named_parameters()
            if param.requires_grad
        ]
        ema_shadow = {
            name: param.detach().clone()
            for name, param in ema_trainable
        }
        ema_backup = {}
        print(
            f"[EMA] tracking {len(ema_trainable)} trainable parameters "
            f"with decay {args.ema_decay}"
        )

    val_loader = video_dataset.create_val_loader(args)

    if args.eval_only:
        print("Running in eval_only mode.")
        model.eval()
        evaluate(
            model,
            val_loader,
            frame_shuffle=getattr(args, "eval_frame_shuffle", False),
            tgl=args.tgl_enable,
        )
        return

    if args.train_list_path is None:
        raise ValueError(
            "Train list path must be specified outside eval_only mode."
        )

    train_loader = video_dataset.create_train_loader(
        args,
        resume_step=resume_step,
    )

    expected_remaining_steps = (
        args.num_steps - resume_step
    )
    if len(train_loader) != expected_remaining_steps:
        raise RuntimeError(
            "Train-loader length does not match the requested schedule: "
            f"loader={len(train_loader)}, "
            f"expected={expected_remaining_steps}."
        )

    batch_start = datetime.now()
    training_start = datetime.now()

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    best_top1 = 0.0
    best_step = 0

    for step, batch in enumerate(
        train_loader,
        resume_step,
    ):
        if args.tgl_enable:
            # TGL mode: dataset yields (frames, label, tracks, vis, birth).
            # Tracks are OFFLINE caches - nothing here ever runs a tracker.
            data, labels, tracks, visibility, birth_frame = batch
            tracks = tracks.cuda(non_blocking=True)
            visibility = visibility.cuda(non_blocking=True)
            birth_frame = birth_frame.cuda(non_blocking=True)
        else:
            data, labels = batch
        data = data.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        data_end = datetime.now()

        optimizer.zero_grad(set_to_none=True)

        if data.size(0) % args.batch_split != 0:
            raise ValueError(
                f"Batch size {data.size(0)} is not divisible by "
                f"batch_split={args.batch_split}."
            )

        split_size = data.size(0) // args.batch_split
        hit1 = 0
        hit5 = 0
        loss_value = 0.0

        for split_index in range(args.batch_split):
            start = split_size * split_index
            end = split_size * (split_index + 1)

            data_slice = data[start:end]
            labels_slice = labels[start:end]
            tgl_kwargs = {}
            if args.tgl_enable:
                tgl_kwargs = {
                    "tracks": tracks[start:end],
                    "visibility": visibility[start:end],
                    "birth_frame": birth_frame[start:end],
                }

            with torch.cuda.amp.autocast(
                enabled=args.fp16
            ):
                logits = model(
                    data_slice,
                    labels_slice,
                    bie_step=step,
                    **tgl_kwargs,
                )
                loss = criterion(
                    logits,
                    labels_slice,
                )
                align_loss = getattr(
                    (
                        model.module
                        if hasattr(model, "module")
                        else model
                    ),
                    "_last_align_loss",
                    None,
                )
                if align_loss is not None:
                    loss = loss + align_loss
                # BIE consistency regularizer: keeps the enhanced video feature
                # close to the original (cosine >= cos_thr). The lambda is
                # already folded into the value inside BIE.
                raw_model = (
                    model.module
                    if hasattr(model, "module")
                    else model
                )
                consistency_loss = getattr(
                    raw_model, "_last_consistency_loss", None
                )
                if consistency_loss is not None:
                    loss = loss + consistency_loss
                # BIE auxiliary loss for the independent α classifier
                # (trains alpha_head; lambda folded in inside BIE).
                alpha_aux_loss = getattr(
                    raw_model, "_last_alpha_aux_loss", None
                )
                if alpha_aux_loss is not None:
                    loss = loss + alpha_aux_loss
                # CVP correspondence reconstruction loss (selected patches
                # only; the target is stop-grad). Weight applied here.
                cvp_corr_loss = getattr(
                    raw_model, "_last_cvp_corr_loss", None
                )
                if cvp_corr_loss is not None:
                    loss = (
                        loss + args.cvp_corr_loss_weight * cvp_corr_loss
                    )
                # TGL v2 auxiliary losses (image-side only; text untouched):
                # trajectory auxiliary classifier + masked reconstruction.
                # Weights are config, not hard-coded module constants.
                if args.tgl_enable and args.tgl_v2:
                    raw_model = (
                        model.module if hasattr(model, "module") else model
                    )
                    v2_aux = getattr(raw_model, "_last_v2_aux", None) or {}
                    zero = logits.new_zeros(())
                    loss = loss + args.tgl_loss_traj_aux * v2_aux.get(
                        "loss_traj_aux", zero
                    )
                    loss = loss + args.tgl_loss_recon * v2_aux.get(
                        "loss_recon", zero
                    )

            if labels.dtype == torch.long:
                hit1 += (
                    logits.topk(1, dim=1)[1]
                    == labels_slice.view(-1, 1)
                ).sum().item()
                hit5 += (
                    logits.topk(5, dim=1)[1]
                    == labels_slice.view(-1, 1)
                ).sum().item()

            loss_value += (
                loss.item() / args.batch_split
            ) / args.grad_accum

            loss_scaler.scale(
                loss / args.batch_split / args.grad_accum
            ).backward()

        # First-batch CVP debug visualization (rank 0, once per run).
        if (
            args.use_cvp_block
            and step == resume_step
            and args.cvp_debug_visualization
        ):
            _cvp_debug_visualize(args, model, data, step, rank)

        # Gradient accumulation: step the optimizer every grad_accum
        # micro-batches (and on the final step of the schedule).
        do_optimizer_step = (
            ((step - resume_step + 1) % args.grad_accum == 0)
            or (step + 1 == args.num_steps)
        )

        if do_optimizer_step:
            if args.grad_clip_norm > 0:
                loss_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    trainable_parameters,
                    max_norm=args.grad_clip_norm,
                )

            loss_scaler.step(optimizer)
            loss_scaler.update()
            clamp_logit_scale(model)
            lr_sched.step()

        if do_optimizer_step and ema_shadow is not None:
            with torch.no_grad():
                for name, param in ema_trainable:
                    ema_shadow[name].mul_(args.ema_decay).add_(
                        param.detach(),
                        alpha=1.0 - args.ema_decay,
                    )

        batch_end = datetime.now()

        if step % args.print_freq == 0:
            sync_tensor = torch.tensor(
                [
                    loss_value,
                    hit1 / data.size(0),
                    hit5 / data.size(0),
                ],
                dtype=torch.float32,
                device=data.device,
            )
            dist.all_reduce(sync_tensor)
            sync_tensor /= world_size

            loss_value, acc1, acc5 = (
                sync_tensor.cpu().tolist()
            )

            lr_message = ", ".join(
                f"{group.get('group_name', index)}="
                f"{group['lr']:.3e}"
                for index, group in enumerate(
                    optimizer.param_groups
                )
            )

            epoch_message = ""
            if steps_per_epoch is not None:
                current_epoch = (
                    (step + 1) / steps_per_epoch
                )
                epoch_message = (
                    f"epoch: {current_epoch:.2f}/{args.epochs}  "
                )

            # BIE runtime monitoring: effective gates and warmup scale
            # (rank-local values, for diagnosing injection behavior during
            # training). BIE has no auxiliary loss, so only the gates are
            # printed.
            bie_message = ""
            if args.use_bie:
                raw_model = (
                    model.module
                    if hasattr(model, "module")
                    else model
                )
                gate_t_value = getattr(
                    raw_model, "_last_gate_t", None
                )
                gate_v_value = getattr(
                    raw_model, "_last_gate_v", None
                )
                gate_scale_value = getattr(
                    raw_model, "_last_gate_scale", None
                )
                bie_parts = []
                if gate_t_value is not None:
                    bie_parts.append(
                        "gate_t: "
                        f"{gate_t_value.detach().float().mean().item():.3f}"
                    )
                if gate_v_value is not None:
                    bie_parts.append(
                        "gate_v: "
                        f"{gate_v_value.detach().float().mean().item():.3f}"
                    )
                if gate_scale_value is not None:
                    bie_parts.append(
                        f"gate_scale: {float(gate_scale_value):.3f}"
                    )
                if bie_parts:
                    bie_message = "  " + "  ".join(bie_parts)

            # CVP Block runtime monitoring (rank-local values).
            cvp_message = ""
            layered_message = ""
            if args.use_cvp_block:
                cvp_raw_model = (
                    model.module
                    if hasattr(model, "module")
                    else model
                )
                cvp_stats = getattr(cvp_raw_model, "_last_cvp_stats", None)
                if cvp_stats:
                    def _v(k):
                        t = cvp_stats.get(k)
                        return f"{t.item():.4f}" if torch.is_tensor(t) else "n/a"

                    cvp_message = (
                        "  "
                        f"topk: {args.cvp_topk}  "
                        f"gamma: {_v('gamma')}  "
                        f"beta: {_v('beta')}  "
                        f"mot_rel: {_v('motion_reliability_mean')}  "
                        f"w_att: {_v('attention_weight_mean')}  "
                        f"w_mot: {_v('motion_weight_mean')}  "
                        f"lambda_sp: {_v('lambda_sp')}  "
                        f"corr_entropy: {_v('full_corr_attention_entropy')}  "
                        f"q_var: {_v('current_context_query_variance')}  "
                        f"res_norm: {_v('selected_residual_norm')}  "
                        f"res_ratio: {_v('residual_to_base_ratio')}  "
                        f"corr_cos: {_v('corr_cosine')}  "
                        f"prompt_norm: {_v('prompt_norm')}  "
                        f"outside_diff: {_v('outside_selected_max_diff')}"
                    )

            # Layered v3 runtime monitoring (mask gates / prompt gates).
            if args.use_mask_adapter or args.use_layered_prompts:
                layered = getattr(
                    (
                        model.module if hasattr(model, "module") else model
                    ),
                    "_last_layered_stats",
                    {},
                ) or {}
                if layered:
                    def _lv(key_suffix):
                        vals = [
                            v.item()
                            for k, v in layered.items()
                            if k.endswith(key_suffix)
                            and torch.is_tensor(v)
                        ]
                        return (
                            "/".join(f"{x:.3f}" for x in vals)
                            if vals
                            else "n/a"
                        )

                    layered_message = (
                        "  mask_gate: "
                        f"{_lv('mask_gate')}"
                        "  mask_ratio: "
                        f"{_lv('mask_res_ratio')}"
                        "  prompt_gate: "
                        f"{_lv('prompt_cond_gate')}"
                        "  vp_norm: "
                        f"{_lv('prompt_norm')}"
                    )

            tgl_message = ""
            if args.tgl_enable:
                tgl_raw_model = (
                    model.module if hasattr(model, "module") else model
                )
                tgl_stats = getattr(
                    getattr(tgl_raw_model, "visual", None),
                    "_last_tgl_stats",
                    {},
                ) or {}
                if tgl_stats:
                    def _flat(d, prefix=""):
                        for k, v in d.items():
                            key = f"{prefix}{k}"
                            if isinstance(v, dict):
                                yield from _flat(v, prefix=f"{key}.")
                            elif isinstance(v, (int, float)):
                                yield key, float(v)
                    flat_stats = dict(_flat(tgl_stats))

                    def _tg(keys):
                        vals = [
                            flat_stats[k]
                            for k in sorted(flat_stats)
                            if any(p in k for p in keys)
                        ]
                        return (
                            "/".join(f"{x:.4f}" for x in vals)
                            if vals
                            else "n/a"
                        )

                    tgl_message = (
                        "  tgl_scale: "
                        f"{_tg(['adapter_scale', 'tgl_gate_mean'])}  "
                        f"tgl_d/p: {_tg(['delta_over_patch'])}  "
                        f"tgl_cov: {_tg(['scatter_coverage'])}  "
                        f"vp_norm: {_tg(['video_prompt_norm'])}  "
                        f"tgl_p_norm: {_tg(['prompt_head_norm'])}"
                        + (
                            f"  base_gate: {_tg(['baseline_gate'])}"
                            if args.tgl_patch_pool_baseline
                            else ""
                        )
                    )

            print(
                f"batch_time: "
                f"{(batch_end - batch_start).total_seconds():.3f}  "
                f"data_time: "
                f"{(data_end - batch_start).total_seconds():.3f}  "
                f"ETA: "
                f"{(batch_end - training_start) / (step - resume_step + 1) * (args.num_steps - step - 1)}  |  "
                f"{epoch_message}"
                f"step: {step + 1}/{args.num_steps}  "
                f"lr: [{lr_message}]  "
                f"loss: {loss_value:.6f}  "
                f"acc1: {acc1 * 100:.2f}%  "
                f"acc5: {acc5 * 100:.2f}%"
                f"{bie_message}"
                f"{cvp_message}"
                f"{layered_message}"
                f"{tgl_message}"
            )

        if (
            (step + 1) % args.save_freq == 0
            and rank == 0
        ):
            checkpoint.save_checkpoint(
                model,
                optimizer,
                lr_sched,
                loss_scaler,
                step + 1,
                args,
            )

        if (step + 1) % args.eval_freq == 0:
            # Skip validation before --eval_start_epoch to save time (the model
            # is far from a useful checkpoint in the first epochs).
            if args.eval_start_epoch > 0.0 and steps_per_epoch > 0:
                cur_epoch = (step + 1) / steps_per_epoch
                if cur_epoch < args.eval_start_epoch:
                    continue
            print(
                "Start model evaluation at step",
                step + 1,
            )
            if ema_shadow is not None:
                # Swap in the EMA weights for evaluation and best-saving.
                with torch.no_grad():
                    for name, param in ema_trainable:
                        ema_backup[name] = param.detach().clone()
                        param.copy_(ema_shadow[name])
            model.eval()
            top1, top5 = evaluate(model, val_loader, tgl=args.tgl_enable)
            model.train()
            if rank == 0 and top1 > best_top1:
                best_top1 = top1
                best_step = step + 1
                if args.checkpoint_dir:
                    os.makedirs(args.checkpoint_dir, exist_ok=True)
                    best_path = os.path.join(
                        args.checkpoint_dir,
                        f"best_top1_{top1:.2f}_step{best_step}.pth",
                    )
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "top1": top1,
                            "top5": top5,
                            "step": best_step,
                        },
                        best_path,
                    )
                    print(f"[Best] saved {best_path}")
            if ema_shadow is not None:
                # Restore the training weights (EMA stays in shadow).
                with torch.no_grad():
                    for name, param in ema_trainable:
                        param.copy_(ema_backup[name])
            if rank == 0:
                print(
                    f"[Best top1 so far] {best_top1:.2f}% at step {best_step}"
                )

        batch_start = datetime.now()

    # ---- Training finished: finalize ----
    if rank == 0:
        print(
            f"[Training completed] total_steps={args.num_steps}, "
            f"total_time={(datetime.now() - training_start)}"
        )
        # Save the final checkpoint if the last step was not already saved.
        if args.num_steps % args.save_freq != 0:
            checkpoint.save_checkpoint(
                model,
                optimizer,
                lr_sched,
                loss_scaler,
                args.num_steps,
                args,
            )
            print(
                "[Training completed] saved final checkpoint at step",
                args.num_steps,
            )
    dist.barrier()


def _cvp_debug_visualize(
    args: argparse.Namespace,
    model: torch.nn.Module,
    data: torch.Tensor,
    step: int,
    rank: int,
) -> None:
    """First-batch CVP visualization: 2x3 panels per sample (rank 0 only)."""
    if rank != 0:
        return

    raw_model = model.module if hasattr(model, "module") else model
    stats = getattr(raw_model, "_last_cvp_stats", None) or {}
    dbg = stats.get("debug") or {}
    if not dbg:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    out_dir = args.cvp_debug_dir
    os.makedirs(out_dir, exist_ok=True)

    B, C, T, H, W = data.shape
    grid = int(round(H / 16))
    mid = T // 2

    for b in range(min(6, B)):
        frame = data[b, :, mid].permute(1, 2, 0).float().cpu().numpy()
        frame = (frame - frame.min()) / (frame.max() - frame.min() + 1e-6)

        sem = dbg["semantic_score"][b, mid].reshape(grid, grid).cpu().numpy()
        mot = dbg["motion_score"][b, mid].reshape(grid, grid).cpu().numpy()
        raw_diff = dbg["frame_diff_raw"][b, mid].reshape(grid, grid).cpu().numpy()
        score = dbg["score"][b, mid].reshape(grid, grid).cpu().numpy()
        sel = (
            dbg["selection_mask"][b, mid].reshape(grid, grid).float().cpu().numpy()
        )
        rnf = dbg["residual_norm"][b, mid].reshape(grid, grid).cpu().numpy()

        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        axes[0, 0].imshow(frame)
        axes[0, 0].set_title(f"mid frame (t={mid}, normalized)")
        axes[0, 1].imshow(sem, cmap="jet")
        axes[0, 1].set_title("1. block4 CLS->patch attention")
        axes[0, 2].imshow(raw_diff, cmap="jet")
        axes[0, 2].set_title("2. raw frame difference")
        axes[1, 0].imshow(mot, cmap="jet")
        axes[1, 0].set_title("3. centered frame difference")
        rel = dbg["motion_reliability_per_frame"][b].cpu().numpy()
        axes[1, 1].bar(range(T), rel)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].set_title("4. per-frame motion reliability c_t")
        axes[1, 2].imshow(score, cmap="jet")
        axes[1, 2].set_title("5. final adaptive score S")
        axes[2, 0].imshow(sel, cmap="gray", vmin=0, vmax=1)
        axes[2, 0].set_title(f"6. top-{args.cvp_topk} selection mask")
        axes[2, 1].imshow(rnf, cmap="jet")
        axes[2, 1].set_title("8. residual norm per patch")

        axes[2, 2].imshow(frame)
        if dbg.get("full_corr_attention") is not None:
            corr_att = dbg["full_corr_attention"][b, mid]  # [K, 196]
            sel_idx_b = dbg["selected_indices"][b, mid]    # [K]
            rn_sel = dbg["residual_norm"][b, mid].gather(0, sel_idx_b)
            strongest_slot = int(rn_sel.argmax().item())
            strongest = int(sel_idx_b[strongest_slot].item())
            row = corr_att[strongest_slot].reshape(grid, grid).cpu().numpy()
            axes[2, 2].imshow(row, cmap="jet", alpha=0.6)
            axes[2, 2].set_title(
                f"7. correspondence attn of selected patch "
                f"{divmod(strongest, grid)} (full frame)"
            )
        else:
            flat = dbg["residual_norm"][b, mid]
            top_patch = int(flat.argmax().item())
            cy, cx = divmod(top_patch, grid)
            half = args.cvp_window_size / 2.0
            axes[2, 2].add_patch(
                Rectangle(
                    (cx * 16 - half * 16, cy * 16 - half * 16),
                    args.cvp_window_size * 16,
                    args.cvp_window_size * 16,
                    fill=False, color="yellow", linewidth=1.5,
                )
            )
            axes[2, 2].set_title("7. local window (local scope)")
        for ax_row in axes:
            for ax in ax_row:
                ax.axis("off")
        plt.tight_layout()
        plt.savefig(
            os.path.join(out_dir, f"step{step + 1}_sample{b}.png"), dpi=120
        )
        plt.close(fig)
    print(
        f"[cvp] saved {min(6, B)} debug panels to {out_dir}"
    )


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    frame_shuffle: bool = False,
    tgl: bool = False,
) -> Tuple[float, float]:
    total = 0
    hit1 = 0
    hit5 = 0
    evaluation_start = datetime.now()

    def _collect_tensors(x):
        """Recursively collect all tensors from a possibly nested list/tuple."""
        if isinstance(x, torch.Tensor):
            return [x]
        if isinstance(x, (list, tuple)):
            return sum((_collect_tensors(item) for item in x), [])
        raise TypeError(f"Unexpected data type in loader: {type(x)}")

    for batch in loader:
        tgl_pack = None
        if tgl:
            # TGL validation: (frames, label, tracks, vis, birth). Tracks are
            # offline caches; nothing here ever runs a tracker.
            data, labels, tracks, visibility, birth_frame = batch
            tgl_pack = (tracks, visibility, birth_frame)
        else:
            data, labels = batch
        data_tensors = _collect_tensors(data)
        if len(data_tensors) == 0:
            raise RuntimeError("No tensor found in validation data.")
        # Each sample returns one tensor shaped [V, C, T, H, W] (V views) or
        # [C, T, H, W] (single view). default_collate prepends a batch dim,
        # yielding [B, V, C, T, H, W] or [B, C, T, H, W]. Flatten all videos
        # and views into a single [B*V, C, T, H, W] tensor for one forward.
        flat_views = []
        view_counts = []
        for t in data_tensors:
            if t.ndim == 6:  # [B, V, C, T, H, W]
                b, v, c, tt, h, w = t.shape
                flat_views.append(t.reshape(b * v, c, tt, h, w))
                view_counts.append(v)
            elif t.ndim == 5:  # [B, C, T, H, W] -> single view per video
                b, c, tt, h, w = t.shape
                flat_views.append(t.reshape(b, c, tt, h, w))
                view_counts.append(1)
            else:
                raise RuntimeError(
                    f"Unexpected validation data shape: {tuple(t.shape)}"
                )
        if len(set(view_counts)) != 1:
            raise RuntimeError(
                "Inconsistent view counts across validation tensors: "
                f"{view_counts}"
            )
        num_views = view_counts[0]
        data = torch.cat(flat_views, dim=0).cuda(non_blocking=True)
        # [B*V, C, T, H, W]

        # Temporal-sensitive evaluation: destroy frame order per view. If
        # accuracy barely drops, the model relies on single-frame semantics
        # rather than temporal structure.
        if frame_shuffle:
            v_total, c, t, h, w = data.shape
            perms = torch.stack(
                [torch.randperm(t, device=data.device) for _ in range(v_total)]
            )
            data = data[
                torch.arange(v_total, device=data.device)[:, None], :, perms, :, :
            ]

        labels_tensors = _collect_tensors(labels)
        labels = labels_tensors[0].view(-1).cuda(non_blocking=True)  # [B]

        if tgl_pack is not None:
            tracks, visibility, birth_frame = tgl_pack
            # Expand per-video tracks across the flattened [B*V] views.
            v = num_views
            tracks = (
                tracks.unsqueeze(1)
                .expand(-1, v, -1, -1, -1)
                .reshape(tracks.shape[0] * v, *tracks.shape[1:])
                .cuda(non_blocking=True)
            )
            visibility = (
                visibility.unsqueeze(1)
                .expand(-1, v, -1, -1)
                .reshape(visibility.shape[0] * v, *visibility.shape[1:])
                .cuda(non_blocking=True)
            )
            birth_frame = (
                birth_frame.unsqueeze(1)
                .expand(-1, v, -1)
                .reshape(birth_frame.shape[0] * v, birth_frame.shape[1])
                .cuda(non_blocking=True)
            )
            logits = model(
                data,
                tracks=tracks,
                visibility=visibility,
                birth_frame=birth_frame,
            )  # [B*V, num_classes]
        else:
            logits = model(data)  # [B*V, num_classes]
        num_classes = logits.size(-1)
        scores = logits.softmax(dim=-1).view(
            -1, num_views, num_classes
        ).mean(dim=1)  # [B, num_classes]

        batch_size = labels.numel()
        total += batch_size
        hit1 += (
            scores.topk(1, dim=1)[1] == labels.view(-1, 1)
        ).sum().item()
        hit5 += (
            scores.topk(5, dim=1)[1] == labels.view(-1, 1)
        ).sum().item()

        if total % 20 == 0:
            print(
                f"[Evaluation] num_samples: {total}  "
                f"ETA: "
                f"{(datetime.now() - evaluation_start) / total * (len(loader.dataset) - total)}  "
                f"cumulative_acc1: {hit1 / total * 100.0:.2f}%  "
                f"cumulative_acc5: {hit5 / total * 100.0:.2f}%"
            )

    sync_tensor = torch.tensor(
        [total, hit1, hit5],
        dtype=torch.long,
        device="cuda",
    )
    dist.all_reduce(sync_tensor)
    total, hit1, hit5 = (
        sync_tensor.cpu().tolist()
    )

    top1 = hit1 / total * 100.0
    top5 = hit5 / total * 100.0

    print(
        "Accuracy on validation set: "
        f"top1={top1:.2f}%, "
        f"top5={top5:.2f}%"
    )
    return top1, top5


if __name__ == "__main__":
    main()
