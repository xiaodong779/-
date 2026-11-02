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
- frequency/Text_MSTFF LR: 2e-4
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
    because parameters such as ``visual.frequency_adapters.3.raw_c1`` and
    ``text_mstff.fusion_logits`` contain an adapter name but should use the
    conservative scalar learning rate.
    """
    lower_name = name.lower()

    scalar_markers = (
        "logit_scale",
        "text_fusion_logits",
        "fusion_logits",
        "text_mstff_gate",
        "residual_scale",
        "patch_mix_logit",
        "raw_c1",
        "raw_c2",
    )
    if any(marker in lower_name for marker in scalar_markers):
        return "scalar"

    if "time_embed" in lower_name:
        return "time_embed"

    if "prompt_learner" in lower_name:
        return "prompt"

    if (
        "frequency_adapter" in lower_name
        or "text_mstff" in lower_name
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
        help="learning rate for Frequency Adapters and Text_MSTFF",
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
    # Frequency Adapter
    # ------------------------------------------------------------------
    parser.add_argument(
        "--use_frequency_adapter",
        action="store_true",
        dest="use_frequency_adapter",
    )
    parser.add_argument(
        "--no_frequency_adapter",
        action="store_false",
        dest="use_frequency_adapter",
    )
    parser.add_argument(
        "--frequency_layers",
        type=str,
        default="3,7,11",
    )
    parser.add_argument(
        "--share_frequency_adapter",
        action="store_true",
        dest="share_frequency_adapter",
    )
    parser.add_argument(
        "--frequency_bottleneck",
        type=int,
        default=192,
    )
    parser.add_argument(
        "--frequency_residual_scale_init",
        type=float,
        default=0.01,
        help="initial raw residual gate for every Frequency Adapter",
    )
    parser.set_defaults(
        use_frequency_adapter=True,
        share_frequency_adapter=False,
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

    frequency_layers = tuple(
        int(value)
        for value in args.frequency_layers.split(",")
        if value.strip()
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
        use_frequency_adapter=args.use_frequency_adapter,
        frequency_layers=(
            frequency_layers
            if args.use_frequency_adapter
            else ()
        ),
        share_frequency_adapter=(
            args.share_frequency_adapter
        ),
        frequency_bottleneck=args.frequency_bottleneck,
        frequency_residual_scale_init=(
            args.frequency_residual_scale_init
        ),
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
    lr_sched = build_warmup_cosine_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        min_lr_ratio=args.min_lr_ratio,
    )

    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[cuda_device_id],
        output_device=cuda_device_id,
        find_unused_parameters=False,
    )

    loss_scaler = torch.cuda.amp.GradScaler(
        enabled=args.fp16,
        init_scale=8192.0,
        growth_interval=2000,
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

    val_loader = video_dataset.create_val_loader(args)

    if args.eval_only:
        print("Running in eval_only mode.")
        model.eval()
        evaluate(model, val_loader)
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

    for step, (data, labels) in enumerate(
        train_loader,
        resume_step,
    ):
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

            with torch.cuda.amp.autocast(
                enabled=args.fp16
            ):
                logits = model(data_slice)
                loss = criterion(
                    logits,
                    labels_slice,
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
            )

            loss_scaler.scale(
                loss / args.batch_split
            ).backward()

        if args.grad_clip_norm > 0:
            loss_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
                max_norm=args.grad_clip_norm,
            )

        # GradScaler may skip optimizer.step() when FP16 gradients contain
        # Inf/NaN. Advance the scheduler only after a real parameter update.
        scale_before = loss_scaler.get_scale()
        loss_scaler.step(optimizer)
        loss_scaler.update()
        scale_after = loss_scaler.get_scale()

        optimizer_step_skipped = (
            args.fp16
            and scale_after < scale_before
        )

        if optimizer_step_skipped:
            if step % args.print_freq == 0:
                print(
                    "[AMP] optimizer step skipped because non-finite "
                    "gradients were detected; scheduler was not advanced."
                )
        else:
            clamp_logit_scale(model)
            lr_sched.step()

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
            )

        # Save first so an evaluation error does not discard training progress.
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

        if (step + 1) % args.save_freq == 0:
            dist.barrier()

        if (step + 1) % args.eval_freq == 0:
            print(
                "Start model evaluation at step",
                step + 1,
            )
            model.eval()
            evaluate(model, val_loader)
            model.train()

        batch_start = datetime.now()


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
) -> Tuple[float, float]:
    total = 0
    hit1 = 0
    hit5 = 0
    evaluation_start = datetime.now()

    for data, labels in loader:
        # Validation loader returns a list of video views.
        # [V] x [B,C,T,H,W] -> [B,V,C,T,H,W]
        if isinstance(data, (list, tuple)):
            if len(data) == 0:
                raise ValueError("Validation data contains no views.")

            if not all(torch.is_tensor(view) for view in data):
                raise TypeError(
                    "Every validation view must be a torch.Tensor."
                )

            data = torch.stack(list(data), dim=1)

        if not torch.is_tensor(data):
            raise TypeError(
                f"Unsupported validation data type: {type(data).__name__}"
            )

        data = data.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True).long().reshape(-1)

        if data.size(0) != 1:
            raise ValueError(
                "The current multi-view evaluator expects validation "
                f"batch size 1, got {data.size(0)}."
            )

        if data.ndim == 6:
            # Remove outer batch dimension; first dimension becomes views.
            data = data[0]

        logits = model(data)
        scores = logits.softmax(dim=-1).mean(dim=0)

        total += 1
        hit1 += (
            scores.topk(1)[1] == labels
        ).sum().item()
        hit5 += (
            scores.topk(5)[1] == labels
        ).sum().item()

        if total % 20 == 0:
            print(
                f"[Evaluation] num_samples: {total}  "
                f"ETA: "
                f"{(datetime.now() - evaluation_start) / total * (len(loader) - total)}  "
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
