#!/bin/bash
set -e

# ----------------------------------------------------------------
# UCF101 split1 - TRUE pure CLIP baseline (bs=24)
# Purpose: measure the raw Vita-CLIP ceiling WITHOUT any of our
# innovations, to quantify their contributions.
#
# What is disabled vs the old "baseline":
#   - --no_text_prompt_learning : no learnable text prompts
#                                (also forces use_learnable_text_branch=False)
#   - --no_learnable_text_branch: no learnable prompt branch
#   - --no_text_mstff           : no Text_MSTFF multi-scale enhancement
#   - --use_discriminative_module / --dr_* : no discriminative RWKV module
#
# What remains:
#   - Frozen OpenAI CLIP ViT-B/16 visual encoder (8 frames, mean pooling)
#   - Frozen OpenAI CLIP text encoder with the plain handcrafted template
#     "a video of a person {}" (fixed_text_branch, the only enabled branch)
#   - Trainable parameters: logit_scale only
#
# Control config: identical to run_ucf101_discr_rwkv_scorer.sh
# (same bs=24 / lr / txt lists / data) EXCEPT all innovation flags removed.
#
# Reference points (old "baseline" had prompt + text_mstff baked in):
#   - old baseline (prompt + MSTFF, no discr)   : top1 = 90.59%
#   - discr RWKV + scorer (txt list)            : top1 = 91.81%
#   - true pure CLIP (this script)              : TBD
# ----------------------------------------------------------------

EXP_DIR=training/checkpoints/ucf101_clip_pure_baseline_bs24
LOG_DIR=training/logs

mkdir -p "${EXP_DIR}" "${LOG_DIR}"

cd /hy-tmp/Vita-CLIP

nohup /hy-tmp/envs/Vita/bin/python -u -m torch.distributed.run --nproc_per_node 1 \
  ./training/train_OPENAI_ALIGNED.py \
    --checkpoint_dir "${EXP_DIR}" \
    --auto_resume \
    --epochs 50 \
    --eval_every_epochs 5 \
    --save_every_epochs 5 \
    --batch_size 24 \
    --lr 1.5e-4 \
    --prompt_lr 1.5e-3 \
    --adapter_lr 6e-4 \
    --scalar_lr 3e-4 \
    --time_embed_lr 3e-4 \
    --backbone_path /hy-tmp/weights/ViT-B-16.pt \
    --train_list_path /hy-tmp/UCF101/annotations/ucfTrainTestlist/trainlist01.txt \
    --val_list_path /hy-tmp/UCF101/annotations/ucfTrainTestlist/testlist01.txt \
    --data_root /hy-tmp/UCF101/rawframes \
    --num_frames 8 \
    --num_workers 8 \
    --no_text_prompt_learning \
    --no_learnable_text_branch \
    --no_text_mstff \
  > "${LOG_DIR}/ucf101_clip_pure_baseline.log" 2>&1 &

echo "PID: $!"
