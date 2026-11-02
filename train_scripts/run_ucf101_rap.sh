#!/bin/bash
set -e

# ----------------------------------------------------------------
# UCF101 split1 - Discriminative RAP (conv3d full + sparse ASA)
# Reference: "Efficient Text-Video Retrieval with Sparse-and-Correlated
#   Adapter" (RAP, arXiv:2405.19465).
#
# top_k=12 default; gates fusion sigma init 0.5; all-fp32 temporal.
# ASA branch: RAP asynchronous self-attention over the SELECTED top-k
#   tokens only, per-query dynamic temporal/spatial offsets predicted by
#   a zero-init offset_net (tanh * max -> identity sampling at start).
#   Unselected tokens stay zero in this branch (sparse semantics).
#   LoRM not yet introduced.
#
# bs=16: 与 dual 相同的显存档位（卷积全量 + 稀疏注意力，预期峰值
#   明显低于 full Mamba 的 15.6GB）。
# 学习率按 linear scaling (16/24=2/3) 从 baseline bs24 缩放。
#
# 对比基线: union 92.86% / 3x3-fix eval 93.02%（dual union），
#   dual full 为本轮同配置对照。
#
# CMSI v2: 关闭 V->T 注入（use_v2t=False，推理 logits 与 baseline 结构一致），
#   仅保留 T->V 文本修正并加 max_gate=0.5 上限；align_lambda 降到 0.02、
#   proto_momentum 提到 0.995 防 EMA 漂移。时序模块保持 3 层 (3,7,11)，
#   epochs 60 + warmup 5。
# ----------------------------------------------------------------

EXP_DIR=training/checkpoints/ucf101_rap_bs16_cmsi_v2
LOG_DIR=training/logs

mkdir -p "${EXP_DIR}" "${LOG_DIR}"

cd /hy-tmp/Vita-CLIP

python -u -m torch.distributed.run --nproc_per_node 1 \
  ./training/train_OPENAI_ALIGNED.py \
    --checkpoint_dir "${EXP_DIR}" \
    --auto_resume \
    --epochs 60 \
    --warmup_epochs 5 \
    --eval_every_epochs 5 \
    --save_every_epochs 5 \
    --batch_size 16 \
    --lr 1.0e-4 \
    --prompt_lr 1.0e-3 \
    --adapter_lr 4.0e-4 \
    --scalar_lr 2.0e-4 \
    --time_embed_lr 2.0e-4 \
    --backbone_path /hy-tmp/weights/ViT-B-16.pt \
    --train_list_path /hy-tmp/UCF101/annotations/ucfTrainTestlist/trainlist01.txt \
    --val_list_path /hy-tmp/UCF101/annotations/ucfTrainTestlist/testlist01.txt \
    --data_root /hy-tmp/UCF101/rawframes \
    --num_frames 8 \
    --num_workers 8 \
    --use_discriminative_module \
    --dr_arch rap \
    --dr_top_k 12 \
    --dr_asa_heads 8 \
    --dr_asa_time_offset_max 1.0 \
    --dr_asa_space_offset_max 4.0 \
    --dr_asa_window_size 5 \
    --dr_layers 3,7,11 \
    --use_cmsi \
    --cmsi_align_lambda 0.02 \
    --cmsi_proto_momentum 0.995 \
    --cmsi_max_gate 0.5 \
  2>&1 | tee "${LOG_DIR}/ucf101_rap_bs16_cmsi_v2-$(date +"%Y%m%d_%H%M%S").log"
