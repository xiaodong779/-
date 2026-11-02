#!/bin/bash
set -e

# ----------------------------------------------------------------
# UCF101 split1 - RAP + SCR (Semantic Class-Relation Refinement) + Mixer
#
# SCR: 把每个类别的文本锚点，推离它最容易混淆的 top-k 个邻居，
#      从而拉开相似动作类（如篮球 vs 投篮），提升 top1。
#      - 纯静态图（只由文本 Q@Q^T 决定，不依赖输入视频）
#      - identity start（theta 零初始化，第一步 = baseline）
#      - 与 BIE 互斥（两者都拥有分类器权重细化阶段）
#
# 目的: 验证 SCR 是否能突破旧 best 94.71% 到 95+。
# ----------------------------------------------------------------

EXP_DIR=training/checkpoints/ucf101_rap_bs16_scr_mixer_transformer
LOG_DIR=training/logs

mkdir -p "${EXP_DIR}" "${LOG_DIR}"

cd /hy-tmp/Vita-CLIP

python -u -m torch.distributed.run --nproc_per_node 1 \
  ./training/train_OPENAI_ALIGNED.py \
    --checkpoint_dir "${EXP_DIR}" \
    --epochs 50 \
    --warmup_epochs 5 \
    --eval_every_epochs 5 \
    --eval_start_epoch 20 \
    --save_every_epochs 5 \
    --batch_size 20 \
    --grad_accum 4 \
    --train_rrc_scale "0.5,1.0" \
    --num_spatial_views 3 \
    --num_temporal_views 3 \
    --label_smoothing 0.1 \
    --min_lr_ratio 0.01 \
    --lr 1.0e-4 \
    --prompt_lr 1.0e-3 \
    --adapter_lr 4.0e-4 \
    --scalar_lr 2.0e-4 \
    --time_embed_lr 2.0e-4 \
    --use_model_ema \
    --ema_decay 0.9995 \
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
    --dr_asa_time_offset_max 3.0 \
    --dr_asa_space_offset_max 4.0 \
    --dr_asa_window_size 5 \
    --dr_layers 3,5,7,9 \
    --use_scr \
    --scr_top_k 5 \
    --scr_max_scale 0.5 \
    --use_aggregation_mixer \
    --aggregation_mixer_kernel 3 \
    --aggregation_mixer_type transformer \
    --aggregation_mixer_heads 8 \
  2>&1 | tee "${LOG_DIR}/ucf101_rap_bs16_scr_mixer-$(date +"%Y%m%d_%H%M%S").log"
