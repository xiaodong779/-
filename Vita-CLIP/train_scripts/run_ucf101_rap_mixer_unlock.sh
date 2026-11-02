#!/bin/bash
set -e

# ----------------------------------------------------------------
# UCF101 split1 - RAP + Transformer Mixer（解锁版完整训练）
#
# 配置: RAP（residual_scale 可学习0.1 + enhance_scale 非零0.1 +
#       scoring 非零 + 卷积分支 depthwise2 非零）
#       + transformer mixer（out_proj 非零 + pos_embed 逐帧）
#       + 正确 LR 退火 + 水平翻转 + EMA
#       去掉 SCR、去掉 BIE
#
# 目的: 验证"解开 RAP/mixer 所有锁死的门控"后能否涨点。
# ----------------------------------------------------------------

EXP_DIR=training/checkpoints/ucf101_rap_bs16_mixer_unlock
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
    --dr_residual_scale_init 0.1 \
    --use_aggregation_mixer \
    --aggregation_mixer_kernel 3 \
    --aggregation_mixer_type transformer \
    --aggregation_mixer_heads 8 \
  2>&1 | tee "${LOG_DIR}/ucf101_rap_mixer_unlock-$(date +"%Y%m%d_%H%M%S").log"
