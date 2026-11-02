#!/bin/bash
set -e

# ----------------------------------------------------------------
# UCF101 split1 - 冒烟测试（smoke test）
#
# 配置: RAP（含新增强: residual_scale 可学习0.1 + enhance_scale 非零0.1）
#       + transformer mixer（pos_embed 逐帧修复）
#       + 正确 LR 退火 + 水平翻转 + EMA
#       去掉 SCR、去掉 BIE
#
# 目的: 快速验证 RAP/mixer 三个增强不炸、loss 正常下降、
#       residual_scale / enhance_scale 不再死锁（能从 0.1 正常演化）。
#       跑 1500 步（约 3 个 epoch），约 40 分钟。
# ----------------------------------------------------------------

EXP_DIR=training/checkpoints/ucf101_rap_smoke_enhanced
LOG_DIR=training/logs

mkdir -p "${EXP_DIR}" "${LOG_DIR}"

cd /hy-tmp/Vita-CLIP

python -u -m torch.distributed.run --nproc_per_node 1 \
  ./training/train_OPENAI_ALIGNED.py \
    --checkpoint_dir "${EXP_DIR}" \
    --num_steps 1500 \
    --warmup_ratio 0.1 \
    --eval_every_epochs 999 \
    --eval_start_epoch 999 \
    --save_every_epochs 999 \
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
  2>&1 | tee "${LOG_DIR}/ucf101_rap_smoke_enhanced-$(date +"%Y%m%d_%H%M%S").log"
