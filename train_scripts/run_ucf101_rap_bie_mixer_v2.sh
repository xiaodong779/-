#!/bin/bash
set -e

# ----------------------------------------------------------------
# UCF101 split1 - RAP + BIE(v2 cross-attention + 一致性 + 阈值) + Mixer
#
# BIE v2 相比 v1 的完整改进:
#   1. 视频侧: 从 p 加权质心插值(qhat=p@text) 改为 可学习投影 cross-attention
#      文本采集（Wq/Wk/Wv + dropout + gate_v 门控）。
#   2. 一致性损失: 约束增强后特征(视频侧 & 文本侧)与原始特征距离 <= max_dist,
#      防 gate 作弊。
#   3. 文本增强阈值 enh_thr: 只有 soft 概率 p > enh_thr 的高置信度类才被增强,
#      低置信度类保持原型（不破坏类别原型）。
#   4. cross-attention dropout + logit_scale 复用防御。
#
# 目的: 验证 top1 是否突破旧 best 94.71% 到 95+。
# ----------------------------------------------------------------

EXP_DIR=training/checkpoints/ucf101_rap_bs16_bie_mixer_v2
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
    --batch_size 20 \
    --grad_accum 4 \
    --train_rrc_scale "0.5,1.0" \
    --num_spatial_views 3 \
    --num_temporal_views 3 \
    --label_smoothing 0.1 \
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
    --dr_layers 3,5,7,9 \
    --use_bie \
    --bie_p_temperature 10.0 \
    --bie_max_gate 0.5 \
    --bie_gate_warmup_epochs 10 \
    --bie_lr 1.0e-4 \
    --bie_transform_bottleneck 128 \
    --bie_gate_bottleneck 128 \
    --bie_enh_thr 0.01 \
    --bie_max_dist 0.05 \
    --bie_max_dist_t 0.05 \
    --use_aggregation_mixer \
    --aggregation_mixer_kernel 3 \
  2>&1 | tee "${LOG_DIR}/ucf101_rap_bs16_bie_mixer_v2-$(date +"%Y%m%d_%H%M%S").log"
