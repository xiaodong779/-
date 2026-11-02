#!/bin/bash
set -e

# ----------------------------------------------------------------
# UCF101 split1 - RAP + BIE(v3 DFR-style) + Mixer
#
# BIE v3 (对齐 MIP-CLIP DFR) 的完整设计:
#   1. α: 独立可学习分类头 FC(LN(video)) 得到视频对每类的置信度分数
#      (非 softmax, 可正负), 由辅助 CE loss 训练 (alpha_aux_lambda)。
#   2. 文本侧: 用非负 α 加权, 只有高置信度正类被增强(保持原型)。
#   3. 视频侧: cross-attention 从文本采集 + sigmoid gate(非负) 叠加
#      video_enh = video + gate_v * delta, 只增强不削弱。
#   4. 一致性损失(视频+文本侧), cross-attention dropout, logit_scale 复用。
#
# 目的: 验证 top1 是否突破旧 best 94.71% 到 95+。
# ----------------------------------------------------------------

EXP_DIR=training/checkpoints/ucf101_rap_bs16_bie_mixer_v4_tadaconv
LOG_DIR=training/logs

mkdir -p "${EXP_DIR}" "${LOG_DIR}"

cd /hy-tmp/Vita-CLIP

python -u -m torch.distributed.run --nproc_per_node 1 \
  ./training/train_OPENAI_ALIGNED.py \
    --checkpoint_dir "${EXP_DIR}" \
    --auto_resume \
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
    --bie_alpha_aux_lambda 0.1 \
    --use_aggregation_mixer \
    --aggregation_mixer_kernel 3 \
    --aggregation_mixer_type transformer \
    --aggregation_mixer_heads 8 \
  2>&1 | tee "${LOG_DIR}/ucf101_rap_bs16_bie_mixer_v3-$(date +"%Y%m%d_%H%M%S").log"
