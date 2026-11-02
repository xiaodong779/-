#!/bin/bash
set -e

# ----------------------------------------------------------------
# UCF101 split1 - RAP + CMSI Phase A (掉点修复)
#
# Phase A 相对 v2 的关键改动（根因见 docs/cmsi_improvement_report.md）:
#   1) align_loss: MSE(P, Q') -> 判别式对比损失（InfoNCE 形式）
#      - 正样本 P_i（EMA 原型, detached），负样本为其他类原始文本方向 Q_j
#      - 保留类间 margin，避免 MSE 各向同性压扁 margin（top5 99% / top1 上不去）
#   2) 原型质量: proto_momentum 0.995 -> 0.99（滞后窗口 ~33 -> ~17 epoch）；
#      原型用文本方向 Q 做种子初始化（proto_text_init，训练首步生效）
#   3) gate 渐进唤醒: cmsi_gate_warmup_epochs 10（前 10 epoch 注入关闭，
#      行为与 baseline 完全一致，随后线性 ramp）
#   4) 超参: align_lambda 0.02 -> 0.005, max_gate 0.5 -> 0.3
#   5) 优化器: CMSI 低秩网络独立 cmsi 组 lr=1e-4（原误入 adapter 组 4e-4），
#      gate logit 归 scalar 组 2e-4
#   6) 训练日志新增 align_loss / gate_t / gate_scale 监控
#
# 对照: RAP baseline 93.84%（ucf101_rap_bs16-20260816_234611.log）
#       CMSI v2   epoch40 93.74%（ucf101_rap_bs16_cmsi_v2-20260818_101045.log）
# 目标: >= 93.9% 且 epoch 35 后不掉点。
# ----------------------------------------------------------------

EXP_DIR=training/checkpoints/ucf101_rap_bs16_cmsi_v3
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
    --dr_layers 3,5,7,9 \
    --use_cmsi \
    --cmsi_align_lambda 0.005 \
    --cmsi_proto_momentum 0.99 \
    --cmsi_max_gate 0.3 \
    --cmsi_lr 1.0e-4 \
    --cmsi_gate_warmup_epochs 10 \
  2>&1 | tee "${LOG_DIR}/ucf101_rap_bs16_cmsi_v3-$(date +"%Y%m%d_%H%M%S").log"
