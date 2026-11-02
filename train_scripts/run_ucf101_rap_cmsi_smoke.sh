#!/bin/bash
set -e

# ----------------------------------------------------------------
# UCF101 split1 - RAP + CMSI Phase A 冒烟训练（2 epochs）
#
# 目的: 在跑 60ep 全量前，用 2 epochs 验证:
#   - 训练循环集成（cmsi_step 透传、参数分组、监控打印）无异常
#   - gate_scale 从 0 -> 1 的 ramp 曲线
#   - align_loss 数值量级（对比损失 + lambda=0.005 缩放后应稳定）
#   - 原型文本初始化后 proto_count 增长
# 观察日志中:  "gate_scale" 每 1000 步打印（print_freq 默认 1000）,
#   2 epochs 只有 ~1192 步，可把 print_freq 调小或直接看 epoch 末 eval。
# ----------------------------------------------------------------

EXP_DIR=training/checkpoints/ucf101_rap_bs16_cmsi_smoke
LOG_DIR=training/logs

mkdir -p "${EXP_DIR}" "${LOG_DIR}"

cd /hy-tmp/Vita-CLIP

python -u -m torch.distributed.run --nproc_per_node 1 \
  ./training/train_OPENAI_ALIGNED.py \
    --checkpoint_dir "${EXP_DIR}" \
    --epochs 2 \
    --warmup_epochs 0 \
    --eval_every_epochs 1 \
    --save_every_epochs 1 \
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
    --cmsi_align_lambda 0.005 \
    --cmsi_proto_momentum 0.99 \
    --cmsi_max_gate 0.3 \
    --cmsi_lr 1.0e-4 \
    --cmsi_gate_warmup_epochs 1 \
  2>&1 | tee "${LOG_DIR}/ucf101_rap_bs16_cmsi_smoke-$(date +"%Y%m%d_%H%M%S").log"
