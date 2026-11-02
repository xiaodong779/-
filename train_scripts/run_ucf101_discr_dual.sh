#!/bin/bash
set -e

# ----------------------------------------------------------------
# UCF101 split1 - Discriminative Dual (conv3d + full Mamba)
# top_k=12 default; gates fusion sigma init 0.5; all-fp32 temporal
#
# bs=16: 实测 forward+backward 峰值 15.6GB，适配共享 GPU。
# 学习率按 linear scaling (16/24=2/3) 从 baseline bs24 缩放。
#
# 为什么用 full 而非 union：
#   full (分解两阶段扫描: 逐位置时间扫描 + 逐帧空间扫描，跨位置轨迹
#        流动) 冒烟实测 fwd+bwd 25.79ms/step (bs=2) —— 已可用
#   union (仅逐位置时间扫描，无跨位置流动) 5.71ms/step (bs=2)
#   union 已完成一轮训练 (best 92.86%，3x3-fix 后 eval 93.02%)，
#   本轮跑 full 验证跨位置信息流动的增益。
#   如需降载/消融，改 --dr_mamba_scope union。
# ----------------------------------------------------------------

EXP_DIR=training/checkpoints/ucf101_discr_dual_full_bs16
LOG_DIR=training/logs

mkdir -p "${EXP_DIR}" "${LOG_DIR}"

cd /hy-tmp/Vita-CLIP

python -u -m torch.distributed.run --nproc_per_node 1 \
  ./training/train_OPENAI_ALIGNED.py \
    --checkpoint_dir "${EXP_DIR}" \
    --auto_resume \
    --epochs 50 \
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
    --dr_arch dual \
    --dr_top_k 12 \
    --dr_mamba_scope full \
    --dr_layers 3,7,11 \
  2>&1 | tee "${LOG_DIR}/ucf101_discr_dual_full_bs16-$(date +"%Y%m%d_%H%M%S").log"
