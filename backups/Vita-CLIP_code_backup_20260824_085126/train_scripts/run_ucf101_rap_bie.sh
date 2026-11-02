#!/bin/bash
set -e

# ----------------------------------------------------------------
# UCF101 split1 - RAP + BIE（Bidirectional Interpolation Enhancement）
#
# 背景（CMSI v4 诊断）:
#   - v3/v4 的 align_loss 死亡 + gate 学负 => "文本/原型注入"与分类目标
#     冲突，模块被网络反向压制。根因是旧 CMSI 用"向量注入 + prototype
#     对齐损失"，其修正方向与 CE 分类目标不一致。
#
# BIE 关键设计（对应 BIE.py，与 MIP-CLIP 的 DFR 不同）:
#   1) 双向"软插值"而非"硬加"注入:
#       文本增强: Q_c' = Q_c + g_t * p_c * (V - Q_c)   （只动"可能的类"）
#       视频增强: V'  = V + g_v * (p @ Q - V)          （门控插值，有界）
#      任何增强结果都落在 [原特征, 对方特征] 线段上，不会被拽出模态空间。
#   2) p = softmax(p_temperature * V@Q^T)（专用温度 10，避免 logit_scale
#      ~100 饱和成 one-hot），双向均 detach（MIP-CLIP 同款技巧，防自指循环）。
#   3) 恒等起点: g_t/g_v 零初始化 + gate warmup，step 0 输出 == 基线。
#   4) 无任何辅助损失，全程 CE(label_smoothing=0.1)。
#   5) 方案 B（默认开启）: 两个方向增强后各过一个残差式变换 MLP
#      （D->128->D，输出层零初始化 -> 初始恒等），提供额外表达力。
#
# 对照: RAP baseline 93.84% / CMSI v3 best 94.21% / v4（应达 94+）
# 目标: >= 95.0% Top-1
# 诊断: 日志中 gate_t / gate_v 应从 0 醒来；若学成负值或恒 0，
#       说明注入方向仍与分类冲突，检查 max_gate / 温度 / LR。
# ----------------------------------------------------------------

EXP_DIR=training/checkpoints/ucf101_rap_bs16_bie
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
  2>&1 | tee "${LOG_DIR}/ucf101_rap_bs16_bie-$(date +"%Y%m%d_%H%M%S").log"
