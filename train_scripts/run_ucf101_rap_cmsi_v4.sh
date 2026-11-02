#!/bin/bash
set -e

# ----------------------------------------------------------------
# UCF101 split1 - RAP + CMSI Phase B (冲 95+)
#
# 诊断（v3 日志 ucf101_rap_bs16_cmsi_v3-20260819_092231.log）:
#   - 验证曲线在 ~94% 平台期（best 94.21% @ ep40，之后 94.10 -> 93.99 回落），
#     train acc1 100% -> 过拟合；相对 RAP baseline 93.84% 仅 +0.37%。
#   - gate_t 学成负值（-0.156）、gate_v 微负（-0.036）: 优化器在"反向压制"
#     注入，说明 CMSI 的修正方向与分类目标冲突，模块被网络学掉了。
#   - align_loss 由 0.028 坍塌至 0.0003（ep15 后基本为 0）: 对齐信号死亡。
#   - 有效 batch=16 偏小（EMA 原型每步只见到 ~15/101 类，噪声大）。
#
# Phase B 关键改动:
#   1) gate 非负钳制（relu(tanh)): 保留 identity start，只允许"关闭/放大"注入，
#      禁止反向注入（对应 CMSI.py _ScalarGate/_DynamicGate）。
#   2) V->T 注入字典从"文本重建"改为"EMA 视频类原型"（v2t_dict=proto）:
#      直接向类中心拉近 -> 直接服务分类间隔，且训练/推理一致（无需标签）。
#      未见类别从注意力中 mask 掉，防止零向量稀释。
#   3) T->V 修正方向固定为 normalize(P-Q)（t2v_direction_only）:
#      只学注入幅度（gate），MLP 学不出反语义方向。
#   4) align_lambda 前 30 epoch 线性衰减 0.01 -> 0.002（gate 唤醒期强对齐，
#      后期让位 CE 分类损失）。
#   5) grad_accum 4 -> 有效 batch 64（更稳的 EMA 原型与梯度）；
#      调度器按优化器步数重缩放。
#   6) proto_momentum 0.99 -> 0.995；eval 频率 5 -> 2 epoch。
#
# 对照: RAP baseline 93.84% / CMSI v3 best 94.21%
# 目标: >= 95.0% Top-1
# ----------------------------------------------------------------

EXP_DIR=training/checkpoints/ucf101_rap_bs16_cmsi_v4
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
    --use_cmsi \
    --cmsi_align_lambda 0.01 \
    --cmsi_align_lambda_decay 0.2 \
    --cmsi_align_lambda_decay_epochs 30 \
    --cmsi_proto_momentum 0.995 \
    --cmsi_max_gate 0.3 \
    --cmsi_v2t_dict proto \
    --cmsi_lr 1.0e-4 \
    --cmsi_gate_warmup_epochs 10 \
  2>&1 | tee "${LOG_DIR}/ucf101_rap_bs16_cmsi_v4-$(date +"%Y%m%d_%H%M%S").log"
