#!/bin/bash
set -e

# ----------------------------------------------------------------
# UCF101 split1 - RAP + BIE + Aggregation Temporal Mixer
#
# 与 run_ucf101_rap_bie.sh 的唯一区别: 新增 --use_aggregation_mixer，
# 在视频特征 reshape [B,T,D] 之后、mean(dim=1) 之前插入 depthwise Conv1d
# 时序混合（恒等起点: conv 零初始化 + scale=0 -> 初始等价纯 mean）。
#
# 期望: 在 BIE 双向插值之上，给帧聚合前补一刀时间维混合，
#       缓解纯 mean 丢失帧序信息的问题。
# ----------------------------------------------------------------

EXP_DIR=training/checkpoints/ucf101_rap_bs16_bie_mixer
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
    --use_aggregation_mixer \
    --aggregation_mixer_kernel 3 \
  2>&1 | tee "${LOG_DIR}/ucf101_rap_bs16_bie_mixer-$(date +"%Y%m%d_%H%M%S").log"
