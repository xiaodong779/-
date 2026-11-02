#!/bin/bash
set -e

# ----------------------------------------------------------------
# UCF101 split1 - 仅 Text_MSTFF + 冻结 Grounding-DINO + SAM2 在线
# Mask Prompt 残差增强（RAP / BIE / Aggregation Mixer 已全部关闭）
#
# 处理顺序（严格遵守）:
#   解码 -> 采样8帧 -> 统一几何变换(RRC/flip, dataset内) -> 保留[0,1] RGB
#   -> 冻结Grounding-DINO检测(person优先) -> 冻结SAM2逐帧mask(0/4/7帧校正)
#   -> Mask Prompt残差增强(spatial=actor+alpha*interaction;
#      enhanced=rgb*(1+gamma*mask)) -> CLIP Normalize(模型内) -> 冻结CLIP
#
# 保留的可训练模块: 文本 prompt learning + Text_MSTFF + time_embed
#                 + logit_scale + mask_prompt alpha/gamma(有界可学习)
#
# 权重与代码（已安装并验证）:
#   Grounding-DINO: third_party/GroundingDINO (commit 856dde20)
#   SAM 2.1 Tiny:   third_party/sam2 (commit 2b90b9f5)
#   环境: /hy-tmp/envs/clip_sam2 (torch 2.5.1+cu121)
# ----------------------------------------------------------------

EXP_DIR=training/checkpoints/ucf101_mstff_maskprompt
LOG_DIR=training/logs

# DINO config 留空时 Python 自动定位 pip 包内 SwinT 配置
# clip_sam2 环境: torch 2.5.1+cu121 (满足 SAM2 >=2.5.1 要求)
PYTHON_BIN=/hy-tmp/envs/clip_sam2/bin/python
export CUDA_HOME=/usr/local/cuda
export HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DINO_CONFIG=""
DINO_CKPT="/hy-tmp/Vita-CLIP/third_party/GroundingDINO/weights/groundingdino_swint_ogc.pth"
SAM2_CONFIG="configs/sam2.1/sam2.1_hiera_t.yaml"
SAM2_CKPT="/hy-tmp/Vita-CLIP/third_party/sam2/checkpoints/sam2.1_hiera_tiny.pt"

mkdir -p "${EXP_DIR}" "${LOG_DIR}"

cd /hy-tmp/Vita-CLIP

"${PYTHON_BIN}" -u -m torch.distributed.run --nproc_per_node 1 \
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
    --mask_prompt_enabled \
    --mask_prompt_online \
    --mask_prompt_num_frames 8 \
    --mask_prompt_text_prompt "person ." \
    --mask_prompt_box_threshold 0.35 \
    --mask_prompt_text_threshold 0.25 \
    --mask_prompt_expand_ratio 0.30 \
    --mask_prompt_sam2_prompt_frames "0,4,7" \
    --mask_prompt_learnable_alpha \
    --mask_prompt_alpha_init 0.40 \
    --mask_prompt_alpha_min 0.35 \
    --mask_prompt_alpha_max 0.80 \
    --mask_prompt_learnable_gamma \
    --mask_prompt_gamma_init 0.20 \
    --mask_prompt_gamma_min 0.15 \
    --mask_prompt_gamma_max 0.40 \
    --mask_prompt_debug_visualization \
    --mask_prompt_visualization_dir "${EXP_DIR}/mask_prompt_vis" \
    --mask_prompt_dino_checkpoint "${DINO_CKPT}" \
    --mask_prompt_sam2_config "${SAM2_CONFIG}" \
    --mask_prompt_sam2_checkpoint "${SAM2_CKPT}" \
  2>&1 | tee "${LOG_DIR}/ucf101_mstff_maskprompt-$(date +"%Y%m%d_%H%M%S").log"
