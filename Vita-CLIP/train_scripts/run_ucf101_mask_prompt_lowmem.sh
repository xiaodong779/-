#!/bin/bash
# 低显存版：GPU 只剩 ~8GB 可用时使用（batch 1 / accum 20，有效 batch 仍为 20）
set -e

EXP_DIR=training/checkpoints/ucf101_mstff_maskprompt
LOG_DIR=training/logs
PYTHON_BIN=/hy-tmp/envs/clip_sam2/bin/python
export CUDA_HOME=/usr/local/cuda
export HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "${EXP_DIR}" "${LOG_DIR}"
cd /hy-tmp/Vita-CLIP

exec "${PYTHON_BIN}" -u -m torch.distributed.run --nproc_per_node 1 \
  ./training/train_OPENAI_ALIGNED.py \
    --checkpoint_dir "${EXP_DIR}" \
    --auto_resume \
    --epochs 60 --warmup_epochs 5 --eval_every_epochs 5 --save_every_epochs 5 \
    --batch_size 1 --grad_accum 20 --num_workers 2 \
    --train_rrc_scale "0.5,1.0" --label_smoothing 0.1 \
    --lr 1.0e-4 --prompt_lr 1.0e-3 --adapter_lr 4.0e-4 --scalar_lr 2.0e-4 --time_embed_lr 2.0e-4 \
    --backbone_path /hy-tmp/weights/ViT-B-16.pt \
    --train_list_path /hy-tmp/UCF101/annotations/ucfTrainTestlist/trainlist01.txt \
    --val_list_path /hy-tmp/UCF101/annotations/ucfTrainTestlist/testlist01.txt \
    --data_root /hy-tmp/UCF101/rawframes --num_frames 8 \
    --mask_prompt_enabled --mask_prompt_online --mask_prompt_num_frames 8 \
    --mask_prompt_text_prompt "person ." --mask_prompt_box_threshold 0.35 \
    --mask_prompt_text_threshold 0.25 --mask_prompt_expand_ratio 0.30 \
    --mask_prompt_sam2_prompt_frames "0,4,7" \
    --mask_prompt_learnable_alpha --mask_prompt_alpha_init 0.40 \
    --mask_prompt_alpha_min 0.35 --mask_prompt_alpha_max 0.80 \
    --mask_prompt_learnable_gamma --mask_prompt_gamma_init 0.20 \
    --mask_prompt_gamma_min 0.15 --mask_prompt_gamma_max 0.40 \
    --mask_prompt_debug_visualization \
    --mask_prompt_visualization_dir "${EXP_DIR}/mask_prompt_vis" \
    --mask_prompt_dino_checkpoint /hy-tmp/Vita-CLIP/third_party/GroundingDINO/weights/groundingdino_swint_ogc.pth \
    --mask_prompt_sam2_config configs/sam2.1/sam2.1_hiera_t.yaml \
    --mask_prompt_sam2_checkpoint /hy-tmp/Vita-CLIP/third_party/sam2/checkpoints/sam2.1_hiera_tiny.pt
