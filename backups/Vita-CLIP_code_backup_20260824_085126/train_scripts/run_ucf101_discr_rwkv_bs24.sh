#!/bin/bash
set -e

# ----------------------------------------------------------------
# UCF101 split1 - Discriminative RWKV (bs=24)  --  resume-able
# ----------------------------------------------------------------

EXP_DIR=training/checkpoints/ucf101_discr_rwkv_bs24
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
    --batch_size 24 \
    --lr 1.5e-4 \
    --prompt_lr 1.5e-3 \
    --adapter_lr 6e-4 \
    --scalar_lr 3e-4 \
    --time_embed_lr 3e-4 \
    --backbone_path /hy-tmp/weights/ViT-B-16.pt \
    --train_list_path /hy-tmp/UCF101/annotations/ucfTrainTestlist/train_split1_frames.csv \
    --val_list_path /hy-tmp/UCF101/annotations/ucfTrainTestlist/val_split1_frames.csv \
    --data_root /hy-tmp/UCF101/rawframes \
    --num_frames 8 \
    --num_workers 8 \
    --use_discriminative_module \
    --dr_layers 3,7,11 \
  2>&1 | tee "${LOG_DIR}/ucf101_discr_rwkv_bs24-$(date +"%Y%m%d_%H%M%S").log"
