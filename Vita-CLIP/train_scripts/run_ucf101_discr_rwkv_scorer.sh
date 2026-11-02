#!/bin/bash
set -e

# ----------------------------------------------------------------
# UCF101 split1 - Discriminative RWKV + learnable spatial scorer (bs=24)
# Baseline v3 (no scorer): top1 = 90.59%.
# Deltas vs v3:
#   - --dr_topk_temperature 0.5      scorer + soft top-k CLS write-back
#   - --dr_top_k 16 (v3: 6)          wider spatial coverage per frame
#   - --dr_residual_scale_init 0.01 (v3: 0.01, now FIXED buffer)
#     rs is now a FIXED buffer (register_buffer, requires_grad=False) -- the
#     learnable scalar self-amplified near the lr peak (positive feedback:
#     rs up -> bigger perturbation -> bigger loss -> bigger grad -> rs up)
#     and caused the step-961 (rs=0.1) and step-1991 (rs=0.005) blowups.
#     Freezing at 0.01 = the v3-verified magnitude keeps the write-back
#     magnitude identical to the 90.59% baseline. CLS write-back is
#     softmax-normalized (sum=1), unaffected by top_k/rs.
# ----------------------------------------------------------------

EXP_DIR=training/checkpoints/ucf101_discr_rwkv_bs24_scorer
LOG_DIR=training/logs

mkdir -p "${EXP_DIR}" "${LOG_DIR}"

cd /hy-tmp/Vita-CLIP

nohup /hy-tmp/envs/Vita/bin/python -u -m torch.distributed.run --nproc_per_node 1 \
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
    --train_list_path /hy-tmp/UCF101/annotations/ucfTrainTestlist/trainlist01.txt \
    --val_list_path /hy-tmp/UCF101/annotations/ucfTrainTestlist/testlist01.txt \
    --data_root /hy-tmp/UCF101/rawframes \
    --num_frames 8 \
    --num_workers 8 \
    --use_discriminative_module \
    --dr_layers 3,7,11 \
    --dr_top_k 16 \
    --dr_residual_scale_init 0.01 \
    --dr_topk_temperature 0.5 \
  > "${LOG_DIR}/ucf101_discr_rwkv_scorer.log" 2>&1 &

echo "PID: $!"
