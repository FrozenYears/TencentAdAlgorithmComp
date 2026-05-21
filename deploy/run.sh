#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3 -u "${SCRIPT_DIR}/train.py" \
    --d_model 32 \
    --emb_dim 16 \
    --num_cross_layers 2 \
    --batch_size 64 \
    --gradient_accumulation_steps 4 \
    --use_amp \
    --emb_skip_threshold 1000000 \
    --lr 1e-4 \
    --patience 5 \
    --loss_type focal \
    --focal_alpha 0.25 \
    --focal_gamma 2.0 \
    --num_workers 4 \
    --buffer_batches 10 \
    "$@"
