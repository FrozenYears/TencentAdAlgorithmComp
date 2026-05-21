#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

python3 -u "${SCRIPT_DIR}/train.py" \
    --d_model 64 \
    --emb_dim 32 \
    --num_cross_layers 3 \
    --batch_size 256 \
    --lr 1e-4 \
    --patience 5 \
    --loss_type focal \
    --focal_alpha 0.25 \
    --focal_gamma 2.0 \
    --num_workers 8 \
    "$@"
