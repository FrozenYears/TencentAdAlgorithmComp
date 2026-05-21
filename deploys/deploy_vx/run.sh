#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --ns_groups_json "" \
    --emb_skip_threshold 1000000 \
    --d_model 64 \
    --emb_dim 64 \
    --num_hyformer_blocks 2 \
    --num_heads 4 \
    --seq_encoder_type transformer \
    --hidden_mult 4 \
    --dropout_rate 0.01 \
    --batch_size 128 \
    --gradient_accumulation_steps 2 \
    --use_amp \
    --loss_type bce \
    --lr 1e-4 \
    --patience 5 \
    --num_workers 4 \
    --buffer_batches 10 \
    "$@"
