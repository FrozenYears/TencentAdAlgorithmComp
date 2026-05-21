#!/bin/bash
# TAAC2026 训练脚本 - 腾讯Angel平台

# 安装依赖
pip install -q scikit-learn tqdm

# 进入代码目录
cd $EVAL_INFER_PATH 2>/dev/null || cd .

# 运行训练
python train_v2.py \
    --data_path $TRAIN_DATA_PATH \
    --epochs 30 \
    --batch_size 256 \
    --embedding_dim 32 \
    --hidden_dims 128 64 32 \
    --dropout 0.3 \
    --lr 0.001 \
    --device cuda \
    --save_dir $TRAIN_CKPT_PATH
