"""
TAAC2026 推理脚本 - 腾讯Angel平台

必须满足：
1. 文件名必须是 infer.py
2. 必须包含 main() 函数（无参数）
3. 输出 predictions.json 到 EVAL_RESULT_PATH
"""

import os
import sys
import json
import numpy as np
import torch

# 添加当前目录到路径
sys.path.insert(0, os.environ.get('EVAL_INFER_PATH', '.'))

from feature_engineering import FeatureProcessorV2, prepare_data_v2, USER_SCALAR_INT, ITEM_SCALAR_INT, USER_LIST_INT, ITEM_LIST_INT, HASH_BUCKET_SIZE
from esmm_din_model import ESMM_DIN_V2


def main():
    """
    主推理函数 - 必须无参数
    
    环境变量:
        EVAL_DATA_PATH: 测试数据目录
        EVAL_RESULT_PATH: 输出目录
        MODEL_OUTPUT_PATH: 模型目录
    """
    print("=" * 50)
    print("TAAC2026 推理开始")
    print("=" * 50)
    
    # 读取环境变量
    eval_data_path = os.environ.get('EVAL_DATA_PATH')
    eval_result_path = os.environ.get('EVAL_RESULT_PATH')
    model_output_path = os.environ.get('MODEL_OUTPUT_PATH')
    
    print(f"EVAL_DATA_PATH: {eval_data_path}")
    print(f"EVAL_RESULT_PATH: {eval_result_path}")
    print(f"MODEL_OUTPUT_PATH: {model_output_path}")
    
    # 加载模型
    print("\n加载模型...")
    
    # 查找最新的checkpoint目录
    ckpt_dirs = [d for d in os.listdir(model_output_path) if d.startswith('global_step')]
    if not ckpt_dirs:
        raise FileNotFoundError(f"No global_step checkpoint found in {model_output_path}")
    
    latest_ckpt_dir = sorted(ckpt_dirs)[-1]
    ckpt_path = os.path.join(model_output_path, latest_ckpt_dir, 'model.pt')
    
    checkpoint = torch.load(
        ckpt_path,
        map_location='cpu',
        weights_only=False
    )
    
    # 从checkpoint获取模型配置
    model_args = checkpoint.get('args', {})
    
    # 准备数据
    print("\n加载数据...")
    import pandas as pd
    
    # 查找测试数据文件
    test_files = [f for f in os.listdir(eval_data_path) if f.endswith('.parquet')]
    if not test_files:
        # 尝试子目录
        for subdir in os.listdir(eval_data_path):
            subpath = os.path.join(eval_data_path, subdir)
            if os.path.isdir(subpath):
                test_files = [os.path.join(subpath, f) for f in os.listdir(subpath) if f.endswith('.parquet')]
                if test_files:
                    break
    
    if not test_files:
        raise FileNotFoundError(f"No parquet files found in {eval_data_path}")
    
    test_path = test_files[0]
    print(f"测试数据: {test_path}")
    
    df = pd.read_parquet(test_path)
    print(f"数据形状: {df.shape}")
    
    # 特征处理
    print("\n特征处理...")
    fp = FeatureProcessorV2(hash_bucket_size=HASH_BUCKET_SIZE)
    
    # 使用训练时的统计信息（如果可用）
    train_stats_path = os.path.join(model_output_path, 'feature_stats.pkl')
    if os.path.exists(train_stats_path):
        import pickle
        with open(train_stats_path, 'rb') as f:
            fp = pickle.load(f)
        print("加载训练时的特征统计")
    else:
        # 用测试数据fit（不理想但可用）
        fp.fit(df)
        print("使用测试数据fit特征统计")
    
    dataset = prepare_data_v2(df, fp, max_seq_len=50, is_test=True)
    
    # 构建模型
    print("\n构建模型...")
    user_scalar_dims = [fp.scalar_dims.get(c, 1) for c in USER_SCALAR_INT if c in fp.scalar_dims]
    item_scalar_dims = [fp.scalar_dims.get(c, 1) for c in ITEM_SCALAR_INT if c in fp.scalar_dims]
    user_list_dims = [fp.scalar_dims.get(c, 1) for c in USER_LIST_INT if c in fp.scalar_dims]
    item_list_dims = [fp.scalar_dims.get(c, 1) for c in ITEM_LIST_INT if c in fp.scalar_dims]
    
    model = ESMM_DIN_V2(
        n_user_scalar_feats=len(user_scalar_dims),
        n_item_scalar_feats=len(item_scalar_dims),
        n_user_list_feats=len(user_list_dims),
        n_item_list_feats=len(item_list_dims),
        user_scalar_dims=user_scalar_dims,
        item_scalar_dims=item_scalar_dims,
        user_list_dims=user_list_dims,
        item_list_dims=item_list_dims,
        user_dense_dim=dataset['user_dense'].shape[1],
        embedding_dim=model_args.get('embedding_dim', 32),
        seq_hash_bucket=HASH_BUCKET_SIZE + 1,
        hidden_dims=model_args.get('hidden_dims', [128, 64, 32]),
        dropout_rate=0.0,  # 推理时关闭dropout
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"模型参数: {sum(p.numel() for p in model.parameters()):,}")
    
    # 推理
    print("\n开始推理...")
    predictions = {}
    n_samples = len(df)
    batch_size = 256
    
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch = {
                'user_scalar': torch.tensor(dataset['user_scalar'][i:i+batch_size], dtype=torch.long),
                'user_list': torch.tensor(dataset['user_list'][i:i+batch_size], dtype=torch.long),
                'user_dense': torch.tensor(dataset['user_dense'][i:i+batch_size], dtype=torch.float32),
                'item_scalar': torch.tensor(dataset['item_scalar'][i:i+batch_size], dtype=torch.long),
                'item_list': torch.tensor(dataset['item_list'][i:i+batch_size], dtype=torch.long),
                'seq_a': torch.tensor(dataset['seq_a'][i:i+batch_size], dtype=torch.long),
                'seq_b': torch.tensor(dataset['seq_b'][i:i+batch_size], dtype=torch.long),
                'seq_c': torch.tensor(dataset['seq_c'][i:i+batch_size], dtype=torch.long),
                'seq_d': torch.tensor(dataset['seq_d'][i:i+batch_size], dtype=torch.long),
                'seq_mask': torch.tensor(dataset['seq_mask'][i:i+batch_size], dtype=torch.long),
            }
            
            p_ctr, p_cvr, p_ctcvr = model(batch)
            
            # 使用CTCVR作为最终预测
            for j in range(len(p_ctcvr)):
                idx = i + j
                if idx < n_samples:
                    user_id = str(df.iloc[idx]['user_id'])
                    predictions[user_id] = float(p_ctcvr[j].item())
            
            if (i // batch_size) % 10 == 0:
                print(f"  进度: {i}/{n_samples}")
    
    # 保存预测结果
    print(f"\n保存预测结果: {len(predictions)} 条")
    output = {"predictions": predictions}
    
    output_path = os.path.join(eval_result_path, 'predictions.json')
    with open(output_path, 'w') as f:
        json.dump(output, f)
    
    print(f"预测结果已保存到: {output_path}")
    print("=" * 50)
    print("推理完成!")
    print("=" * 50)


if __name__ == '__main__':
    main()
