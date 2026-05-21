"""
特征工程V2 - 增强版特征工程模块

新增优化:
1. 交叉特征: user_int × item_int 组合
2. 统计特征: 用户/物品历史CTR/CVR
3. 时间特征: hour, day_of_week, 时间差
4. 序列统计: 长度、唯一数、多样性
5. 所有序列域特征 (不再只用第一个列)
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

# 导入原始特征定义
from feature_engineering import (
    USER_SCALAR_INT, USER_LIST_INT, USER_DENSE,
    ITEM_SCALAR_INT, ITEM_LIST_INT,
    DOMAIN_A_SEQ, DOMAIN_B_SEQ, DOMAIN_C_SEQ, DOMAIN_D_SEQ,
    HASH_BUCKET_SIZE, safe_scalar, safe_list, safe_dense_list, hash_encode,
    FeatureProcessor
)


# ==================== 新增: 交叉特征定义 ====================

# 用于交叉的用户特征 (低基数, 适合做交叉)
CROSS_USER_FEATS = ['user_int_feats_1', 'user_int_feats_82']

# 用于交叉的物品特征 (低基数)
CROSS_ITEM_FEATS = ['item_int_feats_5', 'item_int_feats_6', 'item_int_feats_7']

# 交叉特征hash桶大小
CROSS_HASH_BUCKET = 5000


# ==================== 新增: 统计特征计算 ====================

def compute_statistical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算统计特征: 用户/物品的历史CTR、CVR

    Args:
        df: 包含user_id, item_id, label_type的DataFrame

    Returns:
        添加了统计特征的DataFrame
    """
    df = df.copy()

    # 用户历史统计
    user_stats = df.groupby('user_id').agg(
        user_hist_count=('label_type', 'count'),
        user_hist_click=('label_type', lambda x: (x >= 1).sum()),
        user_hist_convert=('label_type', lambda x: (x == 2).sum()),
    ).reset_index()
    user_stats['user_hist_ctr'] = user_stats['user_hist_click'] / user_stats['user_hist_count'].clip(lower=1)
    user_stats['user_hist_cvr'] = user_stats['user_hist_convert'] / user_stats['user_hist_click'].clip(lower=1)

    # 物品历史统计
    item_stats = df.groupby('item_id').agg(
        item_hist_count=('label_type', 'count'),
        item_hist_click=('label_type', lambda x: (x >= 1).sum()),
        item_hist_convert=('label_type', lambda x: (x == 2).sum()),
    ).reset_index()
    item_stats['item_hist_ctr'] = item_stats['item_hist_click'] / item_stats['item_hist_count'].clip(lower=1)
    item_stats['item_hist_cvr'] = item_stats['item_hist_convert'] / item_stats['item_hist_click'].clip(lower=1)

    # 合并统计特征
    df = df.merge(user_stats, on='user_id', how='left')
    df = df.merge(item_stats, on='item_id', how='left')

    # 填充缺失值
    for col in ['user_hist_ctr', 'user_hist_cvr', 'item_hist_ctr', 'item_hist_cvr']:
        df[col] = df[col].fillna(0.0)
    for col in ['user_hist_count', 'user_hist_click', 'user_hist_convert',
                 'item_hist_count', 'item_hist_click', 'item_hist_convert']:
        df[col] = df[col].fillna(0).astype(np.float32)

    return df


# ==================== 新增: 时间特征 ====================

def compute_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算时间特征: hour, day_of_week, 时间差

    Args:
        df: 包含timestamp, label_time的DataFrame

    Returns:
        添加了时间特征的DataFrame
    """
    df = df.copy()

    if 'timestamp' in df.columns:
        # 转换为datetime
        ts = pd.to_datetime(df['timestamp'], unit='s')
        df['time_hour'] = ts.dt.hour.astype(np.float32)
        df['time_day_of_week'] = ts.dt.dayofweek.astype(np.float32)
        df['time_is_weekend'] = (ts.dt.dayofweek >= 5).astype(np.float32)

        # 时间差 (label_time - timestamp), 单位秒
        if 'label_time' in df.columns:
            df['time_to_label'] = (df['label_time'] - df['timestamp']).clip(lower=0).astype(np.float32)
        else:
            df['time_to_label'] = 0.0
    else:
        df['time_hour'] = 0.0
        df['time_day_of_week'] = 0.0
        df['time_is_weekend'] = 0.0
        df['time_to_label'] = 0.0

    return df


# ==================== 新增: 序列统计特征 ====================

def compute_sequence_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算序列统计特征: 长度、唯一数、多样性

    Args:
        df: 包含domain序列的DataFrame

    Returns:
        添加了序列统计特征的DataFrame
    """
    df = df.copy()

    domains = {
        'a': DOMAIN_A_SEQ[0],
        'b': DOMAIN_B_SEQ[0],
        'c': DOMAIN_C_SEQ[0],
        'd': DOMAIN_D_SEQ[0],
    }

    for domain_name, col in domains.items():
        if col not in df.columns:
            df[f'seq_{domain_name}_len'] = 0.0
            df[f'seq_{domain_name}_unique'] = 0.0
            df[f'seq_{domain_name}_diversity'] = 0.0
            continue

        lengths = []
        uniques = []
        diversities = []
        for vals in df[col]:
            if isinstance(vals, (list, np.ndarray)):
                non_zero = [v for v in vals if v != 0]
                seq_len = len(non_zero)
                seq_unique = len(set(non_zero))
                diversity = seq_unique / max(seq_len, 1)
            else:
                seq_len = 0
                seq_unique = 0
                diversity = 0.0
            lengths.append(seq_len)
            uniques.append(seq_unique)
            diversities.append(diversity)

        df[f'seq_{domain_name}_len'] = np.array(lengths, dtype=np.float32)
        df[f'seq_{domain_name}_unique'] = np.array(uniques, dtype=np.float32)
        df[f'seq_{domain_name}_diversity'] = np.array(diversities, dtype=np.float32)

    return df


# ==================== 新增: 交叉特征编码 ====================

def compute_cross_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算交叉特征: user_int × item_int 组合

    Args:
        df: 包含用户和物品特征的DataFrame

    Returns:
        添加了交叉特征的DataFrame
    """
    df = df.copy()

    for u_feat in CROSS_USER_FEATS:
        for i_feat in CROSS_ITEM_FEATS:
            if u_feat in df.columns and i_feat in df.columns:
                # 组合特征: hash(u_val * 1000000 + i_val)
                u_vals = df[u_feat].fillna(0).astype(int)
                i_vals = df[i_feat].fillna(0).astype(int)
                cross_vals = u_vals * 1000000 + i_vals
                df[f'cross_{u_feat}_{i_feat}'] = cross_vals

    return df


# ==================== 增强版数据集 ====================

class TAACDatasetV2(Dataset):
    """
    TAAC2026数据集V2 - 增强版

    新增:
    - 统计特征 (用户/物品历史CTR/CVR)
    - 时间特征 (hour, day_of_week, is_weekend, time_to_label)
    - 序列统计 (长度、唯一数、多样性)
    - 交叉特征 (user_int × item_int)
    - 所有序列域特征
    """

    # 统计特征列
    STAT_FEATS = [
        'user_hist_count', 'user_hist_click', 'user_hist_convert',
        'user_hist_ctr', 'user_hist_cvr',
        'item_hist_count', 'item_hist_click', 'item_hist_convert',
        'item_hist_ctr', 'item_hist_cvr',
    ]

    # 时间特征列
    TIME_FEATS = ['time_hour', 'time_day_of_week', 'time_is_weekend', 'time_to_label']

    # 序列统计特征列
    SEQ_STAT_FEATS = []
    for d in ['a', 'b', 'c', 'd']:
        SEQ_STAT_FEATS.extend([f'seq_{d}_len', f'seq_{d}_unique', f'seq_{d}_diversity'])

    # 交叉特征列
    CROSS_FEATS = []
    for u_feat in CROSS_USER_FEATS:
        for i_feat in CROSS_ITEM_FEATS:
            CROSS_FEATS.append(f'cross_{u_feat}_{i_feat}')

    def __init__(
        self,
        df: pd.DataFrame,
        feature_processor: 'FeatureProcessorV2',
        max_seq_len: int = 50,
    ):
        self.df = df.reset_index(drop=True)
        self.fp = feature_processor
        self.max_seq_len = max_seq_len
        self.n_samples = len(df)

        # 标签
        self.ctr_label = (df['label_type'] >= 1).astype(np.float32).values
        self.cvr_label = (df['label_type'] == 2).astype(np.float32).values

        # 用户标量特征
        self.user_scalar_feats = np.zeros(
            (self.n_samples, len(USER_SCALAR_INT)), dtype=np.int64
        )
        for j, col in enumerate(USER_SCALAR_INT):
            if col in df.columns:
                for i in range(self.n_samples):
                    val = safe_scalar(df[col].iloc[i])
                    self.user_scalar_feats[i, j] = self.fp.encode_scalar(col, val)

        # 物品标量特征
        self.item_scalar_feats = np.zeros(
            (self.n_samples, len(ITEM_SCALAR_INT)), dtype=np.int64
        )
        for j, col in enumerate(ITEM_SCALAR_INT):
            if col in df.columns:
                for i in range(self.n_samples):
                    val = safe_scalar(df[col].iloc[i])
                    self.item_scalar_feats[i, j] = self.fp.encode_scalar(col, val)

        # 用户列表特征
        max_list_len = max(self.fp.list_max_lens.values()) if self.fp.list_max_lens else 10
        self.user_list_feats = np.zeros(
            (self.n_samples, len(USER_LIST_INT), max_list_len), dtype=np.int64
        )
        for j, col in enumerate(USER_LIST_INT):
            if col in df.columns:
                for i in range(self.n_samples):
                    vals = safe_list(df[col].iloc[i], max_list_len)
                    encoded = self.fp.encode_list(col, vals, max_list_len)
                    self.user_list_feats[i, j, :len(encoded)] = encoded

        # 物品列表特征
        self.item_list_feats = np.zeros(
            (self.n_samples, len(ITEM_LIST_INT), max_list_len), dtype=np.int64
        )
        for j, col in enumerate(ITEM_LIST_INT):
            if col in df.columns:
                for i in range(self.n_samples):
                    vals = safe_list(df[col].iloc[i], max_list_len)
                    encoded = self.fp.encode_list(col, vals, max_list_len)
                    self.item_list_feats[i, j, :len(encoded)] = encoded

        # 用户稠密特征
        total_dense_dim = sum(self.fp.dense_dims.values())
        self.user_dense_feats = np.zeros((self.n_samples, total_dense_dim), dtype=np.float32)
        offset = 0
        for col in USER_DENSE:
            if col in df.columns and col in self.fp.dense_dims:
                dim = self.fp.dense_dims[col]
                for i in range(self.n_samples):
                    vals = safe_dense_list(df[col].iloc[i], dim)
                    self.user_dense_feats[i, offset:offset+dim] = vals[:dim]
                offset += dim

        # 统计特征
        self.stat_feats = np.zeros((self.n_samples, len(self.STAT_FEATS)), dtype=np.float32)
        for j, col in enumerate(self.STAT_FEATS):
            if col in df.columns:
                self.stat_feats[:, j] = df[col].fillna(0).values.astype(np.float32)

        # 时间特征
        self.time_feats = np.zeros((self.n_samples, len(self.TIME_FEATS)), dtype=np.float32)
        for j, col in enumerate(self.TIME_FEATS):
            if col in df.columns:
                self.time_feats[:, j] = df[col].fillna(0).values.astype(np.float32)

        # 序列统计特征
        self.seq_stat_feats = np.zeros((self.n_samples, len(self.SEQ_STAT_FEATS)), dtype=np.float32)
        for j, col in enumerate(self.SEQ_STAT_FEATS):
            if col in df.columns:
                self.seq_stat_feats[:, j] = df[col].fillna(0).values.astype(np.float32)

        # 交叉特征
        n_cross = len(self.CROSS_FEATS)
        self.cross_feats = np.zeros((self.n_samples, n_cross), dtype=np.int64)
        for j, col in enumerate(self.CROSS_FEATS):
            if col in df.columns:
                for i in range(self.n_samples):
                    val = int(df[col].iloc[i]) if pd.notna(df[col].iloc[i]) else 0
                    self.cross_feats[i, j] = hash_encode(val, CROSS_HASH_BUCKET)

        # 序列特征 (使用所有序列列, 不只是第一个)
        self._process_domain_sequences(df)

    def _process_domain_sequences(self, df: pd.DataFrame):
        """处理域序列特征: 使用所有序列列, 截断/padding到max_seq_len"""
        domains = {
            'a': DOMAIN_A_SEQ,
            'b': DOMAIN_B_SEQ,
            'c': DOMAIN_C_SEQ,
            'd': DOMAIN_D_SEQ,
        }

        self.domain_seqs = {}
        for domain_name, cols in domains.items():
            # 使用所有列, 取第一个非零值作为序列
            first_col = cols[0]
            if first_col not in df.columns:
                self.domain_seqs[domain_name] = np.zeros((self.n_samples, self.max_seq_len), dtype=np.int64)
                continue

            seq_data = np.zeros((self.n_samples, self.max_seq_len), dtype=np.int64)
            for i in range(self.n_samples):
                # 收集该domain所有列的值
                all_vals = []
                for col in cols:
                    if col in df.columns:
                        vals = safe_list(df[col].iloc[i], self.max_seq_len)
                        all_vals.extend([int(v) for v in vals if v != 0])

                # 去重并截断
                unique_vals = list(dict.fromkeys(all_vals))[:self.max_seq_len]
                for j, v in enumerate(unique_vals):
                    seq_data[i, j] = hash_encode(v) if v else 0

            self.domain_seqs[domain_name] = seq_data

        # 序列mask
        any_seq = None
        for domain_name, seq_data in self.domain_seqs.items():
            if any_seq is None:
                any_seq = (seq_data > 0).astype(np.float32)
            else:
                any_seq = np.maximum(any_seq, (seq_data > 0).astype(np.float32))
        self.seq_mask = any_seq if any_seq is not None else np.zeros((self.n_samples, self.max_seq_len), dtype=np.float32)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return {
            'user_scalar': torch.tensor(self.user_scalar_feats[idx], dtype=torch.long),
            'user_list': torch.tensor(self.user_list_feats[idx], dtype=torch.long),
            'user_dense': torch.tensor(self.user_dense_feats[idx], dtype=torch.float32),
            'item_scalar': torch.tensor(self.item_scalar_feats[idx], dtype=torch.long),
            'item_list': torch.tensor(self.item_list_feats[idx], dtype=torch.long),
            'stat_feats': torch.tensor(self.stat_feats[idx], dtype=torch.float32),
            'time_feats': torch.tensor(self.time_feats[idx], dtype=torch.float32),
            'seq_stat_feats': torch.tensor(self.seq_stat_feats[idx], dtype=torch.float32),
            'cross_feats': torch.tensor(self.cross_feats[idx], dtype=torch.long),
            'seq_a': torch.tensor(self.domain_seqs['a'][idx], dtype=torch.long),
            'seq_b': torch.tensor(self.domain_seqs['b'][idx], dtype=torch.long),
            'seq_c': torch.tensor(self.domain_seqs['c'][idx], dtype=torch.long),
            'seq_d': torch.tensor(self.domain_seqs['d'][idx], dtype=torch.long),
            'seq_mask': torch.tensor(self.seq_mask[idx], dtype=torch.float32),
            'ctr_label': torch.tensor(self.ctr_label[idx], dtype=torch.float32),
            'cvr_label': torch.tensor(self.cvr_label[idx], dtype=torch.float32),
        }


# ==================== 增强版特征处理器 ====================

class FeatureProcessorV2(FeatureProcessor):
    """特征处理器V2: 继承原始处理器, 添加新特征支持"""

    def __init__(self, hash_bucket_size=HASH_BUCKET_SIZE):
        super().__init__(hash_bucket_size)
        # 交叉特征维度
        self.cross_dims: Dict[str, int] = {}

    def fit(self, df: pd.DataFrame):
        """从训练数据中统计特征信息"""
        # 先调用父类fit
        super().fit(df)

        # 统计交叉特征维度
        for col in TAACDatasetV2.CROSS_FEATS:
            if col in df.columns:
                self.cross_dims[col] = CROSS_HASH_BUCKET + 1

        print(f"[FeatureProcessorV2] 新增: {len(self.cross_dims)}个交叉特征")


# ==================== 工具函数 ====================

def prepare_data_v2(
    data_path: str,
    test_size: float = 0.2,
    random_state: int = 42,
    max_seq_len: int = 50,
) -> Tuple[TAACDatasetV2, TAACDatasetV2, FeatureProcessorV2]:
    """
    加载数据并准备训练/验证集 (V2版本)

    Returns:
        train_dataset, val_dataset, feature_processor
    """
    print(f"[数据V2] 加载 {data_path}...")
    df = pd.read_parquet(data_path)
    print(f"[数据V2] Shape: {df.shape}")
    print(f"[数据V2] 标签分布:\n{df['label_type'].value_counts().sort_index()}")

    # 计算统计特征 (需要在划分前, 因为需要全局统计)
    print("[数据V2] 计算统计特征...")
    df = compute_statistical_features(df)

    # 计算时间特征
    print("[数据V2] 计算时间特征...")
    df = compute_time_features(df)

    # 计算序列统计特征
    print("[数据V2] 计算序列统计特征...")
    df = compute_sequence_statistics(df)

    # 计算交叉特征
    print("[数据V2] 计算交叉特征...")
    df = compute_cross_features(df)

    # 时间排序划分 (避免数据泄漏)
    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp').reset_index(drop=True)
        split_idx = int(len(df) * (1 - test_size))
        train_df = df.iloc[:split_idx].reset_index(drop=True)
        val_df = df.iloc[split_idx:].reset_index(drop=True)
    else:
        from sklearn.model_selection import train_test_split
        train_df, val_df = train_test_split(
            df, test_size=test_size, random_state=random_state, stratify=df['label_type']
        )
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)

    print(f"[数据V2] 训练集: {len(train_df)}, 验证集: {len(val_df)}")

    # 特征处理器 (只在训练集上fit)
    fp = FeatureProcessorV2(hash_bucket_size=HASH_BUCKET_SIZE)
    fp.fit(train_df)

    # 创建数据集
    train_dataset = TAACDatasetV2(train_df, fp, max_seq_len=max_seq_len)
    val_dataset = TAACDatasetV2(val_df, fp, max_seq_len=max_seq_len)

    return train_dataset, val_dataset, fp
