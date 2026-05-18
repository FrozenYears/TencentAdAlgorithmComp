"""
特征工程模块 - 处理TAAC2026的120列flat parquet格式数据

特征分类:
- 用户标量int特征 (user_int_feats_*): 34个
- 用户列表int特征 (user_int_feats_*): 11个
- 用户稠密特征 (user_dense_feats_*): 10个
- 物品标量int特征 (item_int_feats_*): 13个
- 物品列表int特征 (item_int_feats_*): 1个
- 域序列特征 (domain_{a,b,c,d}_seq_*): 45个
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional


# ==================== 特征分类常量 ====================

# 用户标量int特征 (低基数直接embedding，高基数hash)
USER_SCALAR_INT = [
    'user_int_feats_1', 'user_int_feats_3', 'user_int_feats_4',
    'user_int_feats_48', 'user_int_feats_49', 'user_int_feats_50',
    'user_int_feats_51', 'user_int_feats_52', 'user_int_feats_53',
    'user_int_feats_54', 'user_int_feats_55', 'user_int_feats_56',
    'user_int_feats_57', 'user_int_feats_58', 'user_int_feats_59',
    'user_int_feats_82', 'user_int_feats_86', 'user_int_feats_92',
    'user_int_feats_93', 'user_int_feats_94', 'user_int_feats_95',
    'user_int_feats_96', 'user_int_feats_97', 'user_int_feats_98',
    'user_int_feats_99', 'user_int_feats_100', 'user_int_feats_101',
    'user_int_feats_102', 'user_int_feats_103', 'user_int_feats_104',
    'user_int_feats_105', 'user_int_feats_106', 'user_int_feats_107',
    'user_int_feats_108', 'user_int_feats_109',
]

# 用户列表int特征
USER_LIST_INT = [
    'user_int_feats_15', 'user_int_feats_60', 'user_int_feats_62',
    'user_int_feats_63', 'user_int_feats_64', 'user_int_feats_65',
    'user_int_feats_66', 'user_int_feats_80',
    'user_int_feats_89', 'user_int_feats_90', 'user_int_feats_91',
]

# 用户稠密特征
USER_DENSE = [
    'user_dense_feats_61', 'user_dense_feats_62', 'user_dense_feats_63',
    'user_dense_feats_64', 'user_dense_feats_65', 'user_dense_feats_66',
    'user_dense_feats_87', 'user_dense_feats_89', 'user_dense_feats_90',
    'user_dense_feats_91',
]

# 物品标量int特征
ITEM_SCALAR_INT = [
    'item_int_feats_5', 'item_int_feats_6', 'item_int_feats_7',
    'item_int_feats_8', 'item_int_feats_9', 'item_int_feats_10',
    'item_int_feats_12', 'item_int_feats_13', 'item_int_feats_16',
    'item_int_feats_81', 'item_int_feats_83', 'item_int_feats_84',
    'item_int_feats_85',
]

# 物品列表int特征
ITEM_LIST_INT = ['item_int_feats_11']

# 域序列特征分组
DOMAIN_A_SEQ = [f'domain_a_seq_{i}' for i in [38,39,40,41,42,43,44,45,46]]
DOMAIN_B_SEQ = [f'domain_b_seq_{i}' for i in [67,68,69,70,71,72,73,74,75,76,77,78,79,88]]
DOMAIN_C_SEQ = [f'domain_c_seq_{i}' for i in [27,28,29,30,31,32,33,34,35,36,37,47]]
DOMAIN_D_SEQ = [f'domain_d_seq_{i}' for i in [17,18,19,20,21,22,23,24,25,26]]

# Hash桶大小 (高基数特征)
HASH_BUCKET_SIZE = 10000


def safe_scalar(val, default=0):
    """安全提取标量值，处理NaN和None"""
    if val is None:
        return default
    if isinstance(val, (list, np.ndarray)):
        return default
    try:
        if np.isnan(val):
            return default
    except (TypeError, ValueError):
        pass
    return int(val)


def safe_list(val, max_len=10):
    """安全提取列表值，截断到max_len"""
    if val is None:
        return []
    if isinstance(val, (list, np.ndarray)):
        return list(val[:max_len])
    return []


def safe_dense_list(val, dim=None):
    """安全提取稠密向量, 保证返回恰好dim个元素"""
    if val is None:
        return [0.0] * dim if dim else []
    if isinstance(val, (list, np.ndarray)):
        result = list(val)
        if dim is not None:
            if len(result) >= dim:
                return result[:dim]
            else:
                return result + [0.0] * (dim - len(result))
        return result
    return [0.0] * dim if dim else []


# ==================== 特征统计和编码 ====================

class FeatureProcessor:
    """特征处理器：统计特征基数、建立映射、处理hash"""

    def __init__(self, hash_bucket_size=HASH_BUCKET_SIZE):
        self.hash_bucket_size = hash_bucket_size
        # 标量特征的唯一值映射 {col_name: {val: encoded_id}}
        self.scalar_mappings: Dict[str, Optional[Dict[int, int]]] = {}
        # 特征的最大编码ID (用于embedding大小)
        self.scalar_dims: Dict[str, int] = {}
        # 列表特征的最大长度
        self.list_max_lens: Dict[str, int] = {}
        # 稠密特征的维度
        self.dense_dims: Dict[str, int] = {}
        # 序列特征的最大长度
        self.seq_max_len = 50
        # 是否已fit
        self.is_fitted = False

    def fit(self, df: pd.DataFrame):
        """从训练数据中统计特征信息"""
        # 处理用户标量特征
        for col in USER_SCALAR_INT:
            if col not in df.columns:
                continue
            vals = df[col].apply(safe_scalar).astype(int)
            unique_vals = sorted(vals.unique())
            if len(unique_vals) > self.hash_bucket_size:
                # 高基数特征用hash
                self.scalar_mappings[col] = None  # None表示用hash
                self.scalar_dims[col] = self.hash_bucket_size + 1  # hash返回[1, bucket_size], 需要+1
            else:
                mapping = {v: i + 1 for i, v in enumerate(unique_vals)}  # 0留给padding
                self.scalar_mappings[col] = mapping
                self.scalar_dims[col] = len(mapping) + 1

        # 处理物品标量特征
        for col in ITEM_SCALAR_INT:
            if col not in df.columns:
                continue
            vals = df[col].apply(safe_scalar).astype(int)
            unique_vals = sorted(vals.unique())
            if len(unique_vals) > self.hash_bucket_size:
                self.scalar_mappings[col] = None
                self.scalar_dims[col] = self.hash_bucket_size + 1
            else:
                mapping = {v: i + 1 for i, v in enumerate(unique_vals)}
                self.scalar_mappings[col] = mapping
                self.scalar_dims[col] = len(mapping) + 1

        # 处理列表特征: 最大长度 + 建立值映射(同标量特征逻辑)
        for col in USER_LIST_INT + ITEM_LIST_INT:
            if col not in df.columns:
                continue
            max_len = df[col].apply(lambda x: len(safe_list(x, 100))).max()
            self.list_max_lens[col] = min(int(max_len), 20)
            all_vals = []
            for v in df[col].dropna():
                if isinstance(v, (list, np.ndarray)):
                    all_vals.extend([safe_scalar(x) for x in v])
            unique_vals = sorted(set(all_vals)) if all_vals else [0]
            if len(unique_vals) > self.hash_bucket_size:
                self.scalar_mappings[col] = None
                self.scalar_dims[col] = self.hash_bucket_size + 1
            else:
                mapping = {v: i + 1 for i, v in enumerate(unique_vals)}
                self.scalar_mappings[col] = mapping
                self.scalar_dims[col] = len(mapping) + 1

        # 处理稠密特征维度
        for col in USER_DENSE:
            if col not in df.columns:
                continue
            sample = df[col].iloc[0]
            if isinstance(sample, (list, np.ndarray)):
                self.dense_dims[col] = len(sample)
            else:
                self.dense_dims[col] = 1

        self.is_fitted = True
        print(f"[FeatureProcessor] 特征统计完成: "
              f"{len(self.scalar_dims)}个标量特征, "
              f"{len(self.list_max_lens)}个列表特征, "
              f"{len(self.dense_dims)}个稠密特征")

    def encode_scalar(self, col: str, val: int) -> int:
        """编码标量特征值"""
        mapping = self.scalar_mappings.get(col)
        if mapping is None:
            # Hash编码
            return hash(val) % self.hash_bucket_size + 1
        return mapping.get(val, 0)  # 0是unknown/padding

    def encode_list(self, col: str, vals: list, max_len: int = None) -> List[int]:
        """编码列表特征值, 对每个元素应用encode_scalar(含hash), padding到固定长度"""
        if max_len is None:
            max_len = self.list_max_lens.get(col, 10)
        processed = []
        for v in vals[:max_len]:
            processed.append(self.encode_scalar(col, safe_scalar(v)))
        while len(processed) < max_len:
            processed.append(0)
        return processed


def hash_encode(val: int, bucket_size: int = HASH_BUCKET_SIZE) -> int:
    """简单hash编码"""
    return hash(val) % bucket_size + 1


# ==================== 数据集 ====================

class TAACDataset(Dataset):
    """
    TAAC2026数据集

    处理120列flat parquet格式，输出:
    - user_scalar: [batch, n_user_scalar]
    - user_list: [batch, n_user_list, max_list_len]
    - user_dense: [batch, total_dense_dim]
    - item_scalar: [batch, n_item_scalar]
    - item_list: [batch, n_item_list, max_list_len]
    - seq_a/b/c/d: [batch, max_seq_len, n_seq_cols] (每域的序列特征)
    - seq_mask: [batch, max_seq_len]
    - ctr_label: [batch] (0/1: 是否点击)
    - cvr_label: [batch] (0/1: 是否转化, 仅对点击样本有意义)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_processor: FeatureProcessor,
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

        # 序列特征 (使用item_id列作为序列, 取每个domain的第一个seq列)
        self._process_domain_sequences(df)

    def _process_domain_sequences(self, df: pd.DataFrame):
        """处理域序列特征: 每个domain取item_id序列，截断/padding到max_seq_len"""
        # domain_a的第一个列(seq_38)是item_id
        # 各domain的序列长度不同，统一截断到max_seq_len
        domains = {
            'a': DOMAIN_A_SEQ,
            'b': DOMAIN_B_SEQ,
            'c': DOMAIN_C_SEQ,
            'd': DOMAIN_D_SEQ,
        }

        self.domain_seqs = {}
        for domain_name, cols in domains.items():
            # 只用第一个列(item_id)作为序列
            first_col = cols[0]
            if first_col not in df.columns:
                self.domain_seqs[domain_name] = np.zeros((self.n_samples, self.max_seq_len), dtype=np.int64)
                continue

            seq_data = np.zeros((self.n_samples, self.max_seq_len), dtype=np.int64)
            for i in range(self.n_samples):
                vals = safe_list(df[first_col].iloc[i], self.max_seq_len)
                for j, v in enumerate(vals[:self.max_seq_len]):
                    seq_data[i, j] = hash_encode(int(v)) if v else 0
            self.domain_seqs[domain_name] = seq_data

        # 序列mask (非零位置为1)
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
            'seq_a': torch.tensor(self.domain_seqs['a'][idx], dtype=torch.long),
            'seq_b': torch.tensor(self.domain_seqs['b'][idx], dtype=torch.long),
            'seq_c': torch.tensor(self.domain_seqs['c'][idx], dtype=torch.long),
            'seq_d': torch.tensor(self.domain_seqs['d'][idx], dtype=torch.long),
            'seq_mask': torch.tensor(self.seq_mask[idx], dtype=torch.float32),
            'ctr_label': torch.tensor(self.ctr_label[idx], dtype=torch.float32),
            'cvr_label': torch.tensor(self.cvr_label[idx], dtype=torch.float32),
        }


# ==================== 工具函数 ====================

def prepare_data(
    data_path: str,
    test_size: float = 0.2,
    random_state: int = 42,
    max_seq_len: int = 50,
) -> Tuple[TAACDataset, TAACDataset, FeatureProcessor]:
    """
    加载数据并准备训练/验证集

    Returns:
        train_dataset, val_dataset, feature_processor
    """
    print(f"[数据] 加载 {data_path}...")
    df = pd.read_parquet(data_path)
    print(f"[数据] Shape: {df.shape}")
    print(f"[数据] 标签分布:\n{df['label_type'].value_counts().sort_index()}")

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

    print(f"[数据] 训练集: {len(train_df)}, 验证集: {len(val_df)}")

    # 特征处理器 (只在训练集上fit)
    fp = FeatureProcessor(hash_bucket_size=HASH_BUCKET_SIZE)
    fp.fit(train_df)

    # 创建数据集
    train_dataset = TAACDataset(train_df, fp, max_seq_len=max_seq_len)
    val_dataset = TAACDataset(val_df, fp, max_seq_len=max_seq_len)

    return train_dataset, val_dataset, fp
