"""
2026腾讯广告算法大赛 - pCVR预测Baseline
任务：预测目标广告的预测转化率(pCVR)
评估：AUC of ROC
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ==================== 数据处理 ====================
class CVRDataset(Dataset):
    """CVR预测数据集"""
    
    def __init__(self, df, user_encoders, item_encoders, max_seq_len=10):
        self.df = df.reset_index(drop=True)
        self.max_seq_len = max_seq_len
        
        self.user_int_cols = [c for c in df.columns if c.startswith('user_int_feats_')]
        self.item_int_cols = [c for c in df.columns if c.startswith('item_int_feats_')]
        self.seq_cols = [c for c in df.columns if 'seq' in c]
        
        self.labels = (df['label_type'] == 2).astype(np.float32).values
        
        # 处理用户特征 - 只取标量int列，使用LabelEncoder
        user_feats_list = []
        self.user_encoders = {}
        for col in self.user_int_cols:
            try:
                vals = df[col].apply(lambda x: x if isinstance(x, (int, float)) else 0).fillna(0).values
                le = LabelEncoder()
                encoded = le.fit_transform(vals.astype(str))
                user_feats_list.append(encoded)
                self.user_encoders[col] = le
            except:
                pass
        self.user_feats = np.stack(user_feats_list, axis=1) if user_feats_list else np.zeros((len(df), 1), dtype=np.int64)
        
        # 处理物品特征 - 只取标量int列，使用LabelEncoder
        item_feats_list = []
        self.item_encoders = {}
        for col in self.item_int_cols:
            try:
                vals = df[col].apply(lambda x: x if isinstance(x, (int, float)) else 0).fillna(0).values
                le = LabelEncoder()
                encoded = le.fit_transform(vals.astype(str))
                item_feats_list.append(encoded)
                self.item_encoders[col] = le
            except:
                pass
        self.item_feats = np.stack(item_feats_list, axis=1) if item_feats_list else np.zeros((len(df), 1), dtype=np.int64)
        
        # 序列特征（取第一个序列列作为示例）
        self.seq_feats = []
        for idx in range(len(df)):
            seq = []
            for col in self.seq_cols[:5]:  # 只用前5个序列列
                val = df.iloc[idx][col]
                if isinstance(val, list) and len(val) > 0:
                    seq.extend(val[:max_seq_len])
            if len(seq) == 0:
                seq = [0] * max_seq_len
            elif len(seq) < max_seq_len:
                seq = seq + [0] * (max_seq_len - len(seq))
            else:
                seq = seq[:max_seq_len]
            self.seq_feats.append(seq)
        self.seq_feats = np.array(self.seq_feats, dtype=np.int64)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        return {
            'user_feats': torch.tensor(self.user_feats[idx], dtype=torch.long),
            'item_feats': torch.tensor(self.item_feats[idx], dtype=torch.long),
            'seq_feats': torch.tensor(self.seq_feats[idx], dtype=torch.long),
            'label': torch.tensor(self.labels[idx], dtype=torch.float32)
        }


# ==================== 模型定义 ====================
class SimpleCVRModel(nn.Module):
    """简单的CVR预测模型"""
    
    def __init__(self, n_user_feats, n_item_feats, n_seq_feats, 
                 embedding_dim=16, hidden_dim=64):
        super().__init__()
        
        self.user_emb = nn.Embedding(n_user_feats + 1, embedding_dim, padding_idx=0)
        self.item_emb = nn.Embedding(n_item_feats + 1, embedding_dim, padding_idx=0)
        self.seq_emb = nn.Embedding(n_seq_feats + 1, embedding_dim, padding_idx=0)
        
        self.seq_encoder = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=False
        )
        
        self.user_proj = nn.Linear(embedding_dim * 46, hidden_dim)
        self.item_proj = nn.Linear(embedding_dim * 14, hidden_dim)
        self.seq_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # 预测层
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, user_feats, item_feats, seq_feats):
        # 用户特征
        user_embs = self.user_emb(user_feats)  # [batch, 46, emb]
        user_embs = user_embs.reshape(user_embs.size(0), -1)  # [batch, 46*emb]
        user_out = self.user_proj(user_embs)  # [batch, hidden]
        
        # 物品特征
        item_embs = self.item_emb(item_feats)  # [batch, 14, emb]
        item_embs = item_embs.reshape(item_embs.size(0), -1)  # [batch, 14*emb]
        item_out = self.item_proj(item_embs)  # [batch, hidden]
        
        # 序列特征
        seq_embs = self.seq_emb(seq_feats)  # [batch, seq_len, emb]
        _, seq_hidden = self.seq_encoder(seq_embs)  # [1, batch, hidden]
        seq_out = self.seq_proj(seq_hidden.squeeze(0))  # [batch, hidden]
        
        # 拼接所有特征
        combined = torch.cat([user_out, item_out, seq_out], dim=1)  # [batch, hidden*3]
        
        # 预测
        pred = self.predictor(combined)  # [batch, 1]
        return pred.squeeze(1)


# ==================== 训练函数 ====================
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in dataloader:
        user_feats = batch['user_feats'].to(device)
        item_feats = batch['item_feats'].to(device)
        seq_feats = batch['seq_feats'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        preds = model(user_feats, item_feats, seq_feats)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    auc = roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.5
    return avg_loss, auc


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            user_feats = batch['user_feats'].to(device)
            item_feats = batch['item_feats'].to(device)
            seq_feats = batch['seq_feats'].to(device)
            labels = batch['label'].to(device)
            
            preds = model(user_feats, item_feats, seq_feats)
            loss = criterion(preds, labels)
            
            total_loss += loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    auc = roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.5
    return avg_loss, auc


# ==================== 主函数 ====================
def main():
    # 配置
    DATA_PATH = '../data/taac2026_demo.parquet'
    BATCH_SIZE = 64
    EPOCHS = 10
    LR = 0.001
    DEVICE = 'cpu'  # GPU内存不足，用CPU
    
    print(f'Device: {DEVICE}')
    
    # 加载数据
    print('Loading data...')
    df = pd.read_parquet(DATA_PATH)
    print(f'Data shape: {df.shape}')
    print(f'Label distribution:\n{df["label_type"].value_counts().sort_index()}')
    
    # 划分训练集和验证集
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label_type'])
    print(f'Train: {len(train_df)}, Val: {len(val_df)}')
    
    # 创建数据集
    train_dataset = CVRDataset(train_df, None, None)
    val_dataset = CVRDataset(val_df, None, None)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # 获取特征ID范围
    n_user_feats = max(train_dataset.user_feats.max(), val_dataset.user_feats.max()) + 1
    n_item_feats = max(train_dataset.item_feats.max(), val_dataset.item_feats.max()) + 1
    n_seq_feats = max(train_dataset.seq_feats.max(), val_dataset.seq_feats.max()) + 1
    print(f'Feature ranges: user={n_user_feats}, item={n_item_feats}, seq={n_seq_feats}')
    
    model = SimpleCVRModel(
        n_user_feats=n_user_feats,
        n_item_feats=n_item_feats,
        n_seq_feats=n_seq_feats,
        embedding_dim=16,
        hidden_dim=64
    ).to(DEVICE)
    
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')
    
    # 训练
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    best_auc = 0
    for epoch in range(EPOCHS):
        train_loss, train_auc = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_auc = evaluate(model, val_loader, criterion, DEVICE)
        
        print(f'Epoch {epoch+1}/{EPOCHS}: '
              f'Train Loss={train_loss:.4f} AUC={train_auc:.4f} | '
              f'Val Loss={val_loss:.4f} AUC={val_auc:.4f}')
        
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), '../models/best_cvr_model.pt')
            print(f'  -> Saved best model with AUC={best_auc:.4f}')
    
    print(f'\nBest Val AUC: {best_auc:.4f}')
    print('Training complete!')


if __name__ == '__main__':
    main()
