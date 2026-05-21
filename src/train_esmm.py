"""
ESMM + DIN 训练脚本

ESMM损失: L = L_ctcvr(all) + L_ctr(all) + L_cvr(click_only)
- L_ctcvr = BCE(pCTR * pCVR, conversion_label) (conversion样本)
- L_ctr = BCE(pCTR, click_label) (所有样本)
- L_cvr = BCE(pCVR, conversion_label) (仅click样本)
"""

import os
import sys
import time
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from feature_engineering import (
    prepare_data,
    USER_SCALAR_INT, ITEM_SCALAR_INT, USER_LIST_INT, ITEM_LIST_INT,
    HASH_BUCKET_SIZE,
)
from esmm_din_model import ESMM_DIN
from evaluate import evaluate_esmm, format_metrics


class FocalLoss(nn.Module):
    """
    Focal Loss - 处理类别不平衡
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.where(targets == 1, inputs, 1 - inputs)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        loss = focal_weight * bce
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser(description='ESMM+DIN CVR预测')
    parser.add_argument('--data_path', type=str, default='../data/taac2026_demo.parquet')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--embedding_dim', type=int, default=16)
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[128, 64, 32])
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--l2_reg', type=float, default=1e-5)
    parser.add_argument('--max_seq_len', type=int, default=50)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save_dir', type=str, default='../models')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_esmm_loss(p_ctr, p_cvr, p_ctcvr, ctr_label, cvr_label, device):
    """
    ESMM损失函数

    全体样本: L_ctr = BCE(pCTR, click_label)
    转化样本: L_ctcvr = BCE(pCTR * pCVR, 1)
    非转化但点击样本: L_ctcvr = BCE(pCTR * pCVR, 0)
    点击样本额外: L_cvr = BCE(pCVR, conversion_label)
    """
    bce = nn.BCELoss(reduction='mean')

    # CTR loss: 全体样本
    loss_ctr = bce(p_ctr, ctr_label)

    # CVR loss: 仅在点击样本中计算
    click_mask = ctr_label >= 1
    if click_mask.sum() > 0:
        cvr_loss = bce(p_cvr[click_mask], cvr_label[click_mask])
    else:
        cvr_loss = torch.tensor(0.0, device=device)

    # CTCVR loss: 全体样本 (conversion=1的作为正样本)
    ctcvr_label = cvr_label.clone()
    loss_ctcvr = bce(p_ctcvr, ctcvr_label)

    return loss_ctr + loss_ctcvr + cvr_loss


class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_score = score
            self.counter = 0


def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        batch_device = {k: v.to(device) for k, v in batch.items()}

        p_ctr, p_cvr, p_ctcvr = model(batch_device)

        loss = compute_esmm_loss(
            p_ctr, p_cvr, p_ctcvr,
            batch_device['ctr_label'], batch_device['cvr_label'], device
        )

        l2_loss = model.get_l2_reg_loss()
        loss = loss + l2_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def main():
    args = get_args()
    set_seed(args.seed)
    logger.info(f"Config: {vars(args)}")
    logger.info(f"Device: {args.device}")

    os.makedirs(args.save_dir, exist_ok=True)

    # 数据准备
    train_dataset, val_dataset, fp = prepare_data(
        args.data_path, max_seq_len=args.max_seq_len
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    # 模型初始化
    user_scalar_dims = [fp.scalar_dims.get(c, 2) for c in USER_SCALAR_INT if c in fp.scalar_dims]
    item_scalar_dims = [fp.scalar_dims.get(c, 2) for c in ITEM_SCALAR_INT if c in fp.scalar_dims]
    user_list_dims = [fp.scalar_dims.get(c, 100) for c in USER_LIST_INT if c in fp.scalar_dims]
    item_list_dims = [fp.scalar_dims.get(c, 100) for c in ITEM_LIST_INT if c in fp.scalar_dims]

    model = ESMM_DIN(
        n_user_scalar_feats=len(user_scalar_dims),
        n_item_scalar_feats=len(item_scalar_dims),
        n_user_list_feats=len(user_list_dims),
        n_item_list_feats=len(item_list_dims),
        user_scalar_dims=user_scalar_dims,
        item_scalar_dims=item_scalar_dims,
        user_list_dims=user_list_dims,
        item_list_dims=item_list_dims,
        user_dense_dim=train_dataset.user_dense_feats.shape[1],
        embedding_dim=args.embedding_dim,
        seq_hash_bucket=HASH_BUCKET_SIZE + 1,
        hidden_dims=args.hidden_dims,
        dropout_rate=args.dropout,
        l2_reg=args.l2_reg,
    ).to(args.device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model params: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)

    # Warmup + Cosine Annealing
    warmup_steps = max(1, args.epochs // 5)
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - warmup_steps, eta_min=args.min_lr)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])

    early_stopping = EarlyStopping(patience=args.patience, min_delta=1e-4)

    best_auc = 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, args.device)
        scheduler.step()

        val_metrics = evaluate_esmm(model, val_loader, device=args.device)
        elapsed = time.time() - t0

        lr = scheduler.get_last_lr()[0]
        logger.info(f"Epoch {epoch}/{args.epochs} ({elapsed:.1f}s) lr={lr:.2e}")
        logger.info(f"  Train Loss: {train_loss:.4f}")
        logger.info(format_metrics(val_metrics))

        # 以CVR AUC为主指标 (竞赛目标)
        cvr_auc = float(val_metrics['cvr_auc'])
        ctcvr_auc = float(val_metrics['ctcvr_auc'])

        if ctcvr_auc > best_auc:
            best_auc = ctcvr_auc
            save_path = os.path.join(args.save_dir, 'best_esmm_din.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_auc': best_auc,
                'metrics': val_metrics,
                'args': vars(args),
            }, save_path)
            logger.info(f"  -> Saved best model: CTCVR AUC={best_auc:.4f}")

        early_stopping(ctcvr_auc)
        if early_stopping.should_stop:
            logger.info(f"Early stopping at epoch {epoch}")
            break

    logger.info(f"Training complete. Best CTCVR AUC: {best_auc:.4f}")


if __name__ == '__main__':
    main()
