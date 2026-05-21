"""
ESMM V2 训练脚本

训练策略优化:
1. Focal Loss: 处理类别不平衡
2. Label Smoothing: 防止过拟合
3. Cosine Annealing with Warmup: 更好的学习率调度
4. 梯度裁剪: 防止梯度爆炸
5. Self-supervised Auxiliary Loss: 缓解标签稀疏
"""

import os
import sys
import time
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from feature_engineering_v2 import prepare_data_v2, TAACDatasetV2
from esmm_din_v2 import ESMM_DIN_V2
from evaluate import evaluate_esmm, format_metrics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser(description='ESMM+DIN+DCN+Transformer V2')
    parser.add_argument('--data_path', type=str, default='../data/taac2026_demo.parquet')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--warmup_epochs', type=int, default=3)
    parser.add_argument('--embedding_dim', type=int, default=32)
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[256, 128, 64])
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--l2_reg', type=float, default=1e-4)
    parser.add_argument('--max_seq_len', type=int, default=50)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save_dir', type=str, default='../models')
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--focal_gamma', type=float, default=2.0, help='Focal loss gamma')
    parser.add_argument('--focal_alpha', type=float, default=0.25, help='Focal loss alpha')
    parser.add_argument('--label_smoothing', type=float, default=0.05, help='Label smoothing epsilon')
    parser.add_argument('--ssl_weight', type=float, default=0.1, help='Self-supervised loss weight')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping max norm')
    parser.add_argument('--num_cross_layers', type=int, default=3, help='DCN cross layers')
    parser.add_argument('--num_transformer_layers', type=int, default=1, help='Transformer layers')
    parser.add_argument('--num_transformer_heads', type=int, default=4, help='Transformer heads')
    parser.add_argument('--enable_ssl', action='store_true', default=True, help='Enable self-supervised loss')
    return parser.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FocalLoss(nn.Module):
    """Focal Loss: FL(p_t) = -alpha_t * (1-p_t)^gamma * log(p_t)"""

    def __init__(self, gamma=2.0, alpha=0.25, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, pred, target):
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.where(target == 1, pred, 1 - pred)
        focal_weight = (1 - pt) ** self.gamma
        alpha_weight = torch.where(target == 1, self.alpha, 1 - self.alpha)
        loss = alpha_weight * focal_weight * bce
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class LabelSmoothingBCELoss(nn.Module):
    """BCE with Label Smoothing: y_smooth = y * (1 - eps) + 0.5 * eps"""

    def __init__(self, epsilon=0.05, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, pred, target):
        target_smooth = target * (1 - self.epsilon) + 0.5 * self.epsilon
        loss = F.binary_cross_entropy(pred, target_smooth, reduction=self.reduction)
        return loss


class WarmupCosineScheduler:
    """Cosine Annealing with Linear Warmup"""

    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            warmup_factor = (epoch + 1) / self.warmup_epochs
            for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                group['lr'] = base_lr * warmup_factor
        else:
            progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            cosine_factor = 0.5 * (1 + np.cos(np.pi * progress))
            for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                group['lr'] = self.min_lr + (base_lr - self.min_lr) * cosine_factor

    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]


def compute_esmm_loss_v2(p_ctr, p_cvr, p_ctcvr, ctr_label, cvr_label, device,
                         focal_loss=None, label_smoothing_loss=None):
    """ESMM损失函数V2: 支持Focal Loss和Label Smoothing"""
    if focal_loss is not None:
        loss_ctr = focal_loss(p_ctr, ctr_label)
        loss_ctcvr = focal_loss(p_ctcvr, cvr_label)
    elif label_smoothing_loss is not None:
        loss_ctr = label_smoothing_loss(p_ctr, ctr_label)
        loss_ctcvr = label_smoothing_loss(p_ctcvr, cvr_label)
    else:
        bce = nn.BCELoss(reduction='mean')
        loss_ctr = bce(p_ctr, ctr_label)
        loss_ctcvr = bce(p_ctcvr, cvr_label)

    click_mask = ctr_label >= 1
    if click_mask.sum() > 0:
        if focal_loss is not None:
            cvr_loss = focal_loss(p_cvr[click_mask], cvr_label[click_mask])
        elif label_smoothing_loss is not None:
            cvr_loss = label_smoothing_loss(p_cvr[click_mask], cvr_label[click_mask])
        else:
            cvr_loss = F.binary_cross_entropy(p_cvr[click_mask], cvr_label[click_mask])
    else:
        cvr_loss = torch.tensor(0.0, device=device)

    return loss_ctr + loss_ctcvr + cvr_loss


class EarlyStopping:
    def __init__(self, patience=8, min_delta=1e-4):
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


def evaluate_esmm_v2(model, dataloader, device='cpu'):
    """评估ESMM V2模型 (忽略ssl_logits输出)"""
    model.eval()
    all_ctr_preds, all_ctr_labels = [], []
    all_cvr_preds, all_cvr_labels = [], []
    all_ctcvr_preds, all_ctcvr_labels = [], []
    total_ctr_loss, total_cvr_loss = 0.0, 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            batch_device = {k: v.to(device) for k, v in batch.items()}
            p_ctr, p_cvr, p_ctcvr, _ = model(batch_device)

            ctr_labels = batch['ctr_label'].numpy()
            cvr_labels = batch['cvr_label'].numpy()
            ctr_preds = p_ctr.cpu().numpy()
            cvr_preds = p_cvr.cpu().numpy()
            ctcvr_preds = p_ctcvr.cpu().numpy()

            all_ctr_preds.extend(ctr_preds)
            all_ctr_labels.extend(ctr_labels)
            all_ctcvr_preds.extend(ctcvr_preds)
            all_ctcvr_labels.extend(ctr_labels)

            click_mask = ctr_labels >= 1
            if click_mask.sum() > 0:
                all_cvr_preds.extend(cvr_preds[click_mask])
                all_cvr_labels.extend(cvr_labels[click_mask])

            ctr_loss = F.binary_cross_entropy(p_ctr, batch['ctr_label'].to(device))
            total_ctr_loss += ctr_loss.item()

            if click_mask.sum() > 0:
                cvr_loss = F.binary_cross_entropy(
                    p_cvr[torch.tensor(click_mask)],
                    batch['cvr_label'][click_mask].to(device)
                )
                total_cvr_loss += cvr_loss.item()

            n_batches += 1

    from sklearn.metrics import roc_auc_score
    metrics = {
        'ctr_auc': roc_auc_score(all_ctr_labels, all_ctr_preds) if len(set(all_ctr_labels)) > 1 else 0.5,
        'cvr_auc': roc_auc_score(all_cvr_labels, all_cvr_preds) if len(all_cvr_labels) > 1 and len(set(all_cvr_labels)) > 1 else 0.5,
        'ctcvr_auc': roc_auc_score(all_ctcvr_labels, all_ctcvr_preds) if len(set(all_ctcvr_labels)) > 1 else 0.5,
        'ctr_loss': total_ctr_loss / max(n_batches, 1),
        'cvr_loss': total_cvr_loss / max(n_batches, 1),
        'ctr_calibration': {'pred_mean': np.mean(all_ctr_preds), 'label_mean': np.mean(all_ctr_labels), 'calibration_ratio': np.mean(all_ctr_preds) / max(np.mean(all_ctr_labels), 1e-8)},
        'cvr_calibration': {'pred_mean': np.mean(all_cvr_preds), 'label_mean': np.mean(all_cvr_labels), 'calibration_ratio': np.mean(all_cvr_preds) / max(np.mean(all_cvr_labels), 1e-8)} if all_cvr_labels else None,
        'n_click_samples': len(all_cvr_labels),
        'n_total_samples': len(all_ctr_labels),
    }
    return metrics


def train_one_epoch_v2(model, dataloader, optimizer, device, focal_loss=None,
                       label_smoothing_loss=None, ssl_weight=0.1, grad_clip=1.0):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        batch_device = {k: v.to(device) for k, v in batch.items()}

        p_ctr, p_cvr, p_ctcvr, ssl_logits = model(batch_device)

        loss = compute_esmm_loss_v2(
            p_ctr, p_cvr, p_ctcvr,
            batch_device['ctr_label'], batch_device['cvr_label'], device,
            focal_loss=focal_loss, label_smoothing_loss=label_smoothing_loss
        )

        if ssl_logits is not None and ssl_weight > 0:
            seq_a = batch_device['seq_a']
            mask = batch_device['seq_mask']
            valid_positions = (seq_a > 0) & (mask.bool())
            if valid_positions.sum() > 0:
                ssl_loss = F.cross_entropy(ssl_logits, seq_a[:, 0].clamp(0, ssl_logits.size(-1) - 1))
                loss = loss + ssl_weight * ssl_loss

        l2_loss = model.get_l2_reg_loss()
        loss = loss + l2_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
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

    train_dataset, val_dataset, fp = prepare_data_v2(
        args.data_path, max_seq_len=args.max_seq_len
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    user_scalar_dims = [fp.scalar_dims.get(c, 2) for c in
                        ['user_int_feats_1', 'user_int_feats_3', 'user_int_feats_4',
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
                         'user_int_feats_108', 'user_int_feats_109'] if c in fp.scalar_dims]
    item_scalar_dims = [fp.scalar_dims.get(c, 2) for c in
                        ['item_int_feats_5', 'item_int_feats_6', 'item_int_feats_7',
                         'item_int_feats_8', 'item_int_feats_9', 'item_int_feats_10',
                         'item_int_feats_12', 'item_int_feats_13', 'item_int_feats_16',
                         'item_int_feats_81', 'item_int_feats_83', 'item_int_feats_84',
                         'item_int_feats_85'] if c in fp.scalar_dims]
    user_list_dims = [fp.scalar_dims.get(c, 100) for c in
                      ['user_int_feats_15', 'user_int_feats_60', 'user_int_feats_62',
                       'user_int_feats_63', 'user_int_feats_64', 'user_int_feats_65',
                       'user_int_feats_66', 'user_int_feats_80',
                       'user_int_feats_89', 'user_int_feats_90', 'user_int_feats_91'] if c in fp.scalar_dims]
    item_list_dims = [fp.scalar_dims.get(c, 100) for c in
                      ['item_int_feats_11'] if c in fp.scalar_dims]

    n_stat_feats = len(TAACDatasetV2.STAT_FEATS)
    n_time_feats = len(TAACDatasetV2.TIME_FEATS)
    n_seq_stat_feats = len(TAACDatasetV2.SEQ_STAT_FEATS)
    n_cross_feats = len(TAACDatasetV2.CROSS_FEATS)

    model = ESMM_DIN_V2(
        n_user_scalar_feats=len(user_scalar_dims),
        n_item_scalar_feats=len(item_scalar_dims),
        n_user_list_feats=len(user_list_dims),
        n_item_list_feats=len(item_list_dims),
        user_scalar_dims=user_scalar_dims,
        item_scalar_dims=item_scalar_dims,
        user_list_dims=user_list_dims,
        item_list_dims=item_list_dims,
        user_dense_dim=train_dataset.user_dense_feats.shape[1],
        n_stat_feats=n_stat_feats,
        n_time_feats=n_time_feats,
        n_seq_stat_feats=n_seq_stat_feats,
        n_cross_feats=n_cross_feats,
        cross_hash_bucket=5001,
        embedding_dim=args.embedding_dim,
        seq_hash_bucket=10001,
        hidden_dims=args.hidden_dims,
        dropout_rate=args.dropout,
        l2_reg=args.l2_reg,
        num_cross_layers=args.num_cross_layers,
        num_transformer_layers=args.num_transformer_layers,
        num_transformer_heads=args.num_transformer_heads,
        enable_ssl=args.enable_ssl,
    ).to(args.device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.l2_reg)
    scheduler = WarmupCosineScheduler(optimizer, args.warmup_epochs, args.epochs, args.min_lr)
    early_stopping = EarlyStopping(patience=args.patience, min_delta=1e-4)

    focal_loss = FocalLoss(gamma=args.focal_gamma, alpha=args.focal_alpha)
    label_smoothing_loss = LabelSmoothingBCELoss(epsilon=args.label_smoothing)

    best_auc = 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        scheduler.step(epoch - 1)
        train_loss = train_one_epoch_v2(
            model, train_loader, optimizer, args.device,
            focal_loss=focal_loss,
            label_smoothing_loss=label_smoothing_loss,
            ssl_weight=args.ssl_weight,
            grad_clip=args.grad_clip,
        )

        val_metrics = evaluate_esmm_v2(model, val_loader, device=args.device)
        elapsed = time.time() - t0

        lr = scheduler.get_last_lr()[0]
        logger.info(f"Epoch {epoch}/{args.epochs} ({elapsed:.1f}s) lr={lr:.2e}")
        logger.info(f"  Train Loss: {train_loss:.4f}")
        logger.info(format_metrics(val_metrics))

        cvr_auc = float(val_metrics['cvr_auc'])
        ctcvr_auc = float(val_metrics['ctcvr_auc'])

        if ctcvr_auc > best_auc:
            best_auc = ctcvr_auc
            save_path = os.path.join(args.save_dir, f'global_step{epoch}.best_auc={best_auc:.4f}')
            os.makedirs(save_path, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_auc': best_auc,
                'metrics': val_metrics,
                'args': vars(args),
            }, os.path.join(save_path, 'model.pt'))
            logger.info(f"  -> Saved best model: CTCVR AUC={best_auc:.4f}")

        early_stopping(ctcvr_auc)
        if early_stopping.should_stop:
            logger.info(f"Early stopping at epoch {epoch}")
            break

    logger.info(f"Training complete. Best CTCVR AUC: {best_auc:.4f}")


if __name__ == '__main__':
    main()
