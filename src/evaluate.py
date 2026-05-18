"""
ESMM模型评估工具
"""

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def compute_ctr_auc(labels, preds):
    """计算CTR任务的AUC"""
    if len(set(labels)) < 2:
        return 0.5
    return roc_auc_score(labels, preds)


def compute_cvr_auc(labels, preds):
    """
    计算CVR任务的AUC
    只在点击样本(label_type>=1)中计算CVR的AUC
    """
    if len(set(labels)) < 2:
        return 0.5
    return roc_auc_score(labels, preds)


def compute_calibration(preds, labels, n_bins=10):
    """计算预测校准度: 预测均值 vs 实际均值"""
    pred_mean = np.mean(preds)
    label_mean = np.mean(labels)
    ratio = pred_mean / max(label_mean, 1e-8)
    return {
        'pred_mean': pred_mean,
        'label_mean': label_mean,
        'calibration_ratio': ratio,
    }


def evaluate_esmm(model, dataloader, device='cpu'):
    """
    ESMM模型全面评估

    返回:
        metrics: dict with keys:
            ctr_auc, cvr_auc, ctcvr_auc,
            ctr_loss, cvr_loss,
            ctr_calibration, cvr_calibration
    """
    model.eval()
    all_ctr_preds, all_ctr_labels = [], []
    all_cvr_preds, all_cvr_labels = [], []
    all_ctcvr_preds, all_ctcvr_labels = [], []
    total_ctr_loss, total_cvr_loss = 0.0, 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            batch_device = {k: v.to(device) for k, v in batch.items()}

            p_ctr, p_cvr, p_ctcvr = model(batch_device)

            ctr_labels = batch['ctr_label'].numpy()
            cvr_labels = batch['cvr_label'].numpy()
            ctr_preds = p_ctr.cpu().numpy()
            cvr_preds = p_cvr.cpu().numpy()
            ctcvr_preds = p_ctcvr.cpu().numpy()

            all_ctr_preds.extend(ctr_preds)
            all_ctr_labels.extend(ctr_labels)
            all_ctcvr_preds.extend(ctcvr_preds)
            all_ctcvr_labels.extend(ctr_labels)

            # CVR只在点击样本中评估
            click_mask = ctr_labels >= 1
            if click_mask.sum() > 0:
                all_cvr_preds.extend(cvr_preds[click_mask])
                all_cvr_labels.extend(cvr_labels[click_mask])

            # CTR loss
            ctr_loss = torch.nn.functional.binary_cross_entropy(
                p_ctr, batch['ctr_label'].to(device), reduction='mean'
            )
            total_ctr_loss += ctr_loss.item()

            # CVR loss (仅点击样本)
            if click_mask.sum() > 0:
                cvr_loss = torch.nn.functional.binary_cross_entropy(
                    p_cvr[torch.tensor(click_mask)], 
                    batch['cvr_label'][click_mask].to(device),
                    reduction='mean'
                )
                total_cvr_loss += cvr_loss.item()

            n_batches += 1

    metrics = {
        'ctr_auc': compute_ctr_auc(all_ctr_labels, all_ctr_preds),
        'cvr_auc': compute_cvr_auc(all_cvr_labels, all_cvr_preds) if all_cvr_labels else 0.5,
        'ctcvr_auc': compute_ctr_auc(all_ctcvr_labels, all_ctcvr_preds),
        'ctr_loss': total_ctr_loss / max(n_batches, 1),
        'cvr_loss': total_cvr_loss / max(n_batches, 1),
        'ctr_calibration': compute_calibration(all_ctr_preds, all_ctr_labels),
        'cvr_calibration': compute_calibration(all_cvr_preds, all_cvr_labels) if all_cvr_labels else None,
        'n_click_samples': len(all_cvr_labels),
        'n_total_samples': len(all_ctr_labels),
    }

    return metrics


def format_metrics(metrics, epoch=None):
    """格式化打印评估指标"""
    prefix = f"Epoch {epoch} | " if epoch is not None else ""
    lines = [
        f"{prefix}CTR AUC: {metrics['ctr_auc']:.4f}",
        f"{prefix}CVR AUC: {metrics['cvr_auc']:.4f} (click samples: {metrics['n_click_samples']})",
        f"{prefix}CTCVR AUC: {metrics['ctcvr_auc']:.4f}",
        f"{prefix}CTR Loss: {metrics['ctr_loss']:.4f}",
        f"{prefix}CVR Loss: {metrics['cvr_loss']:.4f}",
    ]
    if metrics['ctr_calibration']:
        cal = metrics['ctr_calibration']
        lines.append(f"{prefix}CTR Calibration: pred={cal['pred_mean']:.4f}, "
                     f"actual={cal['label_mean']:.4f}, ratio={cal['calibration_ratio']:.3f}")
    if metrics['cvr_calibration']:
        cal = metrics['cvr_calibration']
        lines.append(f"{prefix}CVR Calibration: pred={cal['pred_mean']:.4f}, "
                     f"actual={cal['label_mean']:.4f}, ratio={cal['calibration_ratio']:.3f}")
    return "\n".join(lines)
