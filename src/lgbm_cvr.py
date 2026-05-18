"""
2026腾讯广告算法大赛 - CVR预测 LightGBM方案
目标：快速达到 Val AUC >= 0.7

核心改进：
1. 修复LabelEncoder致命bug（train/val编码不一致）
2. 使用LightGBM（小数据集最优选择）
3. 完整特征工程（标量、数组、序列、稠密特征）
4. 5折交叉验证
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')


def extract_features(df):
    """从原始数据提取所有可用特征"""
    features = {}

    # === 1. 标量用户特征 ===
    for col in df.columns:
        if col.startswith('user_int_feats_'):
            try:
                vals = df[col].apply(lambda x: x if isinstance(x, (int, float)) else 0).fillna(-1).values
                features[col] = vals.astype(float)
            except:
                pass

    # === 2. 标量物品特征 ===
    for col in df.columns:
        if col.startswith('item_int_feats_'):
            try:
                vals = df[col].apply(lambda x: x if isinstance(x, (int, float)) else 0).fillna(-1).values
                features[col] = vals.astype(float)
            except:
                pass

    # === 3. 数组特征：提取统计量 ===
    for col in df.columns:
        if col.startswith(('user_int_feats_', 'item_int_feats_')):
            if df[col].dtype == object:
                arrs = df[col].values
                lens = np.array([len(a) if isinstance(a, list) else 0 for a in arrs])
                means = np.array([np.mean(a) if isinstance(a, list) and len(a) > 0 else 0 for a in arrs])
                maxs = np.array([np.max(a) if isinstance(a, list) and len(a) > 0 else 0 for a in arrs])
                mins = np.array([np.min(a) if isinstance(a, list) and len(a) > 0 else 0 for a in arrs])
                features[f'{col}_len'] = lens
                features[f'{col}_mean'] = means
                features[f'{col}_max'] = maxs
                features[f'{col}_min'] = mins

    # === 4. 稠密特征：提取统计量 ===
    for col in df.columns:
        if col.startswith('user_dense_feats_'):
            arrs = df[col].values
            means = np.array([np.mean(a) if isinstance(a, list) and len(a) > 0 else 0 for a in arrs])
            stds = np.array([np.std(a) if isinstance(a, list) and len(a) > 0 else 0 for a in arrs])
            maxs = np.array([np.max(a) if isinstance(a, list) and len(a) > 0 else 0 for a in arrs])
            features[f'{col}_mean'] = means
            features[f'{col}_std'] = stds
            features[f'{col}_max'] = maxs

    # === 5. 序列特征统计 ===
    for domain in ['domain_a_seq', 'domain_b_seq', 'domain_c_seq', 'domain_d_seq']:
        domain_cols = [c for c in df.columns if c.startswith(domain)]
        if domain_cols:
            first_col = domain_cols[0]
            arrs = df[first_col].values
            seq_lens = np.array([len(a) if isinstance(a, list) else 0 for a in arrs])
            non_zero = np.array([np.count_nonzero(a) if isinstance(a, list) else 0 for a in arrs])
            features[f'{domain}_len'] = seq_lens
            features[f'{domain}_nonzero'] = non_zero

    # === 6. user_id 和 item_id ===
    features['user_id'] = df['user_id'].values
    features['item_id'] = df['item_id'].values

    # === 7. 时间特征 ===
    if 'timestamp' in df.columns:
        ts = pd.to_datetime(df['timestamp'], unit='s')
        features['hour'] = ts.dt.hour.values
        features['dayofweek'] = ts.dt.dayofweek.values
        features['is_weekend'] = (ts.dt.dayofweek >= 5).astype(int).values

    # === 8. Target Encoding (贝叶斯平滑) ===
    labels = (df['label_type'] == 2).astype(int).values
    global_mean = labels.mean()
    for col in ['user_id', 'item_id']:
        if col in df.columns:
            stats = pd.DataFrame({col: df[col], 'label': labels})
            agg = stats.groupby(col)['label'].agg(['mean', 'count'])
            smoothing = 20
            smoothed = (agg['count'] * agg['mean'] + smoothing * global_mean) / (agg['count'] + smoothing)
            features[f'{col}_te'] = df[col].map(smoothed).fillna(global_mean).values

    return pd.DataFrame(features)


def train_lgbm_cv(df, n_splits=5):
    """5折交叉验证训练LightGBM"""

    # 提取标签
    labels = (df['label_type'] == 2).astype(int).values

    # 提取特征
    print("Extracting features...")
    feat_df = extract_features(df)
    print(f"Feature shape: {feat_df.shape}")

    # 对user_id和item_id做全局编码（修复致命bug）
    for col in ['user_id', 'item_id']:
        le = LabelEncoder()
        feat_df[col] = le.fit_transform(feat_df[col].astype(str))

    # 确定分类特征（unique值较少的）
    cat_cols = []
    for c in feat_df.columns:
        if feat_df[c].nunique() < 50 and '_len' not in c and '_mean' not in c and '_std' not in c and '_max' not in c and '_min' not in c:
            cat_cols.append(c)

    print(f"Total features: {feat_df.shape[1]}")
    print(f"Categorical features: {len(cat_cols)}")
    print(f"Label distribution: 0={sum(labels==0)}, 1={sum(labels==1)}")
    print(f"Positive rate: {labels.mean():.2%}")

    # 交叉验证
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(df))

    params = {
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': 6,
        'min_child_samples': 20,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'verbose': -1,
        'is_unbalance': True,
        'seed': 42,
    }

    print("\n=== Training ===")
    for fold, (train_idx, val_idx) in enumerate(skf.split(feat_df, labels)):
        X_train = feat_df.iloc[train_idx]
        y_train = labels[train_idx]
        X_val = feat_df.iloc[val_idx]
        y_val = labels[val_idx]

        train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols)
        val_data = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_cols, reference=train_data)

        model = lgb.train(
            params,
            train_data,
            num_boost_round=500,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
        )

        val_preds = model.predict(X_val)
        oof_preds[val_idx] = val_preds

        fold_auc = roc_auc_score(y_val, val_preds)
        print(f"Fold {fold+1} AUC: {fold_auc:.4f}")

    overall_auc = roc_auc_score(labels, oof_preds)
    print(f"\n{'='*40}")
    print(f"OOF AUC: {overall_auc:.4f}")
    print(f"{'='*40}")

    # 特征重要性
    print("\n=== Top 20 Features ===")
    importance = model.feature_importance(importance_type='gain')
    feature_names = feat_df.columns
    imp_df = pd.DataFrame({'feature': feature_names, 'importance': importance})
    imp_df = imp_df.sort_values('importance', ascending=False).head(20)
    for _, row in imp_df.iterrows():
        print(f"  {row['feature']}: {row['importance']:.0f}")

    return overall_auc, model


if __name__ == '__main__':
    DATA_PATH = '../data/taac2026_demo.parquet'

    print("Loading data...")
    df = pd.read_parquet(DATA_PATH)
    print(f"Data shape: {df.shape}")

    auc, model = train_lgbm_cv(df)
    print(f"\nFinal OOF AUC: {auc:.4f}")

    if auc >= 0.7:
        print("SUCCESS: AUC >= 0.7 achieved!")
    else:
        print(f"Need more optimization. Current: {auc:.4f}, Target: 0.7")
