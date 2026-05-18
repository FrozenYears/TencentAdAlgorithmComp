"""
2026腾讯广告算法大赛 - CVR预测 集成方案
目标：通过LightGBM + XGBoost + CatBoost集成突破AUC 0.7
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import warnings
warnings.filterwarnings('ignore')


def extract_features(df):
    features = {}
    for col in df.columns:
        if col.startswith('user_int_feats_'):
            try:
                vals = df[col].apply(lambda x: x if isinstance(x, (int, float)) else 0).fillna(-1).values
                features[col] = vals.astype(float)
            except:
                pass
    for col in df.columns:
        if col.startswith('item_int_feats_'):
            try:
                vals = df[col].apply(lambda x: x if isinstance(x, (int, float)) else 0).fillna(-1).values
                features[col] = vals.astype(float)
            except:
                pass
    for col in df.columns:
        if col.startswith(('user_int_feats_', 'item_int_feats_')):
            if df[col].dtype == object:
                arrs = df[col].values
                lens = np.array([len(a) if isinstance(a, list) else 0 for a in arrs])
                means = np.array([np.mean(a) if isinstance(a, list) and len(a) > 0 else 0 for a in arrs])
                maxs = np.array([np.max(a) if isinstance(a, list) and len(a) > 0 else 0 for a in arrs])
                features[f'{col}_len'] = lens
                features[f'{col}_mean'] = means
                features[f'{col}_max'] = maxs
    for col in df.columns:
        if col.startswith('user_dense_feats_'):
            arrs = df[col].values
            means = np.array([np.mean(a) if isinstance(a, list) and len(a) > 0 else 0 for a in arrs])
            stds = np.array([np.std(a) if isinstance(a, list) and len(a) > 0 else 0 for a in arrs])
            features[f'{col}_mean'] = means
            features[f'{col}_std'] = stds
    for domain in ['domain_a_seq', 'domain_b_seq', 'domain_c_seq', 'domain_d_seq']:
        domain_cols = [c for c in df.columns if c.startswith(domain)]
        if domain_cols:
            first_col = domain_cols[0]
            arrs = df[first_col].values
            seq_lens = np.array([len(a) if isinstance(a, list) else 0 for a in arrs])
            features[f'{domain}_len'] = seq_lens
    features['user_id'] = df['user_id'].values
    features['item_id'] = df['item_id'].values
    if 'timestamp' in df.columns:
        ts = pd.to_datetime(df['timestamp'], unit='s')
        features['hour'] = ts.dt.hour.values
        features['dayofweek'] = ts.dt.dayofweek.values
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


def train_ensemble(df, n_splits=5):
    labels = (df['label_type'] == 2).astype(int).values
    feat_df = extract_features(df)
    for col in ['user_id', 'item_id']:
        le = LabelEncoder()
        feat_df[col] = le.fit_transform(feat_df[col].astype(str))
    cat_cols = [c for c in feat_df.columns if feat_df[c].nunique() < 50 and '_len' not in c and '_mean' not in c and '_std' not in c and '_max' not in c]
    print(f"Features: {feat_df.shape[1]}, Labels: 0={sum(labels==0)}, 1={sum(labels==1)}")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_lgb = np.zeros(len(df))
    oof_xgb = np.zeros(len(df))
    oof_cb = np.zeros(len(df))
    lgb_params = {'objective': 'binary', 'metric': 'auc', 'learning_rate': 0.05, 'num_leaves': 31, 'max_depth': 6, 'min_child_samples': 20, 'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5, 'reg_alpha': 0.1, 'reg_lambda': 1.0, 'verbose': -1, 'is_unbalance': True, 'seed': 42}
    xgb_params = {'objective': 'binary:logistic', 'eval_metric': 'auc', 'learning_rate': 0.05, 'max_depth': 6, 'min_child_weight': 20, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 0.1, 'reg_lambda': 1.0, 'scale_pos_weight': sum(labels==0)/sum(labels==1), 'seed': 42, 'verbosity': 0}
    print("\n=== Training Ensemble ===")
    for fold, (train_idx, val_idx) in enumerate(skf.split(feat_df, labels)):
        X_train, X_val = feat_df.iloc[train_idx], feat_df.iloc[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]
        train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols)
        val_data = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_cols, reference=train_data)
        lgb_model = lgb.train(lgb_params, train_data, num_boost_round=500, valid_sets=[val_data], callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        oof_lgb[val_idx] = lgb_model.predict(X_val)
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        xgb_model = xgb.train(xgb_params, dtrain, num_boost_round=500, evals=[(dval, 'val')], early_stopping_rounds=50, verbose_eval=False)
        oof_xgb[val_idx] = xgb_model.predict(dval)
        cb_train_data = X_train.copy()
        cb_val_data = X_val.copy()
        for c in cat_cols:
            if c in cb_train_data.columns:
                cb_train_data[c] = cb_train_data[c].astype(int).astype(str)
                cb_val_data[c] = cb_val_data[c].astype(int).astype(str)
        cb_train = cb.Pool(cb_train_data, label=y_train, cat_features=[i for i, c in enumerate(feat_df.columns) if c in cat_cols])
        cb_val = cb.Pool(cb_val_data, label=y_val, cat_features=[i for i, c in enumerate(feat_df.columns) if c in cat_cols])
        cb_model = cb.CatBoostClassifier(iterations=500, learning_rate=0.05, depth=6, l2_leaf_reg=1.0, auto_class_weights='Balanced', verbose=0, random_seed=42)
        cb_model.fit(cb_train, eval_set=cb_val, early_stopping_rounds=50, verbose=0)
        oof_cb[val_idx] = cb_model.predict_proba(cb_val)[:, 1]
        lgb_auc = roc_auc_score(y_val, oof_lgb[val_idx])
        xgb_auc = roc_auc_score(y_val, oof_xgb[val_idx])
        cb_auc = roc_auc_score(y_val, oof_cb[val_idx])
        print(f"Fold {fold+1}: LGB={lgb_auc:.4f}, XGB={xgb_auc:.4f}, CB={cb_auc:.4f}")
    lgb_oof = roc_auc_score(labels, oof_lgb)
    xgb_oof = roc_auc_score(labels, oof_xgb)
    cb_oof = roc_auc_score(labels, oof_cb)
    for w1 in np.arange(0, 1.01, 0.1):
        for w2 in np.arange(0, 1.01 - w1, 0.1):
            w3 = 1 - w1 - w2
            if w3 < 0:
                continue
            blend = w1 * oof_lgb + w2 * oof_xgb + w3 * oof_cb
            auc = roc_auc_score(labels, blend)
            if auc > 0.7:
                print(f"\nSUCCESS! Weights: LGB={w1:.1f}, XGB={w2:.1f}, CB={w3:.1f} -> AUC={auc:.4f}")
    print(f"\nIndividual OOF: LGB={lgb_oof:.4f}, XGB={xgb_oof:.4f}, CB={cb_oof:.4f}")
    avg_blend = (oof_lgb + oof_xgb + oof_cb) / 3
    avg_auc = roc_auc_score(labels, avg_blend)
    print(f"Average Blend AUC: {avg_auc:.4f}")
    return max(lgb_oof, xgb_oof, cb_oof, avg_auc)


if __name__ == '__main__':
    df = pd.read_parquet('../data/taac2026_demo.parquet')
    print(f"Data: {df.shape}")
    auc = train_ensemble(df)
    print(f"\nBest AUC: {auc:.4f}")
    if auc >= 0.7:
        print("TARGET ACHIEVED!")
    else:
        print(f"Gap: {0.7 - auc:.4f}")
