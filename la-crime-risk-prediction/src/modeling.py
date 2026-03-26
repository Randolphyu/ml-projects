from __future__ import annotations
from typing import Dict, Any, Optional
import os
import time

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    accuracy_score
)

try:
    from xgboost import XGBClassifier
    XGB_OK = True
except Exception:
    XGB_OK = False

try:
    from boruta import BorutaPy
    BORUTA_OK = True
except Exception:
    BORUTA_OK = False

from .config import Config


def run_boruta_feature_selection(
    train: pd.DataFrame,
    feature_cols: list[str],
    cfg: Config,
) -> tuple[list[str], Optional[pd.DataFrame]]:
    """
    使用 Boruta 做 feature selection
    
    Returns:
    --------
    selected_features: list[str]
        被選中的特徵列表
    boruta_df: pd.DataFrame
        詳細結果表（可存成 CSV）
    """
    
    if not BORUTA_OK:
        print("⚠️  Boruta not installed. Install with: pip install boruta")
        print("   Skipping Boruta feature selection...")
        return feature_cols, None
    
    print("\n" + "="*70)
    print("🧬 BORUTA FEATURE SELECTION")
    print("="*70)
    
    # 準備數據（移除 NaN）
    df = train[feature_cols + ["y"]].dropna().copy()
    
    # ⭐ 抽樣（避免太慢）
    n_total = len(df)
    max_samples = getattr(cfg, 'boruta_sample_size', 50000)
    
    if n_total > max_samples:
        # Stratified sampling（保持正負比例）
        pos = df[df["y"] == 1]
        neg = df[df["y"] == 0]
        
        n_pos = len(pos)
        n_neg_needed = max(max_samples - n_pos, int(max_samples * 0.2))
        n_neg_needed = min(len(neg), n_neg_needed)
        
        pos_sample = pos
        neg_sample = neg.sample(n=n_neg_needed, random_state=42)
        
        df_sample = pd.concat([pos_sample, neg_sample]).sample(frac=1.0, random_state=42)
        print(f"\n   Sampling for Boruta: {len(df_sample):,} rows")
        print(f"      Positive: {len(pos_sample):,}")
        print(f"      Negative: {len(neg_sample):,}")
    else:
        df_sample = df
        print(f"\n   Using full train data: {len(df_sample):,} rows")
    
    X = df_sample[feature_cols].values
    y = df_sample["y"].values
    
    # ⭐ Boruta with Random Forest
    print(f"\n   Training Boruta with {len(feature_cols)} features...")
    
    rf_estimator = RandomForestClassifier(
        n_estimators=100,  # 減少樹數量加速
        max_depth=15,
        n_jobs=-1,
        class_weight="balanced",
        random_state=42,
    )
    
    boruta = BorutaPy(
        estimator=rf_estimator,
        n_estimators='auto',
        max_iter=getattr(cfg, 'boruta_max_iter', 50),
        random_state=42,
        verbose=1,  # 顯示進度
    )
    
    start = time.time()
    boruta.fit(X, y)
    elapsed = time.time() - start
    
    # 整理結果
    support = boruta.support_
    ranking = boruta.ranking_
    
    selected_features = [f for f, keep in zip(feature_cols, support) if keep]
    selected_count = len(selected_features)
    
    print(f"\n   ✅ Boruta completed in {elapsed/60:.1f} minutes")
    print(f"   Selected: {selected_count}/{len(feature_cols)} features ({selected_count/len(feature_cols)*100:.1f}%)")
    
    # 創建結果 DataFrame
    boruta_df = pd.DataFrame({
        "feature": feature_cols,
        "selected": support,
        "ranking": ranking,
    }).sort_values(["selected", "ranking"], ascending=[False, True])
    
    # 顯示 top 10 被選中的特徵
    print(f"\n   Top 10 selected features:")
    top_10 = boruta_df[boruta_df['selected']].head(10)
    for idx, (_, row) in enumerate(top_10.iterrows(), 1):
        print(f"      {idx:2d}. {row['feature']:30s} (rank={row['ranking']})")
    
    return selected_features, boruta_df


def get_feature_columns(panel: pd.DataFrame) -> list:
    """動態檢測可用特徵"""
    
    base_features = [
        # Crime lags
        "lag1m_total", "lag3m_total",
        "lag1m_violent", "lag3m_violent",
        "lag1m_property", "lag3m_property",
        
        # Grid statistics
        "hist_mean", "grid_mean", "grid_std", "grid_max",
        
        # Temporal
        "time_block", "time_of_day", "month", "season", "is_night",
        
        # 新增特徵
        "dbscan_cluster_cnt", "is_hotspot",
        "avg_victim_age", "young_victim_cnt", "female_victim_ratio",
        "adult_arrest_cnt", "juv_arrest_cnt",
        "crime_diversity", "premise_diversity",
    ]
    
    # ACS features
    acs_features = [c for c in panel.columns if c.startswith("acs_")]
    
    # 環境特徵
    env_features = []
    for col in ["cases_log", "deaths_log", "state_cases_log", "state_deaths_log", 
                "lag1m_covid_cases", "avg_daily_temp_c", "max_daily_temp_c", 
                "min_daily_temp_c", "total_daily_precip_mm", "lag1m_temp", "is_holiday"]:
        if col in panel.columns:
            env_features.append(col)
    
    # 合併所有特徵
    all_features = base_features + acs_features  + env_features
    
    # 只保留實際存在的欄位
    available = [f for f in all_features if f in panel.columns]
    
    print(f"\n🎯 Feature Selection:")
    print(f"   Crime lags: {len([f for f in available if 'lag' in f and any(x in f for x in ['total', 'violent', 'property'])])}")
    print(f"   ACS: {len(acs_features)}")
    print(f"   Environment: {len([f for f in available if f in env_features])}")
    print(f"   New features: {len([f for f in available if any(x in f for x in ['dbscan', 'victim', 'arrest', 'diversity'])])}")
    print(f"   Temporal: {len([f for f in available if f in ['time_block', 'time_of_day', 'month', 'season', 'is_night']])}")
    print(f"   Grid stats: {len([f for f in available if f in ['hist_mean', 'grid_mean', 'grid_std', 'grid_max']])}")
    print(f"   Total: {len(available)} features")
    
    return available


def _calculate_f1_metrics(y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    """計算 F1 相關指標"""
    f1_fixed = f1_score(y_true, (y_proba >= 0.5).astype(int), zero_division=0)

    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    
    fscores = np.zeros_like(precision)
    valid_indices = (precision + recall) > 0
    fscores[valid_indices] = (2 * precision[valid_indices] * recall[valid_indices]) / (precision[valid_indices] + recall[valid_indices])
    
    best_f1_idx = np.argmax(fscores)
    optimal_threshold = thresholds[best_f1_idx] if best_f1_idx < len(thresholds) else 0.5
    
    return {
        "f1@0.5": f1_fixed,
        "f1_optimal": fscores[best_f1_idx],
        "thresh_optimal": optimal_threshold
    }


def _calculate_recall_at_k(y_true: np.ndarray, y_proba: np.ndarray, k: int) -> float:
    """
    Recall@K: 在預測最高的 K 個樣本中，能抓到多少實際的 positive
    """
    if k > len(y_true):
        k = len(y_true)
    
    # 找出預測概率最高的 K 個樣本
    top_k_idx = np.argsort(y_proba)[-k:]
    
    # 計算這 K 個樣本中有多少是真正的 positive
    n_true_positives_in_top_k = y_true[top_k_idx].sum()
    
    # 總共有多少 positive
    total_positives = y_true.sum()
    
    if total_positives == 0:
        return 0.0
    
    # Recall@K = 抓到的 positive / 總 positive
    return float(n_true_positives_in_top_k / total_positives)


def train_models(train: pd.DataFrame, test: pd.DataFrame, cfg: Config) -> Dict[str, Any]:
    """
    訓練模型 - 整合 Boruta feature selection
    """
    
    # ========================================
    # Step 1: 獲取初始特徵列表
    # ========================================
    
    initial_features = get_feature_columns(train)
    
    if not initial_features:
        raise ValueError("No valid features found!")
    
    # ========================================
    # Step 2: Boruta Feature Selection (可選)
    # ========================================
    
    boruta_df = None
    
    if getattr(cfg, 'use_boruta', False):
        selected_features, boruta_df = run_boruta_feature_selection(
            train, initial_features, cfg
        )
        
        # 保存 Boruta 結果
        if boruta_df is not None:
            boruta_path = os.path.join(cfg.out_dir, "boruta_feature_ranking.csv")
            boruta_df.to_csv(boruta_path, index=False)
            print(f"\n   💾 Boruta ranking saved: {boruta_path}")
        
        # 決定是否使用 Boruta 選出的特徵
        if getattr(cfg, 'use_boruta_for_training', False):
            FEATURE_COLS = selected_features
            print(f"\n   ✅ Using {len(FEATURE_COLS)} Boruta-selected features for training")
        else:
            FEATURE_COLS = initial_features
            print(f"\n   ℹ️  Boruta results saved but using all {len(FEATURE_COLS)} features for training")
    else:
        FEATURE_COLS = initial_features
        print(f"\n   ℹ️  Boruta disabled, using all {len(FEATURE_COLS)} features")
    
    # ========================================
    # Step 3: 準備訓練數據
    # ========================================
    
    X_tr = train[FEATURE_COLS].values
    X_te = test[FEATURE_COLS].values
    y_tr = train["y"].values
    y_te = test["y"].values

    results: Dict[str, Any] = {
        "boruta_results": boruta_df  # 保存 Boruta 結果供後續使用
    }

    # 類別不平衡資訊
    n_pos = np.sum(y_tr == 1)
    n_neg = np.sum(y_tr == 0)
    scale_pos_weight = n_neg / max(n_pos, 1)
    
    print(f"\n📊 Class Distribution:")
    print(f"   Positive: {n_pos:,} ({n_pos/len(y_tr)*100:.2f}%)")
    print(f"   Negative: {n_neg:,} ({n_neg/len(y_tr)*100:.2f}%)")
    print(f"   scale_pos_weight: {scale_pos_weight:.2f}")

    print("\n" + "="*70)
    print("📊 TRAINING MODELS")
    print("="*70)
    
    # ========================================
    # Model 1: Logistic Regression
    # ========================================
    
    print("\n📊 Training Logistic Regression...")
    start = time.time()
    
    lr = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear",
        random_state=42,
    )
    lr.fit(X_tr, y_tr)
    p_lr = lr.predict_proba(X_te)[:, 1]
    lr_f1_metrics = _calculate_f1_metrics(y_te, p_lr)
    
    optimal_threshold_lr = lr_f1_metrics["thresh_optimal"]
    y_pred_optimal_lr = (p_lr >= optimal_threshold_lr).astype(int)
    opt_precision_lr = precision_score(y_te, y_pred_optimal_lr, zero_division=0)
    opt_recall_lr = recall_score(y_te, y_pred_optimal_lr, zero_division=0)
    opt_accuracy_lr = accuracy_score(y_te, y_pred_optimal_lr)

    results["logreg"] = {
        "model": lr,
        "feature_names": FEATURE_COLS,
        "roc_auc": roc_auc_score(y_te, p_lr),
        "pr_auc": average_precision_score(y_te, p_lr),
        "f1@0.5": lr_f1_metrics["f1@0.5"],
        "f1_optimal": lr_f1_metrics["f1_optimal"],
        "thresh_optimal": lr_f1_metrics["thresh_optimal"],
        "precision_optimal": opt_precision_lr,
        "recall_optimal": opt_recall_lr,
        "accuracy_optimal": opt_accuracy_lr,
        "proba": p_lr,
        "train_time": time.time() - start,
    }
    
    print(f"   ✅ Completed in {results['logreg']['train_time']:.1f}s")
    print(f"      ROC-AUC: {results['logreg']['roc_auc']:.4f}")
    print(f"      PR-AUC:  {results['logreg']['pr_auc']:.4f}")
    print(f"      Opt F1: {results['logreg']['f1_optimal']:.4f} @ Thresh={optimal_threshold_lr:.4f}")
    print(f"      Opt Precision: {opt_precision_lr:.4f} | Opt Recall: {opt_recall_lr:.4f} | Opt Acc: {opt_accuracy_lr:.4f}")

    # ========================================
    # Model 2: Random Forest
    # ========================================
    
    print("\n📊 Training Random Forest...")
    start = time.time()
    
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_leaf=10,
        max_features='sqrt',
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
        verbose=0
    )
    rf.fit(X_tr, y_tr)
    p_rf = rf.predict_proba(X_te)[:, 1]
    rf_f1_metrics = _calculate_f1_metrics(y_te, p_rf)
    
    optimal_threshold_rf = rf_f1_metrics["thresh_optimal"]
    y_pred_optimal_rf = (p_rf >= optimal_threshold_rf).astype(int)
    opt_precision_rf = precision_score(y_te, y_pred_optimal_rf, zero_division=0)
    opt_recall_rf = recall_score(y_te, y_pred_optimal_rf, zero_division=0)
    opt_accuracy_rf = accuracy_score(y_te, y_pred_optimal_rf)

    results["rf"] = {
        "model": rf,
        "feature_names": FEATURE_COLS,
        "roc_auc": roc_auc_score(y_te, p_rf),
        "pr_auc": average_precision_score(y_te, p_rf),
        "f1@0.5": rf_f1_metrics["f1@0.5"],
        "f1_optimal": rf_f1_metrics["f1_optimal"],
        "thresh_optimal": rf_f1_metrics["thresh_optimal"],
        "precision_optimal": opt_precision_rf,
        "recall_optimal": opt_recall_rf,
        "accuracy_optimal": opt_accuracy_rf,
        "proba": p_rf,
        "train_time": time.time() - start,
    }
    
    print(f"   ✅ Completed in {results['rf']['train_time']:.1f}s")
    print(f"      ROC-AUC: {results['rf']['roc_auc']:.4f}")
    print(f"      PR-AUC:  {results['rf']['pr_auc']:.4f}")
    print(f"      Opt F1: {results['rf']['f1_optimal']:.4f} @ Thresh={optimal_threshold_rf:.4f}")
    print(f"      Opt Precision: {opt_precision_rf:.4f} | Opt Recall: {opt_recall_rf:.4f} | Opt Acc: {opt_accuracy_rf:.4f}")

    # ========================================
    # Model 3: XGBoost (Tuned)
    # ========================================
    
    if cfg.use_xgboost and XGB_OK:
        print("\n📊 Training XGBoost (with optimized params)...")
        start = time.time()
        
        # ⭐ 使用調優後的最佳參數
        xgb = XGBClassifier(
            n_estimators=300, 
            max_depth=4, #5
            min_child_weight=10, #20
            gamma=0, #1
            learning_rate=0.03, #0.05
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=2,
            scale_pos_weight=scale_pos_weight,
            eval_metric='aucpr',
            objective='binary:logistic',
            random_state=42,
            tree_method='hist',
            use_label_encoder=False,
            n_jobs=-1
        )
        
        # ⭐ 修正：使用 DataFrame 而非 numpy array，讓 XGBoost 自動識別特徵名
        xgb.fit(train[FEATURE_COLS], y_tr)
        
        p_xgb = xgb.predict_proba(X_te)[:, 1]
        xgb_f1_metrics = _calculate_f1_metrics(y_te, p_xgb)

        optimal_threshold_xgb = xgb_f1_metrics["thresh_optimal"]
        y_pred_optimal_xgb = (p_xgb >= optimal_threshold_xgb).astype(int)
        opt_precision_xgb = precision_score(y_te, y_pred_optimal_xgb, zero_division=0)
        opt_recall_xgb = recall_score(y_te, y_pred_optimal_xgb, zero_division=0)
        opt_accuracy_xgb = accuracy_score(y_te, y_pred_optimal_xgb)
        
        results["xgb"] = {
            "model": xgb,
            "feature_names": FEATURE_COLS,
            "roc_auc": roc_auc_score(y_te, p_xgb),
            "pr_auc": average_precision_score(y_te, p_xgb),
            "f1@0.5": xgb_f1_metrics["f1@0.5"],
            "f1_optimal": xgb_f1_metrics["f1_optimal"],
            "thresh_optimal": xgb_f1_metrics["thresh_optimal"],
            "precision_optimal": opt_precision_xgb,
            "recall_optimal": opt_recall_xgb,
            "accuracy_optimal": opt_accuracy_xgb,
            "proba": p_xgb,
            "train_time": time.time() - start,
        }
        
        print(f"   ✅ Completed in {results['xgb']['train_time']:.1f}s")
        print(f"      ROC-AUC: {results['xgb']['roc_auc']:.4f}")
        print(f"      PR-AUC:  {results['xgb']['pr_auc']:.4f}")
        print(f"      Opt F1: {results['xgb']['f1_optimal']:.4f} @ Thresh={optimal_threshold_xgb:.4f}")
        print(f"      Opt Precision: {opt_precision_xgb:.4f} | Opt Recall: {opt_recall_xgb:.4f} | Opt Acc: {opt_accuracy_xgb:.4f}")

    # ========================================
    # Recall@K Evaluation (核心指標!)
    # ========================================
    
    print("\n" + "="*70)
    print("🎯 RECALL@K EVALUATION")
    print("="*70)
    
    k_values = [100, 500, 1000]
    
    for model_name, model_info in results.items():
        if model_name == "boruta_results":  # 跳過 Boruta 結果
            continue
        
        proba = model_info["proba"]
        
        print(f"\n📊 {model_name.upper()}:")
        for k in k_values:
            if k <= len(y_te):
                recall_k = _calculate_recall_at_k(y_te, proba, k)
                results[model_name][f"recall@{k}"] = recall_k
                print(f"   Recall@{k:4d}: {recall_k:6.1%}")
    
    print(f"\n✅ All models trained successfully!")
    
    return results