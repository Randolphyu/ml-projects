# =========================================
# File: src/explain.py
# =========================================

# =========================================
# 完整視覺化模組: Feature Importance + SHAP
# =========================================

from __future__ import annotations
import os
from typing import Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .config import Config

try:
    import shap
    SHAP_OK = True
except Exception:
    SHAP_OK = False

try:
    from PyALE import ale
    ALE_OK = True
except Exception:
    ALE_OK = False

def plot_feature_importance_comparison(
    results: Dict[str, Any], 
    train: pd.DataFrame,
    cfg: Config,
    top_n: int = 20
) -> None:
    """
    比較所有模型的 Feature Importance
    """
    
    print("\n📊 Generating Feature Importance plots...")
    
    os.makedirs(cfg.out_dir, exist_ok=True)
    
    # 準備數據
    importance_data = []
    
    for model_name in ['logreg', 'rf', 'xgb']:
        if model_name not in results:
            continue
        
        model = results[model_name]['model']
        feature_names = results[model_name]['feature_names']
        
        # 獲取 importance
        if model_name == 'logreg':
            # Logistic Regression: 使用係數絕對值
            importance = np.abs(model.coef_[0])
        elif model_name == 'rf':
            # Random Forest: feature_importances_
            importance = model.feature_importances_
        elif model_name == 'xgb':
            # XGBoost: feature_importances_
            importance = model.feature_importances_
        else:
            continue
        
        # 添加到列表
        for feat, imp in zip(feature_names, importance):
            importance_data.append({
                'model': model_name.upper(),
                'feature': feat,
                'importance': imp
            })
    
    df_imp = pd.DataFrame(importance_data)
    
    # ========= 圖1: Top-N 特徵比較 (分模型) =========
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    
    for idx, (model_name, ax) in enumerate(zip(['LOGREG', 'RF', 'XGB'], axes)):
        model_data = df_imp[df_imp['model'] == model_name].copy()
        
        if len(model_data) == 0:
            ax.text(0.5, 0.5, f'{model_name}\nNot Available', 
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            continue
        
        # 排序並取 Top-N
        model_data = model_data.nlargest(top_n, 'importance')
        
        # 特徵分類著色
        colors = []
        for feat in model_data['feature']:
            if 'lag' in feat and ('total' in feat or 'violent' in feat or 'property' in feat):
                colors.append('#e74c3c')  # 紅色: Crime lags
            elif feat.startswith('acs_'):
                colors.append('#3498db')  # 藍色: ACS
            elif feat in ['cases', 'deaths', 'state_cases', 'state_deaths', 'lag1m_covid_cases']:
                colors.append('#f39c12')  # 橘色: COVID
            elif feat in ['avg_daily_temp_c', 'max_daily_temp_c', 'min_daily_temp_c', 
                         'total_daily_precip_mm', 'lag1m_temp']:
                colors.append('#27ae60')  # 綠色: Weather
            elif feat == 'is_holiday':
                colors.append('#9b59b6')  # 紫色: Holiday
            elif feat in ['time_block', 'month', 'season', 'time_of_day', 'is_night']:
                colors.append('#95a5a6')  # 灰色: Temporal
            else:
                colors.append('#34495e')  # 深灰: Others
        
        # 繪圖
        ax.barh(range(len(model_data)), model_data['importance'], color=colors)
        ax.set_yticks(range(len(model_data)))
        ax.set_yticklabels(model_data['feature'], fontsize=9)
        ax.set_xlabel('Importance', fontsize=11)
        ax.set_title(f'{model_name} - Top {top_n} Features', fontsize=12, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
    
    # 添加圖例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', label='Crime Lags'),
        Patch(facecolor='#3498db', label='ACS (Demographics)'),
        Patch(facecolor='#f39c12', label='COVID'),
        Patch(facecolor='#27ae60', label='Weather'),
        Patch(facecolor='#9b59b6', label='Holiday'),
        Patch(facecolor='#95a5a6', label='Temporal'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=6, 
              bbox_to_anchor=(0.5, -0.02), fontsize=10)
    
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(os.path.join(cfg.out_dir, 'feature_importance_comparison.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Saved: {cfg.out_dir}/feature_importance_comparison.png")
    
    # ========= 圖2: 特徵類別重要性統計 =========
    
    # 為每個特徵添加類別標籤
    def categorize_feature(feat):
        if 'lag' in feat and any(x in feat for x in ['total', 'violent', 'property', 'cnt']):
            return 'Crime Lags'
        elif feat.startswith('acs_'):
            return 'ACS'
        elif feat in ['cases', 'deaths', 'state_cases', 'state_deaths', 'lag1m_covid_cases']:
            return 'COVID'
        elif feat in ['avg_daily_temp_c', 'max_daily_temp_c', 'min_daily_temp_c', 
                     'total_daily_precip_mm', 'lag1m_temp']:
            return 'Weather'
        elif feat == 'is_holiday':
            return 'Holiday'
        elif feat in ['time_block', 'month', 'season', 'time_of_day', 'is_night']:
            return 'Temporal'
        elif feat in ['hist_mean', 'grid_mean', 'grid_std', 'grid_max']:
            return 'Grid Stats'
        else:
            return 'Others'
    
    df_imp['category'] = df_imp['feature'].apply(categorize_feature)
    
    # 計算每個類別的平均重要性
    category_importance = df_imp.groupby(['model', 'category'])['importance'].mean().reset_index()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Pivot for grouped bar chart
    pivot = category_importance.pivot(index='category', columns='model', values='importance')
    pivot = pivot.fillna(0)
    
    pivot.plot(kind='bar', ax=ax, width=0.8, 
              color=['#e74c3c', '#3498db', '#2ecc71'])
    
    ax.set_title('Average Feature Importance by Category', fontsize=14, fontweight='bold')
    ax.set_xlabel('Feature Category', fontsize=12)
    ax.set_ylabel('Average Importance', fontsize=12)
    ax.legend(title='Model', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.out_dir, 'feature_category_importance.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Saved: {cfg.out_dir}/feature_category_importance.png")
    
    # ========= 圖3: 詳細重要性表格 =========
    
    # 取最佳模型的 Top-30 特徵
    best_model = max(results.keys(), key=lambda m: results[m].get('pr_auc', 0))
    best_model_data = df_imp[df_imp['model'] == best_model.upper()].copy()
    best_model_data = best_model_data.nlargest(30, 'importance')
    
    # 儲存為 CSV
    best_model_data.to_csv(
        os.path.join(cfg.out_dir, f'{best_model}_top30_features.csv'),
        index=False
    )
    
    print(f"   ✅ Saved: {cfg.out_dir}/{best_model}_top30_features.csv")
    
    # 輸出 Top-10 到 console
    print(f"\n🏆 Top 10 Most Important Features ({best_model.upper()}):")
    print(f"   {'Rank':<5} {'Feature':<30} {'Category':<15} {'Importance':<10}")
    print(f"   {'-'*65}")
    for idx, row in best_model_data.head(10).iterrows():
        print(f"   {idx+1:<5} {row['feature']:<30} {row['category']:<15} {row['importance']:.6f}")




def plot_ale_for_top_features(
    results: Dict[str, Any],
    train: pd.DataFrame,
    cfg: Config,
    top_n: int = 6,
    sample_size: int = 20000,
) -> None:
    """
    使用 ALE (Accumulated Local Effects) 畫出「方向性」：
    特徵變大時，預測風險大致是往上還是往下。
    只對最佳模型做 Top-N 特徵。
    """
    if not cfg.use_ale:
        print("🎨 ALE disabled by config, skip.")
        return

    if not ALE_OK:
        print("⚠️  PyALE 未安裝，請先：pip install PyALE")
        return

    print("\n📊 Generating ALE plots...")

    os.makedirs(cfg.out_dir, exist_ok=True)

    # 1) 找 PR-AUC 最好的模型
    best_name = max(results.keys(), key=lambda m: results[m].get("pr_auc", 0))
    best_model_info = results[best_name]
    model = best_model_info["model"]

    feature_names = best_model_info.get("feature_names")
    if feature_names is None:
        # 後備：排除非特徵欄位
        feature_names = [
            c for c in train.columns
            if c not in ("y", "grid_id", "year_month", "time_block", "cnt")
        ]

    X = train[feature_names].copy()

    # 2) 抽樣（避免太慢）
    if len(X) > sample_size:
        X = X.sample(sample_size, random_state=42)
        print(f"   Using sample of {len(X):,} rows for ALE")

    # 3) 根據最佳模型的重要性決定要畫哪些特徵
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        order = np.argsort(importances)[::-1]
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
        order = np.argsort(importances)[::-1]
    else:
        # fallback: 就照欄位順序
        order = np.arange(len(feature_names))

    top_feats = [feature_names[i] for i in order[:top_n]]
    print(f"   Top-{top_n} features for ALE: {top_feats}")

    # 4) 為了 ALE，我們包一個「只回傳 P(y=1)」的 wrapper
    class ProbWrapper:
        def __init__(self, base_model):
            self.base_model = base_model

        def predict(self, X_):
            return self.base_model.predict_proba(X_)[:, 1]

    wrapped_model = ProbWrapper(model)

    # 5) 逐一產 ALE 圖
    n_rows = int(np.ceil(top_n / 3))
    fig, axes = plt.subplots(n_rows, 3, figsize=(16, 4 * n_rows))
    axes = np.array(axes).reshape(-1)

    for ax, feat in zip(axes, top_feats):
        try:
            ale_eff = ale(
                X=X,
                model=wrapped_model,
                feature=[feat],
                grid_size=20,
                include_CI=False,
            )
            ax.plot(ale_eff["quantiles"], ale_eff["ALE"], marker="o", linewidth=1.5)
            ax.axhline(0, color="gray", linestyle="--", linewidth=1)
            ax.set_title(feat)
            ax.set_xlabel("Feature value")
            ax.set_ylabel("ALE (effect on risk)")
            ax.grid(alpha=0.3)
        except Exception as e:
            ax.text(
                0.5,
                0.5,
                f"Failed ALE for {feat}\n{type(e).__name__}",
                ha="center",
                va="center",
            )
            ax.axis("off")

    # 把多餘的 subplot 關掉
    for j in range(len(top_feats), len(axes)):
        axes[j].axis("off")

    plt.suptitle(
        f"ALE plots - Top {len(top_feats)} features ({best_name.upper()})",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    out_path = os.path.join(cfg.out_dir, "ale_top_features.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ✅ Saved: {out_path}")



def shap_analysis(
    results: Dict[str, Any],
    train: pd.DataFrame,
    test: pd.DataFrame,
    cfg: Config,
    sample_size: int = 500
) -> None:
    """
    SHAP 分析與視覺化
    """
    
    if not SHAP_OK:
        print("⚠️  SHAP not installed, skipping SHAP analysis")
        return
    
    print("\n📊 Generating SHAP plots...")
    
    os.makedirs(cfg.out_dir, exist_ok=True)
    
    # 選擇最佳模型
    best_model_name = max(results.keys(), key=lambda m: results[m].get('pr_auc', 0))
    best_model = results[best_model_name]['model']
    feature_names = results[best_model_name]['feature_names']
    
    print(f"   Using best model: {best_model_name.upper()} (PR-AUC: {results[best_model_name]['pr_auc']:.4f})")
    
    # 準備數據
    X_train = train[feature_names]
    X_test = test[feature_names]
    X_all = pd.concat([X_train, X_test], axis=0)
    
    # 採樣 (SHAP 計算很慢)
    # if len(X_test) > sample_size:
        # sample_idx = np.random.choice(len(X_test), sample_size, replace=False)
        # X_sample = X_test.sample(n=5000, random_state=42).iloc[sample_idx].copy().reset_index(drop=True)  # 重置索引
    # else:
        # X_sample = X_all.sample(n=5000, random_state=42).copy().reset_index(drop=True)
        # 採樣 (SHAP 計算很慢) —— 這裡是關鍵
    if len(X_all) > sample_size:
        X_sample = X_all.sample(n=sample_size, random_state=42).copy().reset_index(drop=True)
    else:
        X_sample = X_all.copy().reset_index(drop=True)
    try:
        # 創建 SHAP explainer
        print(f"   Computing SHAP values for {len(X_sample)} samples...")
        
        if best_model_name == 'xgb':
            explainer = shap.TreeExplainer(best_model)
        elif best_model_name == 'rf':
            explainer = shap.TreeExplainer(best_model)
        else:  # Logistic Regression
            explainer = shap.LinearExplainer(best_model, X_train)
        
        shap_values = explainer(X_sample)


        # --- 修正：處理多輸出情況 ---
        if isinstance(shap_values, shap.Explanation) and shap_values.values.ndim == 3:
            # values shape: (n_samples, n_outputs, n_features)
            n_outputs = shap_values.values.shape[1]
            class_idx = 1 if n_outputs > 1 else 0  # 二元分類取正類
            
            # 提取單一輸出的值
            values_2d = shap_values.values[:, class_idx, :]  # (n_samples, n_features)
            
            # 處理 base_values
            base_values = shap_values.base_values
            if np.ndim(base_values) == 2:
                base_values = base_values[:, class_idx]
            
            # 確保數據形狀匹配
            data_array = X_sample.values  # 使用 .values 而不是 .to_numpy()
            
            # 驗證形狀
            assert values_2d.shape[0] == data_array.shape[0], \
                f"Sample mismatch: {values_2d.shape[0]} vs {data_array.shape[0]}"
            assert values_2d.shape[1] == data_array.shape[1], \
                f"Feature mismatch: {values_2d.shape[1]} vs {data_array.shape[1]}"
            
            # 創建新的 Explanation 對象
            shap_values = shap.Explanation(
                values=values_2d,
                base_values=base_values,
                data=data_array,
                feature_names=list(feature_names),
            )
        
        # ========= 圖1: SHAP Beeswarm Plot =========
        print("   Generating beeswarm plot...")
        plt.figure(figsize=(12, 10))
        shap.plots.beeswarm(shap_values, show=False, max_display=20)
        plt.title(f'SHAP Feature Impact - {best_model_name.upper()}', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(cfg.out_dir, 'shap_beeswarm.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Saved: {cfg.out_dir}/shap_beeswarm.png")
        
        # ========= 圖2: SHAP Summary Bar Plot =========
        print("   Generating summary bar plot...")
        plt.figure(figsize=(10, 8))
        shap.plots.bar(shap_values, show=False, max_display=20)
        plt.title(f'SHAP Mean Absolute Impact - {best_model_name.upper()}', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(cfg.out_dir, 'shap_summary_bar.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Saved: {cfg.out_dir}/shap_summary_bar.png")
        
        # ========= 圖3: SHAP Waterfall Plot (單個預測示例) =========
        print("   Generating waterfall plot...")
        
        # 選擇一個高風險預測樣本
        proba = best_model.predict_proba(X_sample)[:, 1]
        high_risk_idx = np.argmax(proba)
        
        plt.figure(figsize=(10, 8))
        shap.plots.waterfall(shap_values[high_risk_idx], show=False, max_display=15)
        plt.title(f'SHAP Explanation for High-Risk Prediction\n(Predicted Prob: {proba[high_risk_idx]:.3f})', 
                 fontsize=12, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(cfg.out_dir, 'shap_waterfall_example.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Saved: {cfg.out_dir}/shap_waterfall_example.png")
        
        # ========= 圖4: SHAP Dependence Plots (Top 10 features) =========
        print("   Generating dependence plots for top 10 features...")

        # 找出 SHAP 值最大的 10 個特徵
        mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
        top_10_indices = np.argsort(mean_abs_shap)[-10:][::-1]
        top_10_features = [feature_names[i] for i in top_10_indices]

        # 調整為 5x2 的佈局
        fig, axes = plt.subplots(5, 2, figsize=(16, 20))
        axes = axes.ravel()

        for idx, (feat_idx, feat_name) in enumerate(zip(top_10_indices, top_10_features)):
            ax = axes[idx]
            
            # 手動繪製 dependence plot
            if isinstance(X_sample, pd.DataFrame):
                feature_values = X_sample.iloc[:, feat_idx].values
            else:
                feature_values = X_sample[:, feat_idx]

            shap_vals = shap_values.values[:, feat_idx]
            
            # 按特徵值排序
            sorted_idx = np.argsort(feature_values)
            
            scatter = ax.scatter(
                feature_values[sorted_idx], 
                shap_vals[sorted_idx],
                c=feature_values[sorted_idx],
                cmap='coolwarm',
                alpha=0.6,
                s=20
            )
            
            ax.set_xlabel(feat_name, fontsize=10)
            ax.set_ylabel('SHAP value', fontsize=10)
            ax.set_title(f'Impact of {feat_name}', fontsize=11, fontweight='bold')
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            ax.grid(alpha=0.3)
            
            plt.colorbar(scatter, ax=ax)

        plt.suptitle('SHAP Dependence Plots - Top 10 Features', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(os.path.join(cfg.out_dir, 'shap_dependence_top10.png'), 
                dpi=150, bbox_inches='tight')
        plt.close()

        print(f"   ✅ Saved: {cfg.out_dir}/shap_dependence_top10.png")        
        # ========= 輸出 SHAP 統計 =========
        print(f"\n📈 SHAP Analysis Summary:")
        print(f"   Top 5 features by mean |SHAP|:")
        
        mean_abs_shap_sorted = np.argsort(mean_abs_shap)[::-1]
        for rank, idx in enumerate(mean_abs_shap_sorted[:5], 1):
            print(f"   {rank}. {feature_names[idx]:<30} {mean_abs_shap[idx]:.6f}")
        
    except Exception as e:
        print(f"   ⚠️  SHAP analysis failed: {e}")
        import traceback
        traceback.print_exc()


def generate_all_visualizations(
    results: Dict[str, Any],
    train: pd.DataFrame,
    test: pd.DataFrame,
    cfg: Config
) -> None:
    """
    生成所有視覺化
    """
    
    print("\n" + "="*80)
    print("🎨 VISUALIZATION PIPELINE")
    print("="*80)
    
    # 1. Feature Importance
    plot_feature_importance_comparison(results, train, cfg, top_n=20)
    
    # 2. SHAP Analysis
    shap_analysis(results, train, test, cfg, sample_size=500)
    
    print("\n✅ All visualizations complete!")
    print(f"📁 Output directory: {cfg.out_dir}/")
    