# =========================================
# File: src/features.py
# =========================================


from __future__ import annotations
from typing import Callable
import pandas as pd
import numpy as np

from .config import Config

def aggregate_to_grid_month_timeblock(df: pd.DataFrame, time_block_hours: int = 6) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    聚合犯罪數據 + 整合環境特徵
    """
    df = df.dropna(subset=["ts", "grid_id"]).copy()

    numeric_cols = [
        # COVID
        'cases', 'deaths', 'state_cases', 'state_deaths',
        # Weather
        'avg_daily_temp_c', 'max_daily_temp_c', 
        'min_daily_temp_c', 'total_daily_precip_mm',
        # ACS (可能也有逗號)
        'total_pop', 'white_pop', 'black_pop', 'asian_pop', 'hispanic_pop',
        'median_income', 'income_per_capita', 'housing_units',
        'occupied_housing_units', 'housing_units_renter_occupied',
        'owner_occupied_housing_units', 'median_rent',
        'households', 'pop_in_labor_force', 'unemployed_pop',
        'commuters_16_over', 'rent_over_50_percent',
    ]
    
    for col in numeric_cols:
        if col in df.columns:
            # 移除逗號並轉換為數值
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.replace(',', '').replace('', '0')
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)                  
    # 時間切塊
    df["year_month"] = df["ts"].dt.to_period("M")
    df["hour"] = df["ts"].dt.hour
    df["time_block"] = df["hour"] // time_block_hours
    df["dow"] = df["ts"].dt.dayofweek
    df["is_weekend"] = (df["dow"] >= 5).astype(int)

    # 確保 decomposition 欄位存在
    for col in [
        "is_violent", "is_non_violent",
        "is_property", "is_non_property",
        "is_homicide", "is_robbery", "is_assault",
        "is_sexual", "is_shooting", "is_kidnap_child", "is_weapon",
    ]:
        if col not in df.columns:
            df[col] = False

    # ========= Crime aggregation =========
    group_cols = ["grid_id", "year_month", "time_block"]
    
    agg_dict = {
        "ts": "size",  # 總犯罪數
        "is_violent": "sum",
        "is_non_violent": "sum",
        "is_property": "sum",
        "is_non_property": "sum",
        "is_homicide": "sum",
        "is_robbery": "sum",
        "is_assault": "sum",
        "is_sexual": "sum",
        "is_shooting": "sum",
        "is_kidnap_child": "sum",
        "is_weapon": "sum",
        "is_weekend": "mean",
    }
    
    # ========= 動態加入環境特徵 =========
    # COVID features
    covid_features = ["cases", "deaths", "state_cases", "state_deaths"]
    for col in covid_features:
        if col in df.columns:
            agg_dict[col] = "mean"  # 該時段的平均值
    
    # Weather features
    weather_features = [
        "avg_daily_temp_c", "max_daily_temp_c", 
        "min_daily_temp_c", "total_daily_precip_mm"
    ]
    for col in weather_features:
        if col in df.columns:
            agg_dict[col] = "mean"
    
    # Holiday flag
    if "is_holiday" in df.columns:
        agg_dict["is_holiday"] = "max"  # 該時段內是否有假日
    
    # 受害者特徵
    # if "Vict Age" in df.columns:
    #     agg_dict["avg_victim_age"] = ("Vict Age", lambda x: x.mean() if len(x) > 0 else 0)
    #     agg_dict["young_victim_cnt"] = ("Vict Age", lambda x: int((x <= 18).sum()) if len(x) > 0 else 0)

    # if "Vict Sex" in df.columns:
    #     agg_dict["female_victim_ratio"] = ("Vict Sex", lambda x: float((x == 'F').sum() / len(x)) if len(x) > 0 else 0)

    # 轉換成 pandas NamedAgg 的格式，支援 new_col = (src_col, func)
    agg_specs: dict[str, tuple[str, str | Callable]] = {}
    for new_col, spec in agg_dict.items():
        if isinstance(spec, tuple):
            src_col, func = spec
        else:
            src_col, func = new_col, spec
        agg_specs[new_col] = (src_col, func)

    agg = (
        df.groupby(group_cols)
        .agg(**{
            new_col: pd.NamedAgg(column=src_col, aggfunc=func)
            for new_col, (src_col, func) in agg_specs.items()
        })
        .reset_index()
    )
    
    # 重命名欄位
    rename_map = {
        "ts": "cnt",
        "is_violent": "violent_cnt",
        "is_non_violent": "non_violent_cnt",
        "is_property": "property_cnt",
        "is_non_property": "non_property_cnt",
        "is_homicide": "homicide_cnt",
        "is_robbery": "robbery_cnt",
        "is_assault": "assault_cnt",
        "is_sexual": "sexual_cnt",
        "is_shooting": "shooting_cnt",
        "is_kidnap_child": "kidnap_child_cnt",
        "is_weapon": "weapon_cnt",
        "is_weekend": "weekend_ratio",
    }
    agg = agg.rename(columns=rename_map)

    print(f"   ✅ Aggregated:")
    print(f"      Rows: {len(agg):,}")
    print(f"      Grids: {agg['grid_id'].nunique():,}")
    print(f"      Date range: {agg['year_month'].min()} to {agg['year_month'].max()}")
    
    # 檢查哪些環境特徵被成功整合
    env_cols = covid_features + weather_features + ["is_holiday"]
    integrated = [c for c in env_cols if c in agg.columns]
    print(f"      Integrated env features: {len(integrated)}/{len(env_cols)}")
    if integrated:
        print(f"         → {', '.join(integrated)}")
    # ⭐ 加入這段：檢查新增特徵
    new_features = [
        "dbscan_cluster_cnt", "is_hotspot",
        "avg_victim_age", "young_victim_cnt", "female_victim_ratio",
        "adult_arrest_cnt", "juv_arrest_cnt",
        "crime_diversity", "premise_diversity"
    ]
    added = [c for c in new_features if c in agg.columns]
    if added:
        print(f"      New features added: {len(added)}")
        print(f"         → {', '.join(added)}")

    # ========= ACS features (保持原有邏輯) =========
    print(f"\n   🔧 Processing ACS features...")
    
    acs_source = df.drop_duplicates(subset=["grid_id"]).copy()

    raw_acs_cols = [
        "total_pop", "median_age",
        "white_pop", "black_pop", "asian_pop", "hispanic_pop",
        "median_income", "income_per_capita",
        "housing_units", "occupied_housing_units",
        "housing_units_renter_occupied",
        "owner_occupied_housing_units",
        "median_rent", "percent_income_spent_on_rent",
        "rent_over_50_percent", 
        "households",
        "pop_in_labor_force",
        "unemployed_pop",
        "commuters_16_over",
        "commute_35_44_mins", "commute_45_59_mins", "commute_60_more_mins",
        "male_15_to_17", "male_18_to_19",
        "female_15_to_17", "female_18_to_19",
    ]
    
    available_acs = [c for c in raw_acs_cols if c in acs_source.columns]
    print(f"      Available ACS columns: {len(available_acs)}/{len(raw_acs_cols)}")

    for col in available_acs:
        acs_source[col] = pd.to_numeric(acs_source[col], errors="coerce")
    
    acs = acs_source[["grid_id"] + available_acs].copy()

    # 安全除法與特徵工程(保持原有邏輯)
    def safe_divide(numerator, denominator, fill_value=0.0):
        denom = pd.Series(denominator).replace(0, np.nan)
        if isinstance(numerator, (int, float)):
            result = numerator / denom
        else:
            result = pd.Series(numerator) / denom
        result = result.replace([np.inf, -np.inf], np.nan).fillna(fill_value)
        return result

    def safe_get(df, col, default=0):
        if col in df.columns:
            return df[col].fillna(default)
        else:
            return pd.Series(default, index=df.index)

    pop = safe_get(acs, "total_pop", 1)
    hh = safe_get(acs, "households", 1)
    labor = safe_get(acs, "pop_in_labor_force", 1)
    commuters = safe_get(acs, "commuters_16_over", 1)
    housing = safe_get(acs, "housing_units", 1)

    # 基本特徵
    acs["acs_total_pop"] = safe_get(acs, "total_pop", 0)
    acs["acs_median_age"] = safe_get(acs, "median_age", 0)
    acs["acs_median_income"] = safe_get(acs, "median_income", 0)
    acs["acs_income_per_capita"] = safe_get(acs, "income_per_capita", 0)
    acs["acs_median_rent"] = safe_get(acs, "median_rent", 0)

    # 比例特徵
    youth_count = (
        safe_get(acs, "male_15_to_17", 0) +
        safe_get(acs, "male_18_to_19", 0) +
        safe_get(acs, "female_15_to_17", 0) +
        safe_get(acs, "female_18_to_19", 0)
    )
    acs["acs_pct_youth_15_19"] = safe_divide(youth_count, pop, 0)
    acs["acs_pct_black"] = safe_divide(safe_get(acs, "black_pop", 0), pop, 0)
    acs["acs_pct_hispanic"] = safe_divide(safe_get(acs, "hispanic_pop", 0), pop, 0)
    acs["acs_pct_asian"] = safe_divide(safe_get(acs, "asian_pop", 0), pop, 0)
    acs["acs_pct_white"] = safe_divide(safe_get(acs, "white_pop", 0), pop, 0)

    acs["acs_pct_owner_occupied"] = safe_divide(
        safe_get(acs, "owner_occupied_housing_units", 0), housing, 0
    )
    acs["acs_pct_renter_occupied"] = safe_divide(
        safe_get(acs, "housing_units_renter_occupied", 0), housing, 0
    )
    acs["acs_pct_rent_over_50"] = safe_divide(
        safe_get(acs, "rent_over_50_percent", 0), hh, 0
    )

    acs["acs_pct_unemployed"] = safe_divide(
        safe_get(acs, "unemployed_pop", 0), labor, 0
    )
    acs["acs_labor_force_participation"] = safe_divide(labor, pop, 0)

    long_commute = (
        safe_get(acs, "commute_35_44_mins", 0) +
        safe_get(acs, "commute_45_59_mins", 0) +
        safe_get(acs, "commute_60_more_mins", 0)
    )
    acs["acs_pct_commute_long"] = safe_divide(long_commute, commuters, 0)

    feature_cols = [c for c in acs.columns if c.startswith("acs_")]
    
    for col in feature_cols:
        acs[col] = pd.to_numeric(acs[col], errors="coerce")
        acs[col] = acs[col].replace([np.inf, -np.inf], 0)
        acs[col] = acs[col].fillna(0)

    keep_cols = ["grid_id"] + feature_cols
    acs_by_grid = acs[keep_cols].copy()

    print(f"      ✅ Processed {len(acs_by_grid):,} grids with {len(feature_cols)} ACS features")
    
    return agg, acs_by_grid


def make_panel_and_label(
    agg: pd.DataFrame,
    cfg: Config,
    acs_by_grid: pd.DataFrame,
    time_block_hours: int = 3,
) -> pd.DataFrame:
    """
    建立 panel with 環境特徵
    """
    print("🔧 Building panel (crime + ACS + environment)...")

    grid_counts = agg.groupby("grid_id")["cnt"].sum()
    active_grids = grid_counts[grid_counts >= 0].index

    agg = agg[agg["grid_id"].isin(active_grids)].copy()

    grids = sorted(agg["grid_id"].unique())
    months = sorted(agg["year_month"].unique())
    time_blocks = list(range(24 // time_block_hours))

    full_index = pd.MultiIndex.from_product(
        [grids, months, time_blocks],
        names=["grid_id", "year_month", "time_block"],
    )

    value_cols = [c for c in agg.columns if c not in ["grid_id", "year_month", "time_block"]]

    panel = (
        agg[["grid_id", "year_month", "time_block"] + value_cols]
        .set_index(["grid_id", "year_month", "time_block"])
        .reindex(full_index)
        .reset_index()
    )

    # 補 0 給 count 類欄位
    count_like = [
        "cnt", "violent_cnt", "non_violent_cnt",
        "property_cnt", "non_property_cnt",
        "homicide_cnt", "robbery_cnt", "assault_cnt",
        "sexual_cnt", "shooting_cnt", "kidnap_child_cnt", "weapon_cnt",
    ]
    for col in count_like:
        if col in panel.columns:
            panel[col] = panel[col].fillna(0)
    
    # 環境特徵用前向填充(假設同一天內值相同)
    env_cols = [
        "cases", "deaths", "state_cases", "state_deaths",
        "avg_daily_temp_c", "max_daily_temp_c", 
        "min_daily_temp_c", "total_daily_precip_mm",
        "is_holiday"
    ]
    # 1. 前向填充
    for col in env_cols:
        if col in panel.columns:
            panel[col] = panel.groupby("grid_id")[col].ffill().fillna(0)
    
    # 2. COVID 特徵: Log 轉換 (降低數值範圍)
    covid_cols = ['cases', 'deaths', 'state_cases', 'state_deaths']
    for col in covid_cols:
        if col in panel.columns:
            # Log(1+x) 轉換,避免 log(0)
            panel[f'{col}_log'] = np.log1p(panel[col])
            
            # ⚠️ 移除原始值(數值太大)
            panel = panel.drop(columns=[col])
            
            print(f"   ✅ Transformed {col} -> {col}_log")
    
    # 3. 天氣特徵: 保持原值(範圍合理)
    # 溫度 10-30°C, 降雨 0-50mm 都是合理範圍,不需要轉換
    weather_cols = [
        'avg_daily_temp_c', 'max_daily_temp_c', 
        'min_daily_temp_c', 'total_daily_precip_mm'
    ]
    
    for col in weather_cols:
        if col in panel.columns:
            print(f"   ✅ Kept {col} (range: {panel[col].min():.1f} ~ {panel[col].max():.1f})")
    
    # 4. 假日: 保持 binary
    if 'is_holiday' in panel.columns:
        panel['is_holiday'] = panel['is_holiday'].astype(int)
        print(f"   ✅ Kept is_holiday as binary")
    
    print(f"\n   📊 Environment features summary:")
    print(f"      COVID (log-transformed): {len([c for c in panel.columns if 'log' in c and any(x in c for x in covid_cols)])}")
    print(f"      Weather (raw): {len([c for c in weather_cols if c in panel.columns])}")
    print(f"      Holiday: {1 if 'is_holiday' in panel.columns else 0}")
    

    print(f"   ✅ Panel: {len(panel):,} rows")

    # ========= Label =========
    # threshold = 2
    # panel["y"] = (panel["cnt"] >= threshold).astype(int)
    # positive_rate = panel["y"].mean()
    # # print(f"\n📊 Label: cnt >= {threshold}, positive rate: {positive_rate*100:.2f}%")
    def label_top_quantile(panel_df: pd.DataFrame, q: float = 0.80) -> pd.DataFrame:
            def label_group(g: pd.DataFrame) -> pd.DataFrame:
                positive = g[g["cnt"] > 0]["cnt"]
                if len(positive) == 0:
                    g["y"] = 0
                    return g
                thr = positive.quantile(q)
                thr = max(thr, 1.0)  # 至少要 1 件犯罪
                g["y"] = (g["cnt"] >= thr).astype(int)
                return g
            labeled = panel_df.groupby(
                ["year_month", "time_block"], 
                group_keys=False
            ).apply(label_group)
            
            return labeled
    panel = label_top_quantile(panel, q=0.80)
    pos_rate = panel["y"].mean()
    print(f"   Global positive rate: {pos_rate*100:.2f}%")
    print(f"   Positive samples: {panel['y'].sum():,}/{len(panel):,}")
    # ========= Lag features =========
    print("\n🔧 Adding lag features...")
    
    panel["year_month_dt"] = panel["year_month"].dt.to_timestamp()
    panel = panel.sort_values(["grid_id", "time_block", "year_month_dt"]).reset_index(drop=True)

    def add_lags(gdf: pd.DataFrame) -> pd.DataFrame:
        gdf = gdf.sort_values("year_month_dt").reset_index(drop=True)

        metric_map = {
            "cnt": "total",
            "violent_cnt": "violent",
            "property_cnt": "property",
        }

        for col, suffix in metric_map.items():
            if col in gdf.columns:
                gdf[f"lag1m_{suffix}"] = gdf[col].shift(1)
                gdf[f"lag3m_{suffix}"] = gdf[col].rolling(3).sum().shift(1)
        
        # 環境特徵的 lag (COVID/天氣有時序依賴性)
        if "cases_log" in gdf.columns:
            gdf["lag1m_covid_cases"] = gdf["cases_log"].shift(1)
        
        if "avg_daily_temp_c" in gdf.columns:
            gdf["lag1m_temp"] = gdf["avg_daily_temp_c"].shift(1)

        return gdf

    panel = panel.groupby(["grid_id", "time_block"], group_keys=False).apply(add_lags)

    # ========= Temporal features =========
    panel["month"] = panel["year_month_dt"].dt.month
    panel["season"] = pd.cut(panel["month"], bins=[0, 3, 6, 9, 12], labels=[0, 1, 2, 3]).astype(int)
    panel["time_of_day"] = pd.cut(panel["time_block"], bins=[-1, 2, 4, 6, 8], labels=[0, 1, 2, 3]).astype(int)
    panel["is_night"] = (panel["time_block"] <= 2).astype(int)

    # ========= Grid stats =========
    grid_stats = panel.groupby("grid_id")["cnt"].agg(["mean", "std", "max"]).reset_index()
    grid_stats.columns = ["grid_id", "grid_mean", "grid_std", "grid_max"]
    panel = panel.merge(grid_stats, on="grid_id", how="left")

    panel = panel.sort_values(["grid_id", "time_block", "year_month_dt"]).reset_index(drop=True)
    hist = (
        panel.groupby(["grid_id", "time_block"])["cnt"]
        .expanding()
        .mean()
        .reset_index(level=[0, 1], drop=True)
        .shift(1)
    )
    panel["hist_mean"] = hist.values

    # ========= Merge ACS =========
    panel = panel.merge(acs_by_grid, on="grid_id", how="left")

    # 填充缺失值
    lag_cols = [c for c in panel.columns if "lag" in c or c == "hist_mean"]
    for col in lag_cols:
        if col in panel.columns:
            panel[col] = panel[col].fillna(0)

    if "grid_std" in panel.columns:
        panel["grid_std"] = panel["grid_std"].fillna(0)

    panel = panel.drop(columns=["year_month_dt"], errors="ignore")

    print(f"   ✅ Final panel shape: {panel.shape}")
    
    # 列出所有特徵類別
    feature_types = {
        "Crime lags": [c for c in panel.columns if c.startswith("lag") and "cnt" in c],
        "Environment": [c for c in panel.columns if c in env_cols],
        "Environment lags": [c for c in panel.columns if c.startswith("lag") and ("covid" in c or "temp" in c)],
        "ACS": [c for c in panel.columns if c.startswith("acs_")],
        "Temporal": ["month", "season", "time_block", "time_of_day", "is_night"],
        "Grid stats": ["grid_mean", "grid_std", "grid_max", "hist_mean"],
    }
    
    print("\n📋 Feature Categories:")
    for cat, cols in feature_types.items():
        available = [c for c in cols if c in panel.columns]
        if available:
            print(f"   {cat}: {len(available)} features")
    new_feature_cols = [
        "dbscan_cluster_cnt", "is_hotspot",
        "avg_victim_age", "young_victim_cnt",
        "female_victim_ratio", "adult_arrest_cnt", "juv_arrest_cnt",
        "crime_diversity", "premise_diversity"
    ]
    
    for col in new_feature_cols:
        if col in panel.columns:
            panel[col] = panel[col].fillna(0)
            
    return panel



def temporal_train_test_split(panel: pd.DataFrame, cfg: Config, test_months: int = 2):
    """
    Temporal train/test split by month
    
    Parameters:
    -----------
    test_months : int
        Number of recent months for test set (default: 2)
    """
    # Get unique months sorted
    all_months = sorted(panel["year_month"].unique())
    
    if len(all_months) < 3:
        raise ValueError(f"Need at least 3 months of data, got {len(all_months)}")
    
    # Split: last test_months for testing
    train_months = all_months[:-test_months]
    test_months_list = all_months[-test_months:]
    
    train = panel[panel["year_month"].isin(train_months)].copy()
    test = panel[panel["year_month"].isin(test_months_list)].copy()
    
    print(f"\n📊 Train/Test Split (by month):")
    print(f"   Total months: {len(all_months)}")
    print(f"   Train months: {len(train_months)} ({train_months[0]} to {train_months[-1]})")
    print(f"   Test months: {len(test_months_list)} ({test_months_list[0]} to {test_months_list[-1]})")
    print(f"   Train: {len(train):,} rows (positive: {train['y'].sum():,}, {train['y'].mean()*100:.2f}%)")
    print(f"   Test:  {len(test):,} rows (positive: {test['y'].sum():,}, {test['y'].mean()*100:.2f}%)")
    
    # Sanity checks
    if train['y'].nunique() < 2:
        raise ValueError(f"❌ Train set has only {train['y'].nunique()} class(es)!")
    if test['y'].nunique() < 2:
        print(f"   ⚠️  WARNING: Test set has only {test['y'].nunique()} class(es)")
    
    return train, test



# 在 features.py 最後加入

try:
    from boruta import BorutaPy
    BORUTA_OK = True
except Exception:
    BORUTA_OK = False

from sklearn.ensemble import RandomForestClassifier


def run_boruta_analysis(train: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    運行 Boruta feature selection 分析
    
    Returns:
    --------
    pd.DataFrame: 包含 feature, selected, ranking 欄位
    """
    
    if not BORUTA_OK:
        print("⚠️  Boruta not installed. Install with: pip install boruta")
        return None
    
    print("\n🧬 Running Boruta Feature Selection...")
    
    # 獲取特徵列表
    from .modeling import get_feature_columns
    feature_cols = get_feature_columns(train)
    
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
        print(f"   Sampling: {len(df_sample):,} rows (pos={len(pos_sample):,}, neg={len(neg_sample):,})")
    else:
        df_sample = df
        print(f"   Using full data: {len(df_sample):,} rows")
    
    X = df_sample[feature_cols].values
    y = df_sample["y"].values
    
    # Boruta with Random Forest
    print(f"   Training Boruta with {len(feature_cols)} features...")
    
    rf_estimator = RandomForestClassifier(
        n_estimators=100,  # 減少到 100（加速）
        max_depth=15,
        n_jobs=-1,
        class_weight="balanced",
        random_state=42,
    )
    
    boruta = BorutaPy(
        estimator=rf_estimator,
        n_estimators='auto',
        max_iter=getattr(cfg, 'boruta_max_iter', 50),  # 50 次迭代
        random_state=42,
        verbose=1,  # 顯示進度
    )
    
    import time
    start = time.time()
    boruta.fit(X, y)
    elapsed = time.time() - start
    
    # 整理結果
    support = boruta.support_
    ranking = boruta.ranking_
    
    selected_count = support.sum()
    print(f"\n   ✅ Boruta completed in {elapsed/60:.1f} minutes")
    print(f"   Selected: {selected_count}/{len(feature_cols)} features ({selected_count/len(feature_cols)*100:.1f}%)")
    
    # 創建結果 DataFrame
    boruta_df = pd.DataFrame({
        "feature": feature_cols,
        "selected": support,
        "ranking": ranking,
    }).sort_values(["selected", "ranking"], ascending=[False, True])
    
    # 顯示前 10 個被選中的特徵
    print(f"\n   Top 10 selected features:")
    top_10 = boruta_df[boruta_df['selected']].head(10)
    for idx, row in top_10.iterrows():
        print(f"      {row['feature']:30s} (rank={row['ranking']})")
    
    return boruta_df