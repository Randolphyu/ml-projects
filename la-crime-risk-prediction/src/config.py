# =========================================
# File: src/config.py
# =========================================

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    # Paths
    csv_path: str = "data/crime_with_acs_and_more.csv"
    out_dir: str = "outputs"

    # Raw columns
    date_col: str = "DATE OCC"
    time_col: str = "TIME OCC"
    lat_col: str = "lat"
    lon_col: str = "lon"
    violent_col: str = "Crm Cd Desc"  # description-based violent flag
    is_violent_col: Optional[str] = None  # if you later create a boolean violent column

    # Violent crime regex on description (fallback)
    '''violent_regex: str = (
        r"(HOMICIDE|MURDER|ROBBERY|ASSAULT|AGGRAVATED|RAPE|SEXUAL ASSAULT|SHOOTING|"  # noqa: E501
        r"BATTERY W/ SERIOUS|ADW|SHOTS FIRED)"
    )'''
    violent_regex: str = (
    r"(HOMICIDE|MURDER|MANSLAUGHTER|"  # 殺人
    r"ROBBERY|ARMED ROBBERY|"           # 搶劫
    r"ASSAULT|AGGRAVATED|BATTERY|ADW|"  # 攻擊 (包含所有 battery)
    r"RAPE|SEXUAL|SODOMY|PENETRATION|"  # 性犯罪
    r"SHOOTING|SHOTS FIRED|DISCHARGE|"  # 槍擊
    r"KIDNAP|CHILD ABUSE|"              # 綁架/虐童
    r"WEAPON|DEADLY)"                   # 武器相關
    )
    # Time filter
    since: str = "2022-01-01"  # use only incidents since 2020
    end: str = "2023-12-31"

    # Grid settings
    grid: str = "h3"  # "h3" or "bin"
    h3_res: int = 9
    bin_deg: float = 0.005  # ~500m at LA latitude if using bins

    # Clustering
    kmeans_k_min: int = 3
    kmeans_k_max: int = 10
    dbscan_eps: float = 0.0025  # ~250m in degrees
    dbscan_min_samples: int = 15

    # Labeling high-risk grid-cell-hours
    label_rule: str = "count>=1"  # "quantile" or "count>=1"
    risk_quantile: float = 0.95

    # Train/test split
    test_size_days: int = 30

    # Modeling
    use_xgboost: bool = True

    # ---- CV / Feature selection / ALE ----
    use_time_series_cv: bool = True   # 開啟 TimeSeriesSplit 評估
    cv_n_splits: int = 5              # 幾折 CV

    use_boruta: bool = True                 # run Boruta feature selection
    run_boruta_analysis: bool = True        # save Boruta ranking to CSV
    use_boruta_for_training: bool = False   # if False: analyse only, train on all features
    use_boruta_for_selection: bool = False  # alias kept for backward-compat
    boruta_sample_size: int = 50000         # max rows fed to Boruta (memory guard)
    boruta_max_iter: int = 50               # Boruta max iterations

    use_ale: bool = True                    # generate ALE plots
