# =========================================
# File: src/clustering.py
# =========================================

from __future__ import annotations
import os

import numpy as np
import pandas as pd

from .config import Config

from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False

try:
    import folium  # type: ignore
    FOLIUM_OK = True
except Exception:
    FOLIUM_OK = False


def descriptive_analysis(violent_df: pd.DataFrame, cfg: Config) -> None:
    os.makedirs(cfg.out_dir, exist_ok=True)

    violent_df["hour"] = violent_df["ts"].dt.hour
    violent_df["dow"] = violent_df["ts"].dt.dayofweek

    heat = (
        violent_df.groupby(["dow", "hour"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )

    if PLOTLY_OK:
        fig = px.density_heatmap(
            heat,
            x="hour",
            y="dow",
            z="count",
            color_continuous_scale="Inferno",
            category_orders={"dow": [0, 1, 2, 3, 4, 5, 6]},
            title="Violent incidents: Hour x Day-of-Week",
        )
        fig.update_yaxes(
            tickvals=list(range(7)),
            ticktext=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
        )
        fig.write_html(os.path.join(cfg.out_dir, "heatmap_hour_dow.html"))

    monthly = (
        violent_df.set_index("ts")
        .resample("MS")
        .size()
        .rename("count")
        .reset_index()
    )

    if PLOTLY_OK:
        fig2 = px.line(monthly, x="ts", y="count", title="Monthly violent incidents (since 2020)")
        fig2.write_html(os.path.join(cfg.out_dir, "monthly_trend.html"))


def clustering_hotspots(violent_df: pd.DataFrame, cfg: Config) -> None:
    os.makedirs(cfg.out_dir, exist_ok=True)

    recent_cut = violent_df["ts"].max() - pd.Timedelta(days=90)
    # df_recent = violent_df[violent_df["ts"] >= recent_cut].copy()
    df_recent = violent_df.copy()

    df_recent["hour"] = df_recent["ts"].dt.hour
    X = df_recent[[cfg.lat_col, cfg.lon_col, "hour"]].to_numpy()
    X_std = StandardScaler().fit_transform(X)

    Ks = list(range(cfg.kmeans_k_min, cfg.kmeans_k_max + 1))
    inertias, sils = [], []

    for k in Ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X_std)
        inertias.append(km.inertia_)

        if len(X_std) > 5000:
            idx = np.random.choice(len(X_std), 5000, replace=False)
            sil = silhouette_score(X_std[idx], labels[idx])
        else:
            sil = silhouette_score(X_std, labels)
        sils.append(sil)

    if PLOTLY_OK:
        fig_elbow = go.Figure(data=[go.Scatter(x=Ks, y=inertias, mode="lines+markers")])
        fig_elbow.update_layout(
            title="KMeans Elbow (Inertia)",
            xaxis_title="K",
            yaxis_title="Inertia",
        )
        fig_elbow.write_html(os.path.join(cfg.out_dir, "kmeans_elbow.html"))

        fig_sil = go.Figure(data=[go.Scatter(x=Ks, y=sils, mode="lines+markers")])
        fig_sil.update_layout(
            title="KMeans Silhouette",
            xaxis_title="K",
            yaxis_title="Silhouette Score",
        )
        fig_sil.write_html(os.path.join(cfg.out_dir, "kmeans_silhouette.html"))

    # DBSCAN on lat/lon
    X_ll = df_recent[[cfg.lat_col, cfg.lon_col]].to_numpy()
    db = DBSCAN(
        eps=cfg.dbscan_eps,
        min_samples=cfg.dbscan_min_samples,
        n_jobs=-1
    )

    db_labels = db.fit_predict(X_ll)
    df_recent["db_label"] = db_labels

    if FOLIUM_OK:
        center = [df_recent[cfg.lat_col].mean(), df_recent[cfg.lon_col].mean()]
        m = folium.Map(location=center, zoom_start=12)
        for _, r in df_recent.iterrows():
            color = "red" if r["db_label"] >= 0 else "gray"
            folium.CircleMarker(
                location=[r[cfg.lat_col], r[cfg.lon_col]],
                radius=2,
                color=color,
                fill=True,
                fill_opacity=0.6,
            ).add_to(m)
        m.save(os.path.join(cfg.out_dir, "dbscan_clusters_map.html"))
        print("df_recent shape =", df_recent.shape)
        print(df_recent['db_label'].value_counts())
    
    print(f"\n   ✅ DBSCAN clusters: {len(set(db_labels)) - (1 if -1 in db_labels else 0)}")
    print(f"   Noise points: {sum(db_labels == -1):,}")
    
    # 將 DBSCAN 結果映射回原始數據
    violent_df = violent_df.copy()
    violent_df["dbscan_cluster"] = -1  # 預設為噪點
    
    # 只對 recent 數據有聚類標籤
    if len(df_recent) > 0:
        violent_df.loc[df_recent.index, "dbscan_cluster"] = df_recent["db_label"]
    
    return violent_df 
