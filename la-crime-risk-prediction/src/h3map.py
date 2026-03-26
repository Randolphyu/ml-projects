# =========================================
# File: src/h3map.py (Improved Visualization)
# =========================================

import os
import folium
import numpy as np
import pandas as pd
from folium import plugins

# ✅ Auto-detect H3 version and import correct functions
H3_OK = False
_geo_to_h3 = None
_h3_to_geo_boundary = None

try:
    # Try new API (h3>=4)
    from h3 import latlng_to_cell, cell_to_boundary
    _geo_to_h3 = latlng_to_cell
    _h3_to_geo_boundary = cell_to_boundary
    H3_OK = True
    H3_VERSION = "v4"
except Exception:
    try:
        # Try old API (h3==3.7.6)
        import h3
        if hasattr(h3, "geo_to_h3") and hasattr(h3, "h3_to_geo_boundary"):
            _geo_to_h3 = h3.geo_to_h3
            _h3_to_geo_boundary = h3.h3_to_geo_boundary
            H3_OK = True
            H3_VERSION = "v3"
    except Exception:
        H3_OK = False
        H3_VERSION = None


def get_color_from_count(count, percentiles):
    """
    Map crime count to color based on percentiles
    ⭐ 更細緻的分級：從 5 級改為 9 級
    """
    p10, p25, p50, p75, p90, p95, p97, p99 = percentiles
    
    if count <= p10:
        return '#27ae60', 0.2  # 深綠 (very low)
    elif count <= p25:
        return '#2ecc71', 0.3  # 綠色 (low)
    elif count <= p50:
        return '#f1c40f', 0.4  # 黃色 (below avg)
    elif count <= p75:
        return '#e67e22', 0.5  # 橘色 (above avg)
    elif count <= p90:
        return '#e74c3c', 0.6  # 紅色 (high)
    elif count <= p95:
        return '#c0392b', 0.7  # 深紅 (very high)
    elif count <= p97:
        return '#a93226', 0.75  # 暗紅 (extreme)
    elif count <= p99:
        return '#8b0000', 0.85  # 深暗紅 (critical)
    else:
        return '#5c0000', 0.95  # 黑紅 (超級熱點!)


def generate_h3_hotspot_map(violent_df: pd.DataFrame, out_path: str, h3_res: int = 9, 
                           lat_col: str = "LAT", lon_col: str = "LON",
                           show_only_top_percent: float = None):
    """
    Generate an improved H3 hexagon hotspot map with finer color grading.
    
    Parameters:
    -----------
    violent_df : pd.DataFrame
        DataFrame containing latitude and longitude columns
    out_path : str
        Output path for the HTML map
    h3_res : int
        H3 resolution (default: 9, ~0.1km² hexagons)
    lat_col : str
        Name of latitude column
    lon_col : str
        Name of longitude column
    show_only_top_percent : float, optional
        Only show top X% of hexagons (e.g., 0.2 for top 20%)
    """
    if not H3_OK:
        print("⚠️ H3 is not installed. Skipping H3 hotspot map.")
        print("   Install with: pip install h3")
        return

    print(f"🔍 Generating H3 hotspot map (using H3 {H3_VERSION})...")

    # Find correct column names (case-insensitive)
    cols_lower = {c.lower(): c for c in violent_df.columns}
    
    lat_actual = None
    lon_actual = None
    
    for possible_lat in ["lat", "latitude", "y", lat_col.lower()]:
        if possible_lat in cols_lower:
            lat_actual = cols_lower[possible_lat]
            break
    
    for possible_lon in ["lon", "lng", "long", "longitude", "x", lon_col.lower()]:
        if possible_lon in cols_lower:
            lon_actual = cols_lower[possible_lon]
            break
    
    if lat_actual is None or lon_actual is None:
        print(f"❌ Error: Could not find lat/lon columns")
        print(f"   Available columns: {list(violent_df.columns)}")
        return

    print(f"   Using columns: lat='{lat_actual}', lon='{lon_actual}'")

    # Compute H3 index for each incident
    def safe_geo_to_h3(lat, lon):
        try:
            return _geo_to_h3(float(lat), float(lon), h3_res)
        except Exception:
            return None

    violent_df = violent_df.copy()
    violent_df["hex_id"] = violent_df.apply(
        lambda r: safe_geo_to_h3(r[lat_actual], r[lon_actual]),
        axis=1
    )

    # Remove invalid hex IDs
    violent_df = violent_df[violent_df["hex_id"].notna()].copy()
    
    if len(violent_df) == 0:
        print("❌ No valid hex IDs generated. Check your coordinates.")
        return

    # Aggregate
    agg = violent_df.groupby("hex_id").size().reset_index(name="count")
    
    # ⭐ 計算更細緻的百分位數 (9個級別)
    percentiles = [
        agg["count"].quantile(0.10),
        agg["count"].quantile(0.25),
        agg["count"].quantile(0.50),
        agg["count"].quantile(0.75),
        agg["count"].quantile(0.90),
        agg["count"].quantile(0.95),
        agg["count"].quantile(0.97),
        agg["count"].quantile(0.99),
    ]
    
    print(f"   Total hexagons: {len(agg):,}")
    print(f"   Crime count distribution:")
    print(f"      10th percentile: {percentiles[0]:.0f}")
    print(f"      25th percentile: {percentiles[1]:.0f}")
    print(f"      50th percentile (median): {percentiles[2]:.0f}")
    print(f"      75th percentile: {percentiles[3]:.0f}")
    print(f"      90th percentile: {percentiles[4]:.0f}")
    print(f"      95th percentile: {percentiles[5]:.0f}")
    print(f"      97th percentile: {percentiles[6]:.0f}")
    print(f"      99th percentile: {percentiles[7]:.0f}")
    print(f"      Maximum: {agg['count'].max():.0f}")
    
    # Optional: Filter to show only top X% hotspots
    if show_only_top_percent:
        threshold = agg["count"].quantile(1 - show_only_top_percent)
        agg = agg[agg["count"] >= threshold].copy()
        print(f"   Filtering to top {show_only_top_percent*100:.0f}% hotspots")
        print(f"   Showing {len(agg):,} hexagons (threshold: {threshold:.0f} crimes)")

    # Create map centered at data centroid
    center_lat = violent_df[lat_actual].mean()
    center_lon = violent_df[lon_actual].mean()
    m = folium.Map(
        location=[center_lat, center_lon], 
        zoom_start=11,
        tiles='CartoDB positron'
    )

    # Add hexagons with improved coloring
    for _, r in agg.iterrows():
        hex_id = r["hex_id"]
        count = r["count"]
        
        # Get boundary
        try:
            if H3_VERSION == "v4":
                boundary = _h3_to_geo_boundary(hex_id)
            else:
                boundary = _h3_to_geo_boundary(hex_id, geo_json=True)
        except Exception:
            continue

        # Get color based on percentile
        color, opacity = get_color_from_count(count, percentiles)

        # Enhanced tooltip with more info
        percentile_rank = (agg['count'] <= count).sum() / len(agg) * 100
        tooltip = f"""
        <b>Crimes: {int(count)}</b><br>
        Percentile: {percentile_rank:.1f}%<br>
        Hex ID: {hex_id[:10]}...
        """

        folium.Polygon(
            locations=boundary,
            color=color,
            weight=1.5,
            fill=True,
            fill_color=color,
            fill_opacity=opacity,
            tooltip=tooltip,
        ).add_to(m)

    # ⭐ 更新圖例 - 9 個級別
    legend_html = f'''
    <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 260px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:11px; padding: 15px; border-radius: 5px;">
    <p style="margin: 0 0 10px 0"><strong style="font-size:14px">Crime Hotspots</strong></p>
    <p style="margin: 5px 0"><b>H3 Resolution:</b> {h3_res}</p>
    <p style="margin: 5px 0"><b>Total Hexagons:</b> {len(agg):,}</p>
    <p style="margin: 5px 0"><b>Max Crimes:</b> {agg['count'].max():.0f}</p>
    
    <hr style="margin: 10px 0">
    <p style="margin: 5px 0; font-size:11px"><b>Crime Count Ranges:</b></p>
    
    <div style="margin: 3px 0">
        <span style="background-color: #27ae60; padding: 2px 6px; border-radius: 3px; margin-right: 5px;">■</span>
        <span style="font-size:10px">Very Low (0-{percentiles[0]:.0f})</span>
    </div>
    <div style="margin: 3px 0">
        <span style="background-color: #2ecc71; padding: 2px 6px; border-radius: 3px; margin-right: 5px;">■</span>
        <span style="font-size:10px">Low ({percentiles[0]:.0f}-{percentiles[1]:.0f})</span>
    </div>
    <div style="margin: 3px 0">
        <span style="background-color: #f1c40f; padding: 2px 6px; border-radius: 3px; margin-right: 5px;">■</span>
        <span style="font-size:10px">Below Avg ({percentiles[1]:.0f}-{percentiles[2]:.0f})</span>
    </div>
    <div style="margin: 3px 0">
        <span style="background-color: #e67e22; padding: 2px 6px; border-radius: 3px; margin-right: 5px;">■</span>
        <span style="font-size:10px">Above Avg ({percentiles[2]:.0f}-{percentiles[3]:.0f})</span>
    </div>
    <div style="margin: 3px 0">
        <span style="background-color: #e74c3c; padding: 2px 6px; border-radius: 3px; margin-right: 5px;">■</span>
        <span style="font-size:10px">High ({percentiles[3]:.0f}-{percentiles[4]:.0f})</span>
    </div>
    <div style="margin: 3px 0">
        <span style="background-color: #c0392b; padding: 2px 6px; border-radius: 3px; margin-right: 5px;">■</span>
        <span style="font-size:10px">Very High ({percentiles[4]:.0f}-{percentiles[5]:.0f})</span>
    </div>
    <div style="margin: 3px 0">
        <span style="background-color: #a93226; padding: 2px 6px; border-radius: 3px; margin-right: 5px; color: white;">■</span>
        <span style="font-size:10px">Extreme ({percentiles[5]:.0f}-{percentiles[6]:.0f})</span>
    </div>
    <div style="margin: 3px 0">
        <span style="background-color: #8b0000; padding: 2px 6px; border-radius: 3px; margin-right: 5px; color: white;">■</span>
        <span style="font-size:10px">Critical ({percentiles[6]:.0f}-{percentiles[7]:.0f})</span>
    </div>
    <div style="margin: 3px 0">
        <span style="background-color: #5c0000; padding: 2px 6px; border-radius: 3px; margin-right: 5px; color: white;">■</span>
        <span style="font-size:10px"><b>Top 1% (>{percentiles[7]:.0f})</b></span>
    </div>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    # Add fullscreen button
    plugins.Fullscreen().add_to(m)

    # Save
    m.save(out_path)
    print(f"✅ H3 hotspot map saved: {out_path}")
    
    # ⭐ 顯示 Top 1% 的超級熱點
    top_1_percent = agg.nlargest(max(1, int(len(agg) * 0.01)), 'count')
    print(f"\n🔥 Top 1% Super Hotspots ({len(top_1_percent)} hexagons):")
    for idx, (_, row) in enumerate(top_1_percent.iterrows(), 1):
        print(f"   #{idx}: {int(row['count'])} crimes in hex {row['hex_id'][:10]}...")


# Convenience function for generating multiple maps
def generate_all_hotspot_maps(df: pd.DataFrame, cfg, lat_col: str, lon_col: str):
    """
    Generate multiple versions of hotspot maps for comparison
    """
    out_dir = cfg.out_dir
    
    # 1. Full map with all hexagons
    print("\n" + "="*60)
    print("Generating FULL hotspot map...")
    print("="*60)
    generate_h3_hotspot_map(
        df,
        out_path=os.path.join(out_dir, "h3_hotspot_full.html"),
        h3_res=9,
        lat_col=lat_col,
        lon_col=lon_col
    )
    
    # 2. Top 20% hotspots only
    print("\n" + "="*60)
    print("Generating TOP 20% hotspot map...")
    print("="*60)
    generate_h3_hotspot_map(
        df,
        out_path=os.path.join(out_dir, "h3_hotspot_top20.html"),
        h3_res=9,
        lat_col=lat_col,
        lon_col=lon_col,
        show_only_top_percent=0.20
    )
    
    # 3. Top 10% hotspots only
    print("\n" + "="*60)
    print("Generating TOP 10% hotspot map...")
    print("="*60)
    generate_h3_hotspot_map(
        df,
        out_path=os.path.join(out_dir, "h3_hotspot_top10.html"),
        h3_res=9,
        lat_col=lat_col,
        lon_col=lon_col,
        show_only_top_percent=0.10
    )
    
    # 4. ⭐ 新增: Top 5% 超級熱點
    print("\n" + "="*60)
    print("Generating TOP 5% hotspot map (Super Critical)...")
    print("="*60)
    generate_h3_hotspot_map(
        df,
        out_path=os.path.join(out_dir, "h3_hotspot_top5.html"),
        h3_res=9,
        lat_col=lat_col,
        lon_col=lon_col,
        show_only_top_percent=0.05
    )
    
    # 5. ⭐ 新增: Top 1% 極端熱點
    print("\n" + "="*60)
    print("Generating TOP 1% hotspot map (Extreme)...")
    print("="*60)
    generate_h3_hotspot_map(
        df,
        out_path=os.path.join(out_dir, "h3_hotspot_top1.html"),
        h3_res=9,
        lat_col=lat_col,
        lon_col=lon_col,
        show_only_top_percent=0.01
    )
    
    print("\n" + "="*60)
    print("✅ All hotspot maps generated!")
    print("="*60)


# Test function
def test_h3():
    """Test H3 installation and version"""
    if H3_OK:
        print(f"✅ H3 {H3_VERSION} installed correctly")
        test_lat, test_lon = 34.0522, -118.2437
        test_hex = _geo_to_h3(test_lat, test_lon, 9)
        print(f"   Test hex for LA downtown: {test_hex}")
        boundary = _h3_to_geo_boundary(test_hex)
        print(f"   Boundary points: {len(boundary)}")
        return True
    else:
        print("❌ H3 not installed")
        return False


if __name__ == "__main__":
    test_h3()