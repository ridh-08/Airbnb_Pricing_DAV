from __future__ import annotations

from pathlib import Path

import folium
import geopandas as gpd
import pandas as pd
from branca.colormap import linear


BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PLOTS_DIR = BASE_DIR / "outputs" / "plots"
TABLES_DIR = BASE_DIR / "outputs" / "tables"

CITY_META = {
    "newyork": {
        "clustered_file": "ny_clustered.csv",
        "geojson": BASE_DIR / "New York" / "neighbourhoods.geojson",
    },
    "chicago": {
        "clustered_file": "chicago_clustered.csv",
        "geojson": BASE_DIR / "Chicago" / "neighbourhoods.geojson",
    },
}


def _norm_name(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower()


def _load_city_inputs(city_key: str) -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
    meta = CITY_META[city_key]
    clustered_path = PROCESSED_DIR / meta["clustered_file"]
    if not clustered_path.exists():
        raise FileNotFoundError(
            f"Missing clustered file for {city_key}: {clustered_path}. Run clustering.py first."
        )

    df = pd.read_csv(clustered_path)
    gdf = gpd.read_file(meta["geojson"])
    return df, gdf


def _build_neighbourhood_price_merge(
    city_df: pd.DataFrame,
    geo_gdf: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    avg_price = (
        city_df.dropna(subset=["neighbourhood", "price"])
        .groupby("neighbourhood", as_index=False)["price"]
        .mean()
        .rename(columns={"price": "avg_price"})
    )

    avg_price["neighbourhood_key"] = _norm_name(avg_price["neighbourhood"])

    merged = geo_gdf.copy()
    merged["neighbourhood_key"] = _norm_name(merged["neighbourhood"])
    merged = merged.merge(
        avg_price[["neighbourhood_key", "avg_price"]],
        on="neighbourhood_key",
        how="left",
    )

    return merged, avg_price


def _map_center_from_df(df: pd.DataFrame) -> list[float]:
    lat = pd.to_numeric(df["latitude"], errors="coerce")
    lon = pd.to_numeric(df["longitude"], errors="coerce")
    return [float(lat.mean()), float(lon.mean())]


def create_price_choropleth(city_key: str, merged_gdf: gpd.GeoDataFrame, center: list[float]) -> Path:
    map_obj = folium.Map(location=center, zoom_start=10, tiles="CartoDB positron")

    available_prices = merged_gdf["avg_price"].dropna()
    if available_prices.empty:
        raise ValueError(f"No neighbourhood prices available for {city_key} choropleth.")

    colormap = linear.YlOrRd_09.scale(float(available_prices.min()), float(available_prices.max()))
    colormap.caption = f"Average Listing Price ({city_key.title()})"

    value_lookup = {
        row["neighbourhood_key"]: row["avg_price"]
        for _, row in merged_gdf[["neighbourhood_key", "avg_price"]].drop_duplicates().iterrows()
    }

    def style_fn(feature: dict) -> dict:
        key = str(feature["properties"].get("neighbourhood", "")).strip().lower()
        value = value_lookup.get(key)
        if pd.isna(value):
            return {
                "fillColor": "#d3d3d3",
                "color": "#8a8a8a",
                "weight": 0.6,
                "fillOpacity": 0.35,
            }
        return {
            "fillColor": colormap(float(value)),
            "color": "#4f4f4f",
            "weight": 0.7,
            "fillOpacity": 0.7,
        }

    tooltip = folium.features.GeoJsonTooltip(
        fields=["neighbourhood", "avg_price"],
        aliases=["Neighbourhood", "Average Price"],
        localize=True,
        sticky=False,
    )

    folium.GeoJson(
        merged_gdf,
        style_function=style_fn,
        tooltip=tooltip,
        name="Avg Price",
    ).add_to(map_obj)

    colormap.add_to(map_obj)
    folium.LayerControl().add_to(map_obj)

    out_path = PLOTS_DIR / f"{city_key}_price_choropleth.html"
    map_obj.save(out_path)
    return out_path


def create_cluster_map(city_key: str, city_df: pd.DataFrame, center: list[float]) -> Path:
    map_obj = folium.Map(location=center, zoom_start=10, tiles="CartoDB positron")

    points = city_df.dropna(subset=["cluster", "latitude", "longitude"]).copy()
    points["cluster"] = pd.to_numeric(points["cluster"], errors="coerce")
    points = points.dropna(subset=["cluster"]) 
    points["cluster"] = points["cluster"].astype(int)

    # Keep map performant for browsers with large city datasets.
    if len(points) > 8000:
        points = points.sample(8000, random_state=42)

    palette = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02"]
    clusters = sorted(points["cluster"].unique().tolist())
    color_map = {cluster: palette[i % len(palette)] for i, cluster in enumerate(clusters)}

    for _, row in points.iterrows():
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=3,
            color=color_map[row["cluster"]],
            fill=True,
            fill_opacity=0.65,
            popup=(
                f"Cluster: {row['cluster']}<br>"
                f"Price: {row.get('price', 'NA')}<br>"
                f"Room Type: {row.get('room_type', 'NA')}"
            ),
        ).add_to(map_obj)

    out_path = PLOTS_DIR / f"{city_key}_cluster_map.html"
    map_obj.save(out_path)
    return out_path


def run_spatial_analysis() -> tuple[list[Path], Path]:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    saved_maps: list[Path] = []
    summary_rows: list[dict] = []

    for city_key in CITY_META:
        city_df, geo_gdf = _load_city_inputs(city_key)
        center = _map_center_from_df(city_df)

        merged_gdf, avg_price_table = _build_neighbourhood_price_merge(city_df, geo_gdf)

        choropleth_path = create_price_choropleth(city_key, merged_gdf, center)
        cluster_map_path = create_cluster_map(city_key, city_df, center)
        saved_maps.extend([choropleth_path, cluster_map_path])

        matched = merged_gdf["avg_price"].notna().sum()
        coverage = matched / len(merged_gdf) if len(merged_gdf) else 0.0

        top_row = avg_price_table.sort_values("avg_price", ascending=False).head(1)
        top_neigh = top_row["neighbourhood"].iloc[0] if not top_row.empty else "NA"
        top_price = float(top_row["avg_price"].iloc[0]) if not top_row.empty else float("nan")

        summary_rows.append(
            {
                "city": city_key,
                "geo_neighbourhoods": int(len(merged_gdf)),
                "matched_neighbourhoods": int(matched),
                "coverage_rate": round(float(coverage), 4),
                "top_priced_neighbourhood": top_neigh,
                "top_avg_price": top_price,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = TABLES_DIR / "spatial_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    return saved_maps, summary_path


def main() -> None:
    maps, summary_path = run_spatial_analysis()
    print("Spatial maps saved:")
    for path in maps:
        print(path)
    print(f"Spatial summary table saved: {summary_path}")


if __name__ == "__main__":
    main()
