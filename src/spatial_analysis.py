from __future__ import annotations

from pathlib import Path
import importlib
import re

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
from branca.colormap import linear

from src.config import CITY_FOLDERS, PLOTS_DIR, PROCESSED_DIR, TABLES_DIR, normalize_city_name


BASE_DIR = Path(__file__).resolve().parent.parent


def _find_geojson_file(city_dir: Path) -> Path:
    candidates = [
        city_dir / "neighbourhoods.geojson",
        city_dir / "neighbourhoods (1).geojson",
    ]
    for path in candidates:
        if path.exists() and path.is_file():
            return path
    matches = list(city_dir.glob("neighbourhoods*.geojson"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not find neighbourhood geojson under {city_dir}")


CITY_META = {
    city_key: {
        "clustered_file": f"{city_key}_clustered.csv",
        "geojson": _find_geojson_file(BASE_DIR / folder),
    }
    for city_key, folder in CITY_FOLDERS.items()
}


def _norm_name(series: pd.Series) -> pd.Series:
    normalized = series.astype(str).str.lower().str.strip()
    # Canonicalize punctuation/spacing so map joins are stable across data sources.
    normalized = normalized.str.replace(r"[^a-z0-9]+", " ", regex=True)
    return normalized.str.replace(r"\s+", " ", regex=True).str.strip()


def _norm_text(value: object) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


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
    avg_price_raw = (
        city_df.dropna(subset=["neighbourhood", "price"])
        .groupby("neighbourhood", as_index=False)["price"]
        .mean()
        .rename(columns={"price": "avg_price"})
    )

    avg_price_raw["neighbourhood_key"] = _norm_name(avg_price_raw["neighbourhood"])

    # Enforce unique join keys to avoid one-to-many merges that miscolor polygons.
    avg_price = (
        avg_price_raw.groupby("neighbourhood_key", as_index=False)
        .agg(avg_price=("avg_price", "mean"), neighbourhood=("neighbourhood", "first"))
    )

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


def _weights_to_matrix(weights: object, n: int) -> np.ndarray:
    """Convert supported spatial weights formats to an n x n weight matrix."""
    if isinstance(weights, pd.DataFrame):
        W = weights.to_numpy(dtype=float)
    elif isinstance(weights, np.ndarray):
        W = weights.astype(float)
    elif isinstance(weights, dict):
        # Expected format: {i: [j1, j2, ...]} with integer node indices.
        W = np.zeros((n, n), dtype=float)
        for i, neighbours in weights.items():
            i_int = int(i)
            for j in neighbours:
                j_int = int(j)
                if 0 <= i_int < n and 0 <= j_int < n and i_int != j_int:
                    W[i_int, j_int] = 1.0
    else:
        raise TypeError(
            "weights must be a pandas DataFrame, numpy.ndarray, or dict adjacency list."
        )

    if W.shape != (n, n):
        raise ValueError(f"weights matrix must have shape ({n}, {n}), got {W.shape}.")

    np.fill_diagonal(W, 0.0)
    return W


def _manual_morans_i(x: np.ndarray, W: np.ndarray) -> float:
    """Compute Moran's I directly from vector x and weight matrix W."""
    n = x.size
    x_centered = x - x.mean()
    denom = float(np.sum(x_centered**2))
    s0 = float(W.sum())

    if n < 2 or denom <= 0 or s0 <= 0:
        return float("nan")

    num = float(x_centered @ W @ x_centered)
    return float((n / s0) * (num / denom))


def _queen_weights_for_valid_prices(
    merged_gdf: gpd.GeoDataFrame,
    price_col: str = "avg_price",
) -> tuple[np.ndarray, np.ndarray]:
    """Build queen-contiguity weight matrix for rows with valid prices.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (prices_vector, weights_matrix) aligned on valid-price neighbourhoods.
    """
    work = merged_gdf.copy()
    work[price_col] = pd.to_numeric(work[price_col], errors="coerce")
    work = work[work[price_col].notna()].reset_index(drop=True)

    n = len(work)
    if n == 0:
        return np.array([], dtype=float), np.zeros((0, 0), dtype=float)
    if n == 1:
        return work[price_col].to_numpy(dtype=float), np.zeros((1, 1), dtype=float)

    W = np.zeros((n, n), dtype=float)
    sindex = work.sindex

    for i, geom in enumerate(work.geometry):
        if geom is None or geom.is_empty:
            continue
        candidates = list(sindex.intersection(geom.bounds))
        for j in candidates:
            if i == j:
                continue
            other = work.geometry.iloc[j]
            if other is None or other.is_empty:
                continue
            # Queen contiguity: share border or point.
            if geom.touches(other) or geom.intersects(other):
                W[i, j] = 1.0

    np.fill_diagonal(W, 0.0)
    prices = work[price_col].to_numpy(dtype=float)
    return prices, W


def compute_morans_i(
    price_array: object,
    spatial_weights: object,
    permutations: int = 999,
    alpha: float = 0.05,
) -> dict[str, float | bool]:
    """Compute Moran's I and significance for neighbourhood-level average prices.

    Parameters
    ----------
    price_array:
        Array-like numeric prices ordered consistently with the spatial weights.
    spatial_weights:
        Spatial weight structure as one of:
        - n x n pandas DataFrame
        - n x n numpy.ndarray
        - dict adjacency list with integer keys/values
    permutations:
        Number of random permutations for empirical p-value.
    alpha:
        Significance threshold for the returned boolean flag.
    """
    prices = pd.to_numeric(pd.Series(price_array), errors="coerce")
    valid_mask = np.isfinite(prices.to_numpy(dtype=float))
    prices = prices[valid_mask].astype(float)

    if prices.empty:
        raise ValueError("price_array has no valid numeric values.")

    x = prices.to_numpy(dtype=float)
    n = x.size

    W_full = _weights_to_matrix(spatial_weights, len(valid_mask))
    W = W_full[np.ix_(valid_mask, valid_mask)]

    # Row-standardize while preserving zero-row handling.
    row_sums = W.sum(axis=1, keepdims=True)
    W = np.divide(W, row_sums, out=np.zeros_like(W), where=row_sums > 0)

    expected_i = float(-1.0 / (n - 1)) if n > 1 else float("nan")

    try:
        esda = importlib.import_module("esda")
        Moran = getattr(esda, "Moran")
        libpysal_weights = importlib.import_module("libpysal.weights")
        WSP = getattr(libpysal_weights, "WSP")
        scipy_sparse = importlib.import_module("scipy.sparse")
        csr_matrix = getattr(scipy_sparse, "csr_matrix")

        pysal_w = WSP(csr_matrix(W)).to_W()
        moran = Moran(x, pysal_w, permutations=permutations)

        morans_i = float(moran.I)
        p_value = float(moran.p_sim)
        z_score = float(moran.z_sim)
    except Exception:
        # Manual fallback with permutation-based two-sided p-value.
        morans_i = _manual_morans_i(x, W)
        if not np.isfinite(morans_i):
            return {
                "morans_i": float("nan"),
                "expected_i": expected_i,
                "p_value": float("nan"),
                "z_score": float("nan"),
                "significant": False,
            }

        rng = np.random.default_rng(42)
        permuted = np.empty(permutations, dtype=float)
        for idx in range(permutations):
            permuted[idx] = _manual_morans_i(rng.permutation(x), W)

        perm_mean = float(np.nanmean(permuted))
        perm_std = float(np.nanstd(permuted, ddof=1)) if permutations > 1 else float("nan")
        if np.isfinite(perm_std) and perm_std > 0:
            z_score = float((morans_i - perm_mean) / perm_std)
        else:
            z_score = float("nan")

        abs_obs = abs(morans_i)
        abs_perm = np.abs(permuted)
        p_value = float((np.sum(abs_perm >= abs_obs) + 1.0) / (permutations + 1.0))

    return {
        "morans_i": morans_i,
        "expected_i": expected_i,
        "p_value": p_value,
        "z_score": z_score,
        "significant": bool(np.isfinite(p_value) and p_value < alpha),
    }


def classify_morans_i(
    morans_i: float,
    p_value: float,
    alpha: float = 0.05,
) -> str:
    """Classify spatial autocorrelation strength from Moran's I and p-value."""
    if not np.isfinite(morans_i) or not np.isfinite(p_value):
        return "insufficient_data"
    if p_value >= alpha:
        return "not_significant"
    if morans_i < 0:
        return "significant_negative_dispersion"
    if morans_i < 0.2:
        return "significant_positive_weak"
    if morans_i < 0.4:
        return "significant_positive_moderate"
    return "significant_positive_strong"


def create_price_choropleth(
    city_key: str,
    merged_gdf: gpd.GeoDataFrame,
    center: list[float],
    output_dir: Path | None = None,
) -> Path:
    map_obj = folium.Map(location=center, zoom_start=10, tiles="CartoDB positron")

    available_prices = merged_gdf["avg_price"].dropna()
    if available_prices.empty:
        raise ValueError(f"No neighbourhood prices available for {city_key} choropleth.")

    colormap = linear.YlOrRd_09.scale(float(available_prices.min()), float(available_prices.max()))
    colormap.caption = f"Average Listing Price ({city_key.title()})"

    def style_fn(feature: dict) -> dict:
        props = feature.get("properties", {})
        value = pd.to_numeric(pd.Series([props.get("avg_price")]), errors="coerce").iloc[0]
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

    out_dir = output_dir or PLOTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{city_key}_price_choropleth.html"
    map_obj.save(out_path)
    return out_path


def create_cluster_map(
    city_key: str,
    city_df: pd.DataFrame,
    center: list[float],
    output_dir: Path | None = None,
) -> Path:
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

    out_dir = output_dir or PLOTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{city_key}_cluster_map.html"
    map_obj.save(out_path)
    return out_path


def run_spatial_analysis(
    city_name: str | None = None,
    plots_output_dir: Path | None = None,
    tables_output_dir: Path | None = None,
) -> tuple[list[Path], Path]:
    plots_dir = plots_output_dir or PLOTS_DIR
    tables_dir = tables_output_dir or TABLES_DIR
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    saved_maps: list[Path] = []
    summary_rows: list[dict] = []

    city_keys = list(CITY_META.keys())
    if city_name is not None:
        city_keys = [normalize_city_name(city_name)]

    for city_key in city_keys:
        if city_key not in CITY_META:
            raise ValueError(f"Unsupported city: {city_name}")
        city_df, geo_gdf = _load_city_inputs(city_key)
        center = _map_center_from_df(city_df)

        merged_gdf, avg_price_table = _build_neighbourhood_price_merge(city_df, geo_gdf)

        choropleth_path = create_price_choropleth(
            city_key,
            merged_gdf,
            center,
            output_dir=plots_dir,
        )
        cluster_map_path = create_cluster_map(
            city_key,
            city_df,
            center,
            output_dir=plots_dir,
        )
        saved_maps.extend([choropleth_path, cluster_map_path])

        matched = merged_gdf["avg_price"].notna().sum()
        coverage = matched / len(merged_gdf) if len(merged_gdf) else 0.0

        price_vector, weights_matrix = _queen_weights_for_valid_prices(merged_gdf, price_col="avg_price")
        if len(price_vector) >= 2:
            moran_stats = compute_morans_i(
                price_vector,
                weights_matrix,
                permutations=999,
                alpha=0.05,
            )
        else:
            moran_stats = {
                "morans_i": float("nan"),
                "expected_i": float("nan"),
                "p_value": float("nan"),
                "z_score": float("nan"),
                "significant": False,
            }

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
                "morans_i": moran_stats["morans_i"],
                "morans_i_expected": moran_stats["expected_i"],
                "morans_i_p_value": moran_stats["p_value"],
                "morans_i_z_score": moran_stats["z_score"],
                "morans_i_significant": moran_stats["significant"],
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    suffix = f"_{normalize_city_name(city_name)}" if city_name else ""

    if not summary_df.empty:
        summary_df["spatial_cluster_class"] = summary_df.apply(
            lambda row: classify_morans_i(
                float(row["morans_i"]),
                float(row["morans_i_p_value"]),
            ),
            axis=1,
        )

    summary_path = tables_dir / f"spatial_summary{suffix}.csv"
    summary_df.to_csv(summary_path, index=False)

    classification_cols = [
        "city",
        "morans_i",
        "morans_i_p_value",
        "morans_i_significant",
        "spatial_cluster_class",
    ]
    classification_df = summary_df[classification_cols].copy()
    classification_path = tables_dir / f"spatial_clustering_classification{suffix}.csv"
    classification_df.to_csv(classification_path, index=False)

    return saved_maps, summary_path


def main() -> None:
    maps, summary_path = run_spatial_analysis()
    print("Spatial maps saved:")
    for path in maps:
        print(path)
    print(f"Spatial summary table saved: {summary_path}")


if __name__ == "__main__":
    main()
