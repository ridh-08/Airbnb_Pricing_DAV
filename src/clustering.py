from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.config import CITY_COLORS, PROCESSED_DIR, TABLES_DIR, PLOTS_DIR, discover_processed_city_files, normalize_city_name


BASE_DIR = Path(__file__).resolve().parent.parent

CLUSTER_FEATURES = [
    "log_price",
    "availability_365",
    "number_of_reviews",
    "amenities_count",
]

DBSCAN_FEATURES = ["latitude", "longitude", "log_price", "amenities_count"]
GMM_FEATURES = CLUSTER_FEATURES.copy()

WINSOR_LOWER_Q = 0.01
WINSOR_UPPER_Q = 0.99


def _prepare_cluster_input(
    df: pd.DataFrame,
    feature_cols: list[str],
    city: str | None = None,
) -> pd.DataFrame:
    """Return model-ready frame for selected features and optional city filter."""
    model_df = df.copy()
    if city is not None and "city" in model_df.columns:
        model_df = model_df[model_df["city"].astype(str).str.lower() == city.lower()].copy()

    for col in feature_cols:
        model_df[col] = pd.to_numeric(model_df[col], errors="coerce")

    model_df = model_df.dropna(subset=feature_cols)
    return model_df


def _winsorize_features(model_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Clip extreme tails to make clustering less sensitive to outliers."""
    clipped = model_df.copy()
    for col in feature_cols:
        lower = clipped[col].quantile(WINSOR_LOWER_Q)
        upper = clipped[col].quantile(WINSOR_UPPER_Q)
        clipped[col] = clipped[col].clip(lower=lower, upper=upper)
    return clipped


def _select_best_k(features_scaled: pd.DataFrame, k_values: list[int]) -> tuple[int, dict[int, float]]:
    """Evaluate candidate k values and return the best by silhouette score."""
    scores: dict[int, float] = {}
    for k in k_values:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(features_scaled)
        score = silhouette_score(features_scaled, labels)
        scores[k] = score

    best_k = max(scores, key=scores.get)
    return best_k, scores


def _numeric_profile_columns(df: pd.DataFrame) -> list[str]:
    """Select numeric columns suitable for per-cluster profiling."""
    excluded = {"cluster", "cluster_dbscan", "cluster_gmm", "cluster_kmeans", "cluster_pooled"}
    numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns if col not in excluded]
    return numeric_cols


def build_cluster_profile_table(
    df: pd.DataFrame,
    cluster_col: str,
    method_name: str,
) -> pd.DataFrame:
    """Build per-city per-cluster profile table with means across numeric features."""
    if cluster_col not in df.columns:
        raise KeyError(f"Cluster column not found: {cluster_col}")
    if "city" not in df.columns:
        raise KeyError("Input dataframe must include 'city' for profile tables.")

    work = df.copy()
    work = work.dropna(subset=[cluster_col])
    if work.empty:
        return pd.DataFrame()

    numeric_cols = _numeric_profile_columns(work)
    grouped = (
        work.groupby(["city", cluster_col], dropna=False)[numeric_cols]
        .mean(numeric_only=True)
        .reset_index()
        .rename(columns={cluster_col: "cluster"})
        .sort_values(["city", "cluster"]) 
        .reset_index(drop=True)
    )
    grouped.insert(0, "method", method_name)
    counts = (
        work.groupby(["city", cluster_col], dropna=False)
        .size()
        .reset_index(name="listing_count")
        .rename(columns={cluster_col: "cluster"})
    )
    return grouped.merge(counts, on=["city", "cluster"], how="left")


def create_host_level_features(
    listings_df: pd.DataFrame,
    city_name: str | None = None,
) -> pd.DataFrame:
    """Aggregate listing-level rows into host-level strategy features.

    Output columns include:
    - host_id
    - city
    - avg_price
    - availability_365
    - review_density
    - number_of_listings_per_host
    - avg_reviews
    """
    if listings_df.empty:
        raise ValueError("Input listings_df is empty.")

    work = listings_df.copy()
    if "city" not in work.columns:
        work["city"] = "unknown"
    work["city"] = work["city"].astype(str).str.lower()

    if city_name is not None:
        city_key = normalize_city_name(city_name)
        work = work[work["city"] == city_key].copy()

    if work.empty:
        raise ValueError("No rows available after city filtering for host feature creation.")

    host_col = "host_id" if "host_id" in work.columns else "listing_id"
    if host_col not in work.columns:
        raise KeyError("Expected 'host_id' or fallback 'listing_id' in listings_df.")

    work["host_id"] = work[host_col].astype(str)
    work["price"] = pd.to_numeric(work.get("price"), errors="coerce")
    work["availability_365"] = pd.to_numeric(work.get("availability_365"), errors="coerce")

    if "review_density" in work.columns:
        work["review_density"] = pd.to_numeric(work["review_density"], errors="coerce")
    elif "reviews_per_month" in work.columns:
        work["review_density"] = pd.to_numeric(work["reviews_per_month"], errors="coerce")
    elif "number_of_reviews" in work.columns:
        work["review_density"] = pd.to_numeric(work["number_of_reviews"], errors="coerce")
    else:
        work["review_density"] = np.nan

    if "number_of_reviews" in work.columns:
        work["number_of_reviews"] = pd.to_numeric(work["number_of_reviews"], errors="coerce")
    else:
        work["number_of_reviews"] = np.nan

    listing_count_source = "listing_id" if "listing_id" in work.columns else "host_id"

    host_df = (
        work.groupby(["city", "host_id"], as_index=False)
        .agg(
            avg_price=("price", "mean"),
            availability_365=("availability_365", "mean"),
            review_density=("review_density", "mean"),
            number_of_listings_per_host=(listing_count_source, "nunique"),
            avg_reviews=("number_of_reviews", "mean"),
        )
        .sort_values(["city", "host_id"])
        .reset_index(drop=True)
    )

    for col in ["avg_price", "availability_365", "review_density", "number_of_listings_per_host"]:
        host_df[col] = pd.to_numeric(host_df[col], errors="coerce")

    host_df = host_df.dropna(
        subset=["avg_price", "availability_365", "review_density", "number_of_listings_per_host"]
    )
    if host_df.empty:
        raise ValueError("No valid host rows after cleaning required strategy features.")
    return host_df


def cluster_host_strategies(
    host_features_df: pd.DataFrame,
    k: int | None = None,
    k_range: tuple[int, int] = (2, 7),
    random_state: int = 42,
    silhouette_sample: int = 5_000,
) -> dict[str, Any]:
    """Cluster hosts into strategy groups using KMeans.

    Parameters
    ----------
    host_features_df
        Per-host feature frame; must contain avg_price, availability_365,
        review_density, number_of_listings_per_host.
    k
        If given, fit exactly this k. If None, scan `k_range` and pick
        the silhouette-maximising k.
    k_range
        Inclusive range (k_min, k_max) to scan when k is None.
    random_state
        KMeans seed.
    silhouette_sample
        Subsample size for silhouette_score (for speed on large data).

    Returns
    -------
    dict with keys: host_labels, summary_stats, model, scaler,
    feature_columns, chosen_k, silhouette_scan (list of (k, score)).
    """
    required = [
        "avg_price",
        "availability_365",
        "review_density",
        "number_of_listings_per_host",
    ]
    missing = [col for col in required if col not in host_features_df.columns]
    if missing:
        raise KeyError(f"Missing required host features: {missing}")

    work = host_features_df.copy()
    for col in required:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=required).copy()
    k_min = max(2, k_range[0])
    k_max = k_range[1]
    if len(work) <= k_max:
        raise ValueError(f"Need more than {k_max} host rows; got {len(work)}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(work[required])

    silhouette_scan: list[tuple[int, float]] = []
    chosen_k = k
    if chosen_k is None:
        best_k, best_score = None, -1.0
        for ki in range(k_min, k_max + 1):
            km = KMeans(n_clusters=ki, random_state=random_state, n_init=10)
            labels = km.fit_predict(X_scaled)
            score = float(
                silhouette_score(
                    X_scaled,
                    labels,
                    sample_size=min(silhouette_sample, len(X_scaled)),
                    random_state=random_state,
                )
            )
            silhouette_scan.append((ki, score))
            if score > best_score:
                best_score = score
                best_k = ki
        chosen_k = int(best_k)

    model = KMeans(n_clusters=chosen_k, random_state=random_state, n_init=10)
    work["host_cluster"] = model.fit_predict(X_scaled).astype(int)

    summary = (
        work.groupby("host_cluster", as_index=False)
        .agg(
            host_count=("host_id", "nunique"),
            avg_price=("avg_price", "mean"),
            median_price=("avg_price", "median"),
            avg_availability=("availability_365", "mean"),
            avg_review_density=("review_density", "mean"),
            avg_listings_per_host=("number_of_listings_per_host", "mean"),
            avg_reviews=("avg_reviews", "mean"),
        )
        .sort_values("host_count", ascending=False)
        .reset_index(drop=True)
    )
    summary["pct"] = (summary["host_count"] / summary["host_count"].sum() * 100).round(1)

    return {
        "host_labels": work,
        "summary_stats": summary,
        "model": model,
        "scaler": scaler,
        "feature_columns": required,
        "chosen_k": chosen_k,
        "silhouette_scan": silhouette_scan,
    }


def interpret_host_strategy_clusters(cluster_summary_df: pd.DataFrame) -> pd.DataFrame:
    """Assign meaningful strategy names to clusters based on profile patterns."""
    required = ["host_cluster", "avg_price", "avg_availability", "avg_review_density"]
    missing = [col for col in required if col not in cluster_summary_df.columns]
    if missing:
        raise KeyError(f"Missing required summary columns: {missing}")

    profile = cluster_summary_df.copy()
    cols = ["avg_price", "avg_availability", "avg_review_density"]
    z = profile[cols].apply(
        lambda s: (s - s.mean()) / (s.std(ddof=0) if s.std(ddof=0) not in [0, np.nan] else 1.0)
    )

    scores = pd.DataFrame(
        {
            "host_cluster": profile["host_cluster"],
            "occupancy maximizers": (-z["avg_price"] - z["avg_availability"] + z["avg_review_density"]),
            "premium pricers": (z["avg_price"] + z["avg_availability"]),
            "passive hosts": (z["avg_availability"] - z["avg_review_density"]),
        }
    )

    remaining_clusters = set(scores["host_cluster"].tolist())
    assigned: dict[int, str] = {}
    for label_col in ["occupancy maximizers", "premium pricers", "passive hosts"]:
        candidates = scores[scores["host_cluster"].isin(remaining_clusters)].copy()
        if candidates.empty:
            continue
        best_cluster = int(candidates.sort_values(label_col, ascending=False).iloc[0]["host_cluster"])
        assigned[best_cluster] = label_col
        remaining_clusters.discard(best_cluster)

    for cluster in sorted(remaining_clusters):
        assigned[int(cluster)] = f"other_cluster_{int(cluster)}"

    labelled = profile.copy()
    labelled["cluster_name"] = labelled["host_cluster"].map(assigned)

    out_cols = [
        "host_cluster",
        "cluster_name",
        "host_count",
        "avg_price",
        "avg_availability",
        "avg_review_density",
        "avg_reviews",
    ]
    existing_cols = [c for c in out_cols if c in labelled.columns]
    return labelled[existing_cols].sort_values("host_cluster").reset_index(drop=True)


def run_host_strategy_segmentation(
    listings_df: pd.DataFrame,
    city_name: str | None = None,
    output_dir: Path | None = None,
    random_state: int = 42,
    exclude_cities: tuple[str, ...] = ("losangeles",),
    price_cap: float = 10_000.0,
) -> dict[str, Any]:
    """Cluster host strategies and export labels, summaries, and scan diagnostics."""

    tables_dir = output_dir or TABLES_DIR
    tables_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{normalize_city_name(city_name)}" if city_name else ""

    host_features = create_host_level_features(listings_df, city_name=city_name)

    # NEW: filter out artefact cities and artefact price values
    if "city" in host_features.columns and exclude_cities:
        n_before = len(host_features)
        host_features = host_features[~host_features["city"].isin(exclude_cities)]
        n_excluded_city = n_before - len(host_features)
        if n_excluded_city:
            print(f"[host_strategy] excluded {n_excluded_city} rows from {exclude_cities}")

    n_before = len(host_features)
    host_features = host_features[
        (host_features["avg_price"] > 0) & (host_features["avg_price"] < price_cap)
    ]
    n_dropped_price = n_before - len(host_features)
    if n_dropped_price:
        print(f"[host_strategy] dropped {n_dropped_price} rows with price >= ${price_cap:,.0f}")

    # Scan k, pick silhouette optimum
    clustered = cluster_host_strategies(host_features, k=None, random_state=random_state)
    interpreted = interpret_host_strategy_clusters(clustered["summary_stats"])

    labels_path = tables_dir / f"host_strategy_cluster_labels{suffix}.csv"
    summary_path = tables_dir / f"host_strategy_cluster_summary{suffix}.csv"
    scan_path = tables_dir / f"host_strategy_silhouette_scan{suffix}.csv"

    clustered["host_labels"].to_csv(labels_path, index=False)
    interpreted.to_csv(summary_path, index=False)
    pd.DataFrame(
        clustered["silhouette_scan"], columns=["k", "silhouette"]
    ).to_csv(scan_path, index=False)

    print(
        f"[host_strategy] chose k={clustered['chosen_k']} "
        f"(silhouette scan: {clustered['silhouette_scan']})"
    )

    return {
        "host_features": host_features,
        "host_labels": clustered["host_labels"],
        "summary_stats": clustered["summary_stats"],
        "interpreted_summary": interpreted,
        "silhouette_scan": clustered["silhouette_scan"],
        "chosen_k": clustered["chosen_k"],
        "labels_path": labels_path,
        "summary_path": summary_path,
        "scan_path": scan_path,
    }



def run_kmeans_with_pca(
    df: pd.DataFrame,
    city: str,
    k_values: list[int] | None = None,
    random_state: int = 42,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run KMeans clustering with silhouette-based k selection and PCA plot for one city."""
    if k_values is None:
        k_values = [3, 4, 5]

    out_dir = output_dir or PLOTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    model_df = _prepare_cluster_input(df, CLUSTER_FEATURES, city=city)
    if len(model_df) < 50:
        raise ValueError(f"Insufficient rows for KMeans in {city}: {len(model_df)}")

    model_df = _winsorize_features(model_df, CLUSTER_FEATURES)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(model_df[CLUSTER_FEATURES])

    best_k, scores = _select_best_k(X_scaled, k_values)
    model = KMeans(n_clusters=best_k, random_state=random_state, n_init=10)
    model_df["cluster_kmeans"] = model.fit_predict(X_scaled)

    pca = PCA(n_components=2, random_state=random_state)
    pcs = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(
        {
            "pc1": pcs[:, 0],
            "pc2": pcs[:, 1],
            "cluster": model_df["cluster_kmeans"].astype(str),
        }
    )

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=pca_df, x="pc1", y="pc2", hue="cluster", palette="tab10", alpha=0.65, s=32)
    plt.title(f"{city.title()} KMeans Clusters in PCA Space")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid(alpha=0.25)
    pca_plot_path = out_dir / f"{city}_kmeans_pca_scatter.png"
    plt.tight_layout()
    plt.savefig(pca_plot_path, dpi=160)
    plt.close()

    return {
        "city": city,
        "labels": model_df["cluster_kmeans"],
        "indices": model_df.index,
        "best_k": best_k,
        "silhouette_scores": scores,
        "pca_plot_path": pca_plot_path,
    }


def run_dbscan_spatial_clustering(
    df: pd.DataFrame,
    city: str,
    eps: float = 0.6,
    min_samples: int = 12,
) -> dict[str, Any]:
    """Run DBSCAN using scaled spatial and pricing/amenity features for one city."""
    model_df = _prepare_cluster_input(df, DBSCAN_FEATURES, city=city)
    if len(model_df) < 50:
        raise ValueError(f"Insufficient rows for DBSCAN in {city}: {len(model_df)}")

    model_df = _winsorize_features(model_df, DBSCAN_FEATURES)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(model_df[DBSCAN_FEATURES])

    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X_scaled)

    noise_count = int((labels == -1).sum())
    noise_pct = float(noise_count / len(labels) * 100)
    print(
        f"[clustering] {city.upper()} DBSCAN: {noise_count} noise points "
        f"({noise_pct:.1f}% of listings)"
    )
    if noise_pct > 20.0:
        print(
            f"[clustering] WARNING: {city.upper()} DBSCAN noise rate {noise_pct:.1f}% exceeds 20%. "
            f"Consider reducing eps (current: {eps}) or min_samples (current: {min_samples})."
        )

    return {
        "city": city,
        "labels": pd.Series(labels, index=model_df.index),
        "indices": model_df.index,
        "noise_count": noise_count,
        "noise_pct": noise_pct,
    }


def run_gmm_clustering(
    df: pd.DataFrame,
    city: str,
    n_components_range: range | None = None,
    random_state: int = 42,
) -> dict[str, Any]:
    """Run GMM clustering for one city with BIC model selection."""
    if n_components_range is None:
        n_components_range = range(2, 9)

    model_df = _prepare_cluster_input(df, GMM_FEATURES, city=city)
    if len(model_df) < 50:
        raise ValueError(f"Insufficient rows for GMM in {city}: {len(model_df)}")

    model_df = _winsorize_features(model_df, GMM_FEATURES)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(model_df[GMM_FEATURES])

    bic_scores: dict[int, float] = {}
    best_components = None
    best_bic = np.inf
    best_model: GaussianMixture | None = None

    for n_components in n_components_range:
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type="full",
            random_state=random_state,
        )
        gmm.fit(X_scaled)
        bic = gmm.bic(X_scaled)
        bic_scores[n_components] = float(bic)
        if bic < best_bic:
            best_bic = bic
            best_components = n_components
            best_model = gmm

    if best_model is None or best_components is None:
        raise RuntimeError(f"Failed to fit GMM model for city={city}")

    labels = best_model.predict(X_scaled)
    return {
        "city": city,
        "labels": pd.Series(labels, index=model_df.index),
        "indices": model_df.index,
        "best_n_components": int(best_components),
        "bic_scores": bic_scores,
    }


def run_city_clustering_methods(
    city_df: pd.DataFrame,
    city: str,
    output_dir: Path | None = None,
    random_state: int = 42,
) -> dict[str, Any]:
    """Run KMeans, DBSCAN, and GMM clustering methods for one city."""
    work = city_df.copy()
    if "city" not in work.columns:
        work["city"] = city
    work["city"] = work["city"].astype(str).str.lower()

    kmeans_result = run_kmeans_with_pca(
        work,
        city=city,
        output_dir=output_dir,
        random_state=random_state,
    )
    dbscan_result = run_dbscan_spatial_clustering(work, city=city)
    gmm_result = run_gmm_clustering(work, city=city, random_state=random_state)

    labelled = work.copy()
    labelled["cluster_kmeans"] = pd.NA
    labelled["cluster_dbscan"] = pd.NA
    labelled["cluster_gmm"] = pd.NA

    labelled.loc[kmeans_result["indices"], "cluster_kmeans"] = kmeans_result["labels"].astype("int64")
    labelled.loc[dbscan_result["indices"], "cluster_dbscan"] = dbscan_result["labels"].astype("int64")
    labelled.loc[gmm_result["indices"], "cluster_gmm"] = gmm_result["labels"].astype("int64")

    profile_kmeans = build_cluster_profile_table(labelled, "cluster_kmeans", "kmeans")
    profile_dbscan = build_cluster_profile_table(labelled, "cluster_dbscan", "dbscan")
    profile_gmm = build_cluster_profile_table(labelled, "cluster_gmm", "gmm")

    return {
        "city": city,
        "labelled_df": labelled,
        "kmeans": kmeans_result,
        "dbscan": dbscan_result,
        "gmm": gmm_result,
        "profiles": {
            "kmeans": profile_kmeans,
            "dbscan": profile_dbscan,
            "gmm": profile_gmm,
        },
    }


def run_pooled_kmeans_clustering(
    df: pd.DataFrame,
    k: int,
    output_dir: Path | None = None,
    random_state: int = 42,
) -> dict[str, Any]:
    """Run pooled cross-city KMeans and summarize cluster composition by city."""
    if "city" not in df.columns:
        raise KeyError("Pooled clustering requires a 'city' column.")

    out_dir = output_dir or PLOTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    model_df = _prepare_cluster_input(df, CLUSTER_FEATURES)
    model_df = _winsorize_features(model_df, CLUSTER_FEATURES)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(model_df[CLUSTER_FEATURES])

    model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    model_df["cluster_pooled"] = model.fit_predict(X_scaled)

    composition = (
        model_df.groupby(["cluster_pooled", "city"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    total_per_cluster = composition.groupby("cluster_pooled")["count"].transform("sum")
    composition["pct_within_cluster"] = 100 * composition["count"] / total_per_cluster

    pivot = composition.pivot(index="cluster_pooled", columns="city", values="pct_within_cluster").fillna(0)
    city_order = sorted(pivot.columns.tolist())
    pivot = pivot.reindex(columns=city_order)

    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = np.zeros(len(pivot))
    for city in pivot.columns:
        vals = pivot[city].values
        ax.bar(
            pivot.index.astype(str),
            vals,
            bottom=bottom,
            label=city.title(),
            color=CITY_COLORS.get(city, "#777777"),
            alpha=0.9,
        )
        bottom += vals
    ax.set_title("Pooled KMeans Cluster Composition by City")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Percent within Cluster")
    ax.legend()
    composition_plot_path = out_dir / "pooled_kmeans_cluster_composition_by_city.png"
    fig.tight_layout()
    fig.savefig(composition_plot_path, dpi=160)
    plt.close(fig)

    max_city_share = composition.groupby("cluster_pooled")["pct_within_cluster"].max()
    avg_max_share = float(max_city_share.mean()) if not max_city_share.empty else np.nan
    if np.isnan(avg_max_share):
        interpretation = "Insufficient data for pooled cluster city-composition interpretation."
    elif avg_max_share >= 75:
        interpretation = "Clusters appear largely city-specific (dominant city share is high)."
    elif avg_max_share <= 60:
        interpretation = "Clusters appear mostly city-agnostic with mixed city composition."
    else:
        interpretation = "Clusters show partial city separation with moderate overlap across cities."

    pooled_profile = build_cluster_profile_table(model_df, "cluster_pooled", "pooled_kmeans")
    return {
        "labelled_df": model_df,
        "composition_table": composition,
        "composition_plot_path": composition_plot_path,
        "interpretation": interpretation,
        "profile": pooled_profile,
    }


def run_clustering_on_dataframe(
    df: pd.DataFrame,
    output_tables_dir: Path | None = None,
    output_plots_dir: Path | None = None,
    random_state: int = 42,
    city_name: str | None = None,
) -> dict[str, Any]:
    """Run all clustering methods on a pooled dataframe and return method outputs."""
    if "city" not in df.columns:
        raise KeyError("Input dataframe must include a 'city' column.")

    tables_dir = output_tables_dir or TABLES_DIR
    plots_dir = output_plots_dir or PLOTS_DIR
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    work_df = df.copy()
    if city_name is not None:
        city_key = normalize_city_name(city_name)
        work_df = work_df[work_df["city"].astype(str).str.lower() == city_key].copy()

    city_results: dict[str, Any] = {}
    profiles: list[pd.DataFrame] = []
    best_k_values: list[int] = []

    city_keys = sorted(work_df["city"].astype(str).str.lower().unique().tolist())
    for city in city_keys:
        city_df = work_df[work_df["city"].astype(str).str.lower() == city].copy()
        if city_df.empty:
            continue

        result = run_city_clustering_methods(
            city_df,
            city=city,
            output_dir=plots_dir,
            random_state=random_state,
        )
        city_results[city] = result
        best_k_values.append(int(result["kmeans"]["best_k"]))

        for profile_df in result["profiles"].values():
            if not profile_df.empty:
                profiles.append(profile_df)

    pooled_k = int(np.round(np.median(best_k_values))) if best_k_values else 4
    print(
        f"[clustering] Pooled KMeans: using k={pooled_k} "
        f"(median of per-city best-k values: {best_k_values})"
    )
    pooled_result = run_pooled_kmeans_clustering(
        work_df,
        k=pooled_k,
        output_dir=plots_dir,
        random_state=random_state,
    )
    if not pooled_result["profile"].empty:
        profiles.append(pooled_result["profile"])

    profile_table = pd.concat(profiles, ignore_index=True) if profiles else pd.DataFrame()
    profile_path = tables_dir / "cluster_summary.csv"
    profile_table.to_csv(profile_path, index=False)

    composition_path = tables_dir / "pooled_cluster_composition.csv"
    pooled_result["composition_table"].to_csv(composition_path, index=False)

    return {
        "city_results": city_results,
        "pooled_result": pooled_result,
        "cluster_summary": profile_table,
        "cluster_summary_path": profile_path,
        "pooled_composition_path": composition_path,
    }


def run_clustering(city_name: str | None = None) -> tuple[list[Path], Path]:
    """Execute enhanced clustering pipeline for all available featured city files."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    city_files = discover_processed_city_files("featured")
    if city_name is not None:
        city_key = normalize_city_name(city_name)
        city_files = {k: v for k, v in city_files.items() if k == city_key}

    if not city_files:
        raise FileNotFoundError("No featured city files found. Run feature_engineering first.")

    saved_cluster_files: list[Path] = []
    per_method_profiles: list[pd.DataFrame] = []
    best_k_values: list[int] = []
    all_city_frames: list[pd.DataFrame] = []

    for city_key, in_path in city_files.items():

        city_df = pd.read_csv(in_path)
        if "city" not in city_df.columns:
            city_df["city"] = city_key
        all_city_frames.append(city_df)

        result = run_city_clustering_methods(
            city_df,
            city=city_key,
            output_dir=PLOTS_DIR,
        )

        clustered_df = result["labelled_df"].copy()
        clustered_df["cluster"] = clustered_df["cluster_kmeans"]

        out_path = PROCESSED_DIR / f"{city_key}_clustered.csv"
        clustered_df.to_csv(out_path, index=False)
        saved_cluster_files.append(out_path)
        best_k = int(result["kmeans"]["best_k"])
        scores = result["kmeans"]["silhouette_scores"]
        best_k_values.append(best_k)

        for profile_df in result["profiles"].values():
            if not profile_df.empty:
                per_method_profiles.append(profile_df)

        print(
            f"[clustering] {city_key.upper()}: best k = {best_k} | "
            f"scores = {', '.join([f'k={k}:{scores[k]:.3f}' for k in sorted(scores)])}"
        )

        gmm_bic = result["gmm"]["bic_scores"]
        print(
            f"[clustering] {city_key.upper()} GMM best n_components = {result['gmm']['best_n_components']} | "
            f"BIC = {', '.join([f'n={n}:{gmm_bic[n]:.1f}' for n in sorted(gmm_bic)])}"
        )

        dbscan_result = result["dbscan"]
        dbscan_labels = dbscan_result["labels"]
        n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in set(dbscan_labels) else 0)
        print(
            f"[clustering] {city_key.upper()} DBSCAN clusters (excluding noise): {n_clusters_dbscan}"
        )
        print(
            f"[clustering] {city_key.upper()} DBSCAN diagnostics: "
            f"noise_count={dbscan_result['noise_count']}, noise_pct={dbscan_result['noise_pct']:.1f}%"
        )

    cluster_summary = pd.concat(per_method_profiles, ignore_index=True) if per_method_profiles else pd.DataFrame()

    pooled_df = pd.concat(all_city_frames, ignore_index=True)
    pooled_k = int(np.round(np.median(best_k_values))) if best_k_values else 4
    print(
        f"[clustering] Pooled KMeans: using k={pooled_k} "
        f"(median of per-city best-k values: {best_k_values})"
    )
    pooled_result = run_pooled_kmeans_clustering(pooled_df, k=pooled_k, output_dir=PLOTS_DIR)
    if not pooled_result["profile"].empty:
        cluster_summary = pd.concat([cluster_summary, pooled_result["profile"]], ignore_index=True)

    summary_path = TABLES_DIR / "cluster_summary.csv"
    cluster_summary.to_csv(summary_path, index=False)

    pooled_composition_path = TABLES_DIR / "pooled_cluster_composition.csv"
    pooled_result["composition_table"].to_csv(pooled_composition_path, index=False)
    print(f"[clustering] Pooled KMeans interpretation: {pooled_result['interpretation']}")

    return saved_cluster_files, summary_path


def main() -> None:
    cluster_files, summary_file = run_clustering()
    print("\nClustered datasets saved:")
    for path in cluster_files:
        print(path)
    print(f"Cluster summary table saved: {summary_file}")


if __name__ == "__main__":
    main()
