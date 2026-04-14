from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import CITY_COLORS, PROCESSED_DIR, discover_processed_city_files, normalize_city_name


BASE_DIR = Path(__file__).resolve().parent.parent
PLOTS_DIR = BASE_DIR / "outputs" / "plots"


def _count_amenities(value: object) -> int:
    """Count amenities from InsideAirbnb-style string representations."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0

    text = str(value).strip()
    if text in {"", "{}", "[]", "nan"}:
        return 0

    # Amenity strings are typically comma-separated inside braces/brackets.
    inner = text.strip("{}[]")
    if not inner:
        return 0

    return sum(1 for part in inner.split(",") if part.strip())


def _min_max_normalize(series: pd.Series) -> pd.Series:
    """Return min-max normalized values in [0, 1], handling constant series."""
    numeric = pd.to_numeric(series, errors="coerce")
    min_val = numeric.min()
    max_val = numeric.max()
    if pd.isna(min_val) or pd.isna(max_val) or min_val == max_val:
        return pd.Series(0.0, index=series.index)
    return (numeric - min_val) / (max_val - min_val)


def add_demand_score(df: pd.DataFrame) -> pd.DataFrame:
    """Add composite demand_score based on normalized reviews and availability."""
    featured = df.copy()

    if "reviews_per_month" not in featured.columns:
        reviews_total = pd.to_numeric(featured.get("reviews_total"), errors="coerce")
        reviews_last_365d = pd.to_numeric(featured.get("reviews_last_365d"), errors="coerce")

        if reviews_last_365d.notna().any():
            featured["reviews_per_month"] = reviews_last_365d.fillna(0) / 12.0
        elif reviews_total.notna().any():
            featured["reviews_per_month"] = reviews_total.fillna(0) / 12.0
        else:
            featured["reviews_per_month"] = 0.0

    required = ["reviews_per_month", "number_of_reviews", "availability_365"]
    missing = [col for col in required if col not in featured.columns]
    if missing:
        raise KeyError(f"Missing columns required for demand_score: {missing}")

    reviews_pm_norm = _min_max_normalize(featured["reviews_per_month"]).fillna(0.0)
    num_reviews_norm = _min_max_normalize(featured["number_of_reviews"]).fillna(0.0)
    availability_norm = _min_max_normalize(featured["availability_365"]).fillna(0.0)

    featured["demand_score"] = (
        0.4 * reviews_pm_norm
        + 0.3 * num_reviews_norm
        + 0.3 * (1.0 - availability_norm)
    )
    return featured


def plot_demand_score_vs_price(
    df: pd.DataFrame,
    output_dir: Path | None = None,
    cluster_col: str = "cluster_kmeans",
    city_name: str | None = None,
) -> Path:
    """Plot demand_score vs price faceted by city and colored by cluster label."""
    if cluster_col not in df.columns:
        raise KeyError(
            f"Column '{cluster_col}' not found. "
            "Call plot_demand_score_vs_price() AFTER clustering has assigned labels. "
            "Pass cluster_col='cluster_kmeans' when calling from notebook 04."
        )

    required = ["city", "price", "demand_score", cluster_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Missing columns for demand_score plot: {missing}")

    plot_df = df.copy()
    if city_name is not None:
        city_key = normalize_city_name(city_name)
        plot_df = plot_df[plot_df["city"].astype(str).str.lower() == city_key].copy()

    plot_df["price"] = pd.to_numeric(plot_df["price"], errors="coerce")
    plot_df["demand_score"] = pd.to_numeric(plot_df["demand_score"], errors="coerce")
    plot_df = plot_df.dropna(subset=["price", "demand_score", cluster_col, "city"])
    plot_df = plot_df[plot_df["price"] > 0].copy()
    if plot_df.empty:
        raise ValueError("No valid rows found for demand_score vs price plotting.")

    out_dir = output_dir or PLOTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")
    g = sns.relplot(
        data=plot_df,
        x="demand_score",
        y="price",
        hue=cluster_col,
        col="city",
        kind="scatter",
        alpha=0.55,
        s=30,
        height=5,
        aspect=1.1,
        facet_kws={"sharey": False, "sharex": True},
        palette="tab10",
    )

    for ax in g.axes.flat:
        if ax is None:
            continue
        title = ax.get_title()
        city = "newyork" if "newyork" in title.lower() else "chicago"
        ax.set_facecolor("#f7f7f7")
        ax.grid(alpha=0.25)
        ax.set_title(title, color=CITY_COLORS.get(city, "#222222"), fontweight="bold")

    g.fig.suptitle("Demand Score vs Price by City", y=1.02)
    out_path = out_dir / "demand_score_vs_price_by_cluster_city_facet.png"
    g.fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(g.fig)
    return out_path


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add Step 3 engineered features used for modeling and segmentation."""
    featured = df.copy()

    featured["price"] = pd.to_numeric(featured["price"], errors="coerce")
    featured["accommodates"] = pd.to_numeric(featured["accommodates"], errors="coerce")
    featured["number_of_reviews"] = pd.to_numeric(
        featured["number_of_reviews"], errors="coerce"
    )
    featured["availability_365"] = pd.to_numeric(
        featured["availability_365"], errors="coerce"
    )

    safe_price = featured["price"].where(featured["price"] > 0)
    featured["log_price"] = np.log(safe_price)

    safe_accommodates = featured["accommodates"].where(featured["accommodates"] > 0)
    featured["price_per_person"] = featured["price"] / safe_accommodates

    featured["amenities_count"] = featured["amenities"].apply(_count_amenities)

    safe_availability = featured["availability_365"].where(featured["availability_365"] > 0)
    featured["review_density"] = featured["number_of_reviews"] / safe_availability
    featured["review_density"] = featured["review_density"].replace(
        [math.inf, -math.inf], np.nan
    )

    featured = add_demand_score(featured)

    return featured


def run_feature_engineering(city_name: str | None = None) -> list[Path]:
    """Build feature-enriched datasets from analysis-ready city files."""
    city_files = discover_processed_city_files("analysis_ready")
    if city_name is not None:
        city_key = normalize_city_name(city_name)
        city_files = {k: v for k, v in city_files.items() if k == city_key}

    if not city_files:
        raise FileNotFoundError(
            "No analysis-ready files found. Run preprocessing first."
        )

    saved_paths: list[Path] = []
    for city_key, path in city_files.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Missing input file for {city_key}: {path}. Run preprocessing first."
            )

        city_df = pd.read_csv(path)
        featured_df = add_engineered_features(city_df)

        out_path = PROCESSED_DIR / f"{city_key}_featured.csv"
        featured_df.to_csv(out_path, index=False)
        saved_paths.append(out_path)

    return saved_paths


def main() -> None:
    output_files = run_feature_engineering()
    print("Feature engineering completed. Files saved:")
    for path in output_files:
        print(path)


if __name__ == "__main__":
    main()
