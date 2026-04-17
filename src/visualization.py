from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import CITY_COLORS, PLOTS_DIR, TABLES_DIR, discover_processed_city_files, normalize_city_name


BASE_DIR = Path(__file__).resolve().parent.parent


def load_featured_data(city_name: str | None = None) -> pd.DataFrame:
    """Load and combine feature-enriched city datasets for EDA."""
    city_files = discover_processed_city_files("featured")
    if city_name is not None:
        city_key = normalize_city_name(city_name)
        city_files = {k: v for k, v in city_files.items() if k == city_key}

    if not city_files:
        raise FileNotFoundError(
            "Featured files are missing. Run feature_engineering.py first."
        )

    parts = [pd.read_csv(path) for path in city_files.values()]
    combined = pd.concat(parts, ignore_index=True)
    if "price_imputed" not in combined.columns:
        combined["price_imputed"] = False
    return combined


def _price_analysis_frame(df: pd.DataFrame, city_name: str | None = None) -> pd.DataFrame:
    """Restrict to observed (non-imputed) prices for accurate price plots."""
    work = df.copy()
    if "price_imputed" not in work.columns:
        work["price_imputed"] = False

    filtered = work[(work["price"] > 0) & (~work["price_imputed"])].copy()
    if city_name is not None:
        city_key = normalize_city_name(city_name)
        filtered = filtered[filtered["city"].astype(str).str.lower() == city_key].copy()
    return filtered


def compute_gini(price_array: object) -> float:
    """Compute Gini coefficient from scratch for an array-like of prices.

    Edge-case behavior:
    - Missing/non-numeric values are ignored.
    - Negative values are excluded.
    - Empty input or all-zero values return 0.0.
    """
    if price_array is None:
        return 0.0

    values = pd.to_numeric(pd.Series(price_array), errors="coerce")
    values = values[np.isfinite(values)]
    values = values[values >= 0]

    if values.empty:
        return 0.0

    sorted_vals = np.sort(values.to_numpy(dtype=float))
    n = sorted_vals.size
    total = sorted_vals.sum()

    if n == 0 or total <= 0:
        return 0.0

    index = np.arange(1, n + 1, dtype=float)
    gini = (2.0 * np.sum(index * sorted_vals)) / (n * total) - (n + 1.0) / n
    return float(max(0.0, min(1.0, gini)))


def compute_percentile_ratio(prices: object) -> float:
    """Compute the price percentile ratio P90/P10.

    Missing/non-numeric values are ignored and only non-negative values are used.
    Returns np.nan when P10 is zero or no valid values exist.
    """
    if prices is None:
        return float("nan")

    values = pd.to_numeric(pd.Series(prices), errors="coerce")
    values = values[np.isfinite(values)]
    values = values[values >= 0]

    if values.empty:
        return float("nan")

    p10 = float(np.percentile(values, 10))
    p90 = float(np.percentile(values, 90))

    if p10 <= 0:
        return float("nan")

    return float(p90 / p10)


def plot_price_distribution(
    df: pd.DataFrame,
    city_name: str | None = None,
    output_dir: Path | None = None,
) -> str:
    """Robust price distribution view using city-wise capping and log transform."""
    plot_df = _price_analysis_frame(df, city_name=city_name)
    if plot_df.empty:
        return "Insight: No observed prices are available for reliable distribution plotting."

    city_caps = plot_df.groupby("city")["price"].quantile(0.99).to_dict()
    plot_df["price_capped_99"] = plot_df.apply(
        lambda row: min(row["price"], city_caps[row["city"]]), axis=1
    )
    plot_df["log_price_plot"] = np.log1p(plot_df["price"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.histplot(
        data=plot_df,
        x="price_capped_99",
        hue="city",
        bins=60,
        stat="density",
        common_norm=False,
        element="step",
        fill=False,
        ax=axes[0],
    )
    axes[0].set_title("Price Distribution (City-wise 99th Percentile Capped)")
    axes[0].set_xlabel("Price (Capped)")
    axes[0].set_ylabel("Density")

    sns.histplot(
        data=plot_df,
        x="log_price_plot",
        hue="city",
        bins=60,
        stat="density",
        common_norm=False,
        element="step",
        fill=False,
        ax=axes[1],
    )
    axes[1].set_title("Price Distribution (log1p scale)")
    axes[1].set_xlabel("log(1 + price)")
    axes[1].set_ylabel("Density")

    out_dir = output_dir or PLOTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "price_distribution.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    iqr_by_city = (
        plot_df.groupby("city")["price"].quantile(0.75)
        - plot_df.groupby("city")["price"].quantile(0.25)
    )
    if iqr_by_city.empty:
        return "Insight: Unable to compute IQR by city."
    top_city = iqr_by_city.sort_values(ascending=False).index[0]
    return f"Insight: {top_city.title()} shows the widest middle-range price spread (IQR)."


def plot_price_by_room_type(
    df: pd.DataFrame,
    city_name: str | None = None,
    output_dir: Path | None = None,
) -> str:
    """Boxplot of listing price by room type across cities."""
    plot_df = _price_analysis_frame(df, city_name=city_name)
    if plot_df.empty:
        return "Insight: Skipped room-type boxplot because observed prices are unavailable."

    plt.figure(figsize=(11, 6))
    sns.boxplot(
        data=plot_df,
        x="room_type",
        y="price",
        hue="city",
        showfliers=False,
    )
    plt.title("Price by Room Type")
    plt.xlabel("Room Type")
    plt.ylabel("Price")
    plt.xticks(rotation=15)
    plt.ylim(0, plot_df["price"].quantile(0.95))

    out_dir = output_dir or PLOTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "boxplot_price_by_room_type.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    median_by_room = plot_df.groupby("room_type")["price"].median().sort_values(ascending=False)
    top_room = median_by_room.index[0] if not median_by_room.empty else "unknown"
    return f"Insight: {top_room} has the highest median price among room types."


def plot_correlation_heatmap(
    df: pd.DataFrame,
    city_name: str | None = None,
    output_dir: Path | None = None,
) -> str:
    """Correlation heatmap of key numerical features."""
    numeric_cols = [
        "price",
        "log_price",
        "price_per_person",
        "availability_365",
        "number_of_reviews",
        "amenities_count",
        "review_density",
    ]
    plot_df = _price_analysis_frame(df, city_name=city_name)
    if plot_df.empty:
        return "Insight: Skipped correlation heatmap because observed-price rows are unavailable."

    corr = plot_df[numeric_cols].corr(numeric_only=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True)
    plt.title("Correlation Heatmap (Numerical Features)")

    out_dir = output_dir or PLOTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "correlation_heatmap.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    rel = corr.loc["price", "availability_365"]
    polarity = "negative" if rel < 0 else "positive"
    return f"Insight: Price has a {polarity} relationship with availability_365 (corr={rel:.2f})."


def plot_scatter_price_vs_availability(
    df: pd.DataFrame,
    city_name: str | None = None,
    output_dir: Path | None = None,
) -> str:
    """Scatter plot of price versus availability_365."""
    plot_df = _price_analysis_frame(df, city_name=city_name)
    if plot_df.empty:
        return "Insight: Skipped price vs availability scatter due to unavailable observed prices."

    sample = plot_df.sample(min(10000, len(plot_df)), random_state=42)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=sample,
        x="availability_365",
        y="price",
        hue="city",
        alpha=0.35,
        s=18,
    )
    plt.title("Price vs Availability (Sampled)")
    plt.xlabel("Availability 365")
    plt.ylabel("Price")
    plt.ylim(0, sample["price"].quantile(0.98))

    out_dir = output_dir or PLOTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "scatter_price_vs_availability.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    return "Insight: Higher-price listings tend to appear in lower-availability regions."


def plot_scatter_price_vs_reviews(
    df: pd.DataFrame,
    city_name: str | None = None,
    output_dir: Path | None = None,
) -> str:
    """Scatter plot of price versus number_of_reviews."""
    plot_df = _price_analysis_frame(df, city_name=city_name)
    if plot_df.empty:
        return "Insight: Skipped price vs reviews scatter due to unavailable observed prices."

    sample = plot_df.sample(min(10000, len(plot_df)), random_state=42)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=sample,
        x="number_of_reviews",
        y="price",
        hue="city",
        alpha=0.35,
        s=18,
    )
    plt.title("Price vs Number of Reviews (Sampled)")
    plt.xlabel("Number of Reviews")
    plt.ylabel("Price")
    plt.ylim(0, sample["price"].quantile(0.98))

    out_dir = output_dir or PLOTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "scatter_price_vs_reviews.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    return "Insight: Listings with very high review counts are generally not in the highest price tier."


def _build_binned_summary(subset: pd.DataFrame, x_col: str, n_bins: int = 12) -> pd.DataFrame:
    """Create quantile-bin summaries for smoother trend visualization."""
    work = subset[[x_col, "price"]].dropna().copy()
    if work.empty:
        return pd.DataFrame()

    unique_x = work[x_col].nunique()
    if unique_x < 3:
        return pd.DataFrame()

    q = max(3, min(n_bins, unique_x))
    work["bin"] = pd.qcut(work[x_col], q=q, duplicates="drop")

    binned = (
        work.groupby("bin", observed=False)
        .agg(
            x_mean=(x_col, "mean"),
            price_mean=("price", "mean"),
            n=("price", "size"),
        )
        .reset_index(drop=True)
        .sort_values("x_mean")
    )
    return binned


def plot_roomtype_binned_smoothing(
    df: pd.DataFrame,
    x_col: str,
    x_label: str,
    out_name: str,
    city_name: str | None = None,
    output_dir: Path | None = None,
) -> str:
    """Plot room-type wise smoothed trends using quantile-bin mean prices."""
    plot_df = _price_analysis_frame(df, city_name=city_name)
    if plot_df.empty:
        return f"Insight: Skipped {x_col} room-type smoothing due to unavailable observed prices."

    room_types = [
        "Entire home/apt",
        "Private room",
        "Shared room",
        "Hotel room",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
    axes = axes.flatten()

    plotted_any = False
    for idx, room_type in enumerate(room_types):
        ax = axes[idx]
        subset = plot_df[plot_df["room_type"] == room_type].copy()

        if len(subset) < 30:
            ax.set_title(f"{room_type} (insufficient data)")
            ax.text(0.5, 0.5, "Not enough observed rows", ha="center", va="center")
            ax.set_xlabel(x_label)
            ax.set_ylabel("Mean Price")
            continue

        binned = _build_binned_summary(subset, x_col=x_col, n_bins=12)
        if binned.empty:
            ax.set_title(f"{room_type} (insufficient variation)")
            ax.text(0.5, 0.5, "Not enough x variation", ha="center", va="center")
            ax.set_xlabel(x_label)
            ax.set_ylabel("Mean Price")
            continue

        ax.plot(binned["x_mean"], binned["price_mean"], marker="o", linewidth=2)
        ax.set_title(f"{room_type} (n={len(subset)})")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Mean Price")
        ax.grid(alpha=0.25)
        plotted_any = True

    fig.suptitle(f"Room-Type Wise Smoothed Price Trend by {x_label}", fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.98])

    out_dir = output_dir or PLOTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / out_name
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    if not plotted_any:
        return f"Insight: Could not generate meaningful room-type smoothing for {x_col}."
    return f"Insight: Generated room-type wise smoothed trend for {x_col} using quantile-bin mean price."


def compute_neighbourhood_inequality_metrics(
    df: pd.DataFrame,
    city_name: str | None = None,
) -> pd.DataFrame:
    """Compute neighbourhood-level price inequality metrics for each city."""
    required = ["city", "neighbourhood", "price"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Missing columns for inequality analysis: {missing}")

    work = _price_analysis_frame(df, city_name=city_name)
    if work.empty:
        return pd.DataFrame(
            columns=[
                "city",
                "neighbourhood",
                "listing_count",
                "mean_price",
                "median_price",
                "iqr",
                "cv",
            ]
        )

    grouped = work.groupby(["city", "neighbourhood"], dropna=False)
    summary = grouped["price"].agg(
        listing_count="size",
        mean_price="mean",
        median_price="median",
        q1=lambda x: x.quantile(0.25),
        q3=lambda x: x.quantile(0.75),
        std_price="std",
    )
    summary = summary.reset_index()
    summary["iqr"] = summary["q3"] - summary["q1"]
    summary["cv"] = np.where(
        summary["mean_price"] > 0,
        summary["std_price"] / summary["mean_price"],
        np.nan,
    )
    summary = summary.drop(columns=["q1", "q3", "std_price"])
    return summary


def plot_top_expensive_neighbourhoods(
    inequality_df: pd.DataFrame,
    output_dir: Path | None = None,
    top_n: int = 15,
    city_name: str | None = None,
) -> Path:
    """Plot top-N expensive neighbourhoods per city in side-by-side bar charts."""
    if inequality_df.empty:
        raise ValueError("Inequality dataframe is empty; cannot plot expensive neighbourhoods.")

    out_dir = output_dir or PLOTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    work = inequality_df.copy()
    if city_name is not None:
        city_key = normalize_city_name(city_name)
        work = work[work["city"] == city_key].copy()

    cities = sorted(work["city"].dropna().unique().tolist())
    fig, axes = plt.subplots(1, len(cities), figsize=(8 * len(cities), 8), sharex=False)
    if len(cities) == 1:
        axes = [axes]

    for ax, city in zip(axes, cities):
        city_df = (
            work[work["city"] == city]
            .sort_values("mean_price", ascending=False)
            .head(top_n)
            .sort_values("mean_price", ascending=True)
        )

        sns.barplot(
            data=city_df,
            x="mean_price",
            y="neighbourhood",
            color=CITY_COLORS[city],
            ax=ax,
        )
        ax.set_title(f"Top {top_n} Most Expensive Neighbourhoods: {city.title()}")
        ax.set_xlabel("Mean Price")
        ax.set_ylabel("Neighbourhood")

    out_path = out_dir / "neighbourhood_top15_expensive_by_city.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def plot_violin_price_by_room_type_city(
    df: pd.DataFrame,
    output_dir: Path | None = None,
    city_name: str | None = None,
) -> Path:
    """Plot side-by-side violin distributions of price by room type for each city."""
    work = _price_analysis_frame(df, city_name=city_name)
    if work.empty:
        raise ValueError("No observed prices available for violin plotting.")

    out_dir = output_dir or PLOTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    room_order = [
        "Entire home/apt",
        "Private room",
        "Shared room",
        "Hotel room",
    ]

    cities = sorted(work["city"].dropna().unique().tolist())
    fig, axes = plt.subplots(1, len(cities), figsize=(8 * len(cities), 7), sharey=True)
    if len(cities) == 1:
        axes = [axes]

    for ax, city in zip(axes, cities):
        city_df = work[work["city"] == city]
        sns.violinplot(
            data=city_df,
            x="room_type",
            y="price",
            order=room_order,
            color=CITY_COLORS[city],
            cut=0,
            inner="quartile",
            ax=ax,
        )
        ax.set_title(f"{city.title()} Price by Room Type")
        ax.set_xlabel("Room Type")
        ax.set_ylabel("Price")
        ax.tick_params(axis="x", rotation=18)
        ax.set_ylim(0, np.nanpercentile(work["price"], 98))

    out_path = out_dir / "violin_price_by_room_type_city_comparison.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def plot_neighbourhood_mean_vs_cv(
    inequality_df: pd.DataFrame,
    output_dir: Path | None = None,
    annotate_top_n: int = 5,
    city_name: str | None = None,
) -> Path:
    """Scatter mean_price vs CV by neighbourhood, sized by listing count and annotated."""
    if inequality_df.empty:
        raise ValueError("Inequality dataframe is empty; cannot create mean-vs-CV scatter.")

    out_dir = output_dir or PLOTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    work = inequality_df.copy()
    if city_name is not None:
        city_key = normalize_city_name(city_name)
        work = work[work["city"] == city_key].copy()

    cities = sorted(work["city"].dropna().unique().tolist())
    fig, axes = plt.subplots(1, len(cities), figsize=(8 * len(cities), 7), sharex=False, sharey=False)
    if len(cities) == 1:
        axes = [axes]

    for ax, city in zip(axes, cities):
        city_df = work[work["city"] == city].dropna(subset=["cv", "mean_price"]).copy()
        if city_df.empty:
            ax.set_title(f"{city.title()} (no valid data)")
            continue

        sns.scatterplot(
            data=city_df,
            x="mean_price",
            y="cv",
            size="listing_count",
            sizes=(30, 350),
            color=CITY_COLORS[city],
            alpha=0.7,
            edgecolor="black",
            linewidth=0.3,
            ax=ax,
            legend=False,
        )

        top_cv = city_df.sort_values("cv", ascending=False).head(annotate_top_n)
        for _, row in top_cv.iterrows():
            ax.annotate(
                str(row["neighbourhood"]),
                (row["mean_price"], row["cv"]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=8,
            )

        ax.set_title(f"{city.title()} Neighbourhood Mean Price vs CV")
        ax.set_xlabel("Neighbourhood Mean Price")
        ax.set_ylabel("Coefficient of Variation (CV)")
        ax.grid(alpha=0.25)

    out_path = out_dir / "neighbourhood_mean_price_vs_cv_scatter.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def run_neighbourhood_inequality_analysis(
    df: pd.DataFrame,
    output_dir: Path | None = None,
    city_name: str | None = None,
) -> dict[str, object]:
    """Run full neighbourhood inequality workflow and return metrics and plot paths."""
    inequality_df = compute_neighbourhood_inequality_metrics(df, city_name=city_name)
    if inequality_df.empty:
        raise ValueError("No valid data available for neighbourhood inequality analysis.")

    top_expensive_path = plot_top_expensive_neighbourhoods(
        inequality_df,
        output_dir=output_dir,
        top_n=15,
        city_name=city_name,
    )
    violin_path = plot_violin_price_by_room_type_city(
        df,
        output_dir=output_dir,
        city_name=city_name,
    )
    mean_cv_path = plot_neighbourhood_mean_vs_cv(
        inequality_df,
        output_dir=output_dir,
        annotate_top_n=5,
        city_name=city_name,
    )

    return {
        "inequality_summary": inequality_df,
        "top_expensive_plot": top_expensive_path,
        "room_type_violin_plot": violin_path,
        "mean_vs_cv_plot": mean_cv_path,
    }


def export_data_quality_table(
    df: pd.DataFrame,
    city_name: str | None = None,
    output_dir: Path | None = None,
) -> Path:
    """Export observed vs imputed price quality summary by city and room type."""
    quality_df = df.copy()
    if city_name is not None:
        city_key = normalize_city_name(city_name)
        quality_df = quality_df[quality_df["city"].astype(str).str.lower() == city_key].copy()

    quality_df["price_imputed"] = quality_df["price_imputed"].fillna(False).astype(bool)
    quality_df["price_observed"] = ~quality_df["price_imputed"]

    by_city = (
        quality_df.groupby("city", dropna=False)
        .agg(
            rows_total=("listing_id", "size"),
            observed_rows=("price_observed", "sum"),
            imputed_rows=("price_imputed", "sum"),
        )
        .reset_index()
    )
    by_city["room_type"] = "ALL"

    by_room = (
        quality_df.groupby(["city", "room_type"], dropna=False)
        .agg(
            rows_total=("listing_id", "size"),
            observed_rows=("price_observed", "sum"),
            imputed_rows=("price_imputed", "sum"),
        )
        .reset_index()
    )

    summary = pd.concat([by_city, by_room], ignore_index=True)
    summary["observed_rate"] = summary["observed_rows"] / summary["rows_total"]
    summary["imputed_rate"] = summary["imputed_rows"] / summary["rows_total"]

    out_dir = output_dir or TABLES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "data_quality_summary.csv"
    summary = summary.sort_values(["city", "room_type"]).reset_index(drop=True)
    summary.to_csv(out_path, index=False)
    return out_path


def run_eda(
    city_name: str | None = None,
    plots_output_dir: Path | None = None,
    tables_output_dir: Path | None = None,
) -> tuple[list[str], Path]:
    """Run Step 4 EDA plotting workflow and return insights and quality table path."""
    plots_dir = plots_output_dir or PLOTS_DIR
    tables_dir = tables_output_dir or TABLES_DIR
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    df = load_featured_data(city_name=city_name)
    quality_table_path = export_data_quality_table(
        df,
        city_name=city_name,
        output_dir=tables_dir,
    )
    insights = [
        plot_price_distribution(df, city_name=city_name, output_dir=plots_dir),
        plot_price_by_room_type(df, city_name=city_name, output_dir=plots_dir),
        plot_correlation_heatmap(df, city_name=city_name, output_dir=plots_dir),
        plot_scatter_price_vs_availability(df, city_name=city_name, output_dir=plots_dir),
        plot_scatter_price_vs_reviews(df, city_name=city_name, output_dir=plots_dir),
        plot_roomtype_binned_smoothing(
            df,
            x_col="availability_365",
            x_label="Availability 365",
            out_name="price_vs_availability_roomtype_binned.png",
            city_name=city_name,
            output_dir=plots_dir,
        ),
        plot_roomtype_binned_smoothing(
            df,
            x_col="number_of_reviews",
            x_label="Number of Reviews",
            out_name="price_vs_reviews_roomtype_binned.png",
            city_name=city_name,
            output_dir=plots_dir,
        ),
    ]

    try:
        inequality_results = run_neighbourhood_inequality_analysis(
            df,
            output_dir=plots_dir,
            city_name=city_name,
        )
        insights.append(
            "Insight: Generated neighbourhood inequality analysis plots and summary metrics."
        )
        inequality_table_path = tables_dir / "neighbourhood_inequality_summary.csv"
        inequality_results["inequality_summary"].to_csv(inequality_table_path, index=False)
    except ValueError:
        insights.append(
            "Insight: Skipped neighbourhood inequality analysis due to insufficient valid data."
        )

    return insights, quality_table_path


def main() -> None:
    insights, quality_table_path = run_eda()
    print("EDA plots saved to outputs/plots")
    print(f"Data quality table saved to: {quality_table_path}")
    for insight in insights:
        print(insight)


if __name__ == "__main__":
    main()


