from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import TABLES_DIR, PLOTS_DIR, discover_processed_city_files, normalize_city_name


BASE_DIR = Path(__file__).resolve().parent.parent

MONTH_ORDER = list(range(1, 13))
MONTH_LABELS = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]
WEEKDAY_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _load_calendar_data(city_name: str | None = None) -> pd.DataFrame:
    calendar_files = discover_processed_city_files("calendar_sample_clean")
    if city_name is not None:
        city_key = normalize_city_name(city_name)
        calendar_files = {k: v for k, v in calendar_files.items() if k == city_key}

    if not calendar_files:
        raise FileNotFoundError("No cleaned calendar samples found. Run preprocessing first.")

    parts: list[pd.DataFrame] = []
    for city, path in calendar_files.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Missing cleaned calendar sample for {city}: {path}. Run preprocessing first."
            )
        df = pd.read_csv(path)
        df["city"] = city
        parts.append(df)

    data = pd.concat(parts, ignore_index=True)
    data["date"] = pd.to_datetime(data["date"], errors="coerce", dayfirst=True)
    data["price"] = pd.to_numeric(data["price"], errors="coerce")
    data["available"] = pd.to_numeric(data["available"], errors="coerce")
    data["listing_id"] = pd.to_numeric(data["listing_id"], errors="coerce")

    data = data.dropna(subset=["date", "price", "available", "listing_id"])
    data["listing_id"] = data["listing_id"].astype("int64")
    data["available"] = data["available"].astype("int64")
    return data


def _pick_representative_listings(city_df: pd.DataFrame, n_listings: int = 5) -> list[int]:
    """Pick listings with deepest coverage and moderate price levels for clean trend lines."""
    coverage = (
        city_df.groupby("listing_id", as_index=False)
        .agg(days=("date", "nunique"), median_price=("price", "median"))
        .sort_values(["days", "median_price"], ascending=[False, True])
    )
    if coverage.empty:
        return []

    # Avoid selecting only extreme-price listings by taking a middle slice of the top coverage group.
    top = coverage.head(200).copy()
    top = top.sort_values("median_price").reset_index(drop=True)
    if len(top) <= n_listings:
        return top["listing_id"].tolist()

    step = max(1, len(top) // n_listings)
    chosen = top.iloc[::step].head(n_listings)["listing_id"].tolist()
    return chosen


def engineer_calendar_temporal_features(
    calendar_df: pd.DataFrame,
    city_name: str | None = None,
) -> pd.DataFrame:
    """Build month x weekday aggregates for price and availability."""
    work = calendar_df.copy()
    if city_name is not None:
        city_key = normalize_city_name(city_name)
        work = work[work["city"].astype(str).str.lower() == city_key].copy()

    if work.empty:
        return pd.DataFrame(
            columns=[
                "city",
                "month_num",
                "month",
                "weekday_num",
                "weekday",
                "avg_price",
                "availability_rate",
            ]
        )

    work["month_num"] = work["date"].dt.month
    work["weekday_num"] = work["date"].dt.weekday
    temporal = (
        work.groupby(["city", "month_num", "weekday_num"], as_index=False)
        .agg(
            avg_price=("price", "mean"),
            availability_rate=("available", "mean"),
        )
        .sort_values(["city", "month_num", "weekday_num"]) 
        .reset_index(drop=True)
    )

    temporal["month"] = pd.Categorical(
        temporal["month_num"].map(dict(zip(MONTH_ORDER, MONTH_LABELS))),
        categories=MONTH_LABELS,
        ordered=True,
    )
    temporal["weekday"] = pd.Categorical(
        temporal["weekday_num"].map(dict(enumerate(WEEKDAY_ORDER))),
        categories=WEEKDAY_ORDER,
        ordered=True,
    )

    temporal = temporal[
        [
            "city",
            "month_num",
            "month",
            "weekday_num",
            "weekday",
            "avg_price",
            "availability_rate",
        ]
    ]
    return temporal


def export_calendar_temporal_summary(
    temporal_df: pd.DataFrame,
    output_dir: Path | None = None,
    city_name: str | None = None,
) -> Path:
    """Export month-weekday calendar aggregates for downstream analysis."""
    out_dir = output_dir or TABLES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{normalize_city_name(city_name)}" if city_name else ""
    out_path = out_dir / f"calendar_temporal_summary{suffix}.csv"
    temporal_df.to_csv(out_path, index=False)
    return out_path


def _plot_temporal_heatmap(
    temporal_df: pd.DataFrame,
    metric_col: str,
    title_prefix: str,
    out_name: str,
    output_dir: Path | None = None,
    city_name: str | None = None,
) -> Path:
    """Plot month x weekday heatmaps with per-city color scales for comparability within each panel."""
    if temporal_df.empty:
        raise ValueError("Temporal dataframe is empty; cannot create heatmap.")

    out_dir = output_dir or PLOTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    work = temporal_df.copy()
    if city_name is not None:
        city_key = normalize_city_name(city_name)
        work = work[work["city"].astype(str).str.lower() == city_key].copy()

    if work.empty:
        raise ValueError("No city rows available after filtering for heatmap.")

    city_keys = sorted(work["city"].astype(str).str.lower().unique().tolist())
    n_cities = len(city_keys)

    if n_cities == 1:
        fig, ax = plt.subplots(1, 1, figsize=(8.5, 6.5))
        axes = [ax]
    else:
        ncols = 3
        nrows = (n_cities + ncols - 1) // ncols
        fig, axes_arr = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.5 * nrows), sharex=False, sharey=False)
        axes = list(axes_arr.ravel())

    for idx, city in enumerate(city_keys):
        ax = axes[idx]
        city_df = work[work["city"].astype(str).str.lower() == city].copy()
        pivot = city_df.pivot(index="month", columns="weekday", values=metric_col)
        pivot = pivot.reindex(index=MONTH_LABELS, columns=WEEKDAY_ORDER)

        city_vmin = float(city_df[metric_col].min())
        city_vmax = float(city_df[metric_col].max())
        flat_city = city_vmin == city_vmax
        if flat_city:
            city_vmax = city_vmin + 1e-9

        sns.heatmap(
            pivot,
            ax=ax,
            cmap="YlOrRd",
            vmin=city_vmin,
            vmax=city_vmax,
            cbar=True,
            cbar_kws={"shrink": 0.78, "pad": 0.02},
            linewidths=0.3,
            linecolor="#f2f2f2",
        )
        ax.set_title(city.title())
        ax.set_xlabel("Weekday")
        ax.set_ylabel("Month")

        if flat_city:
            ax.text(
                0.5,
                0.5,
                "No price variability\nin source data",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=9,
                color="#6f6f6f",
                bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
            )

    for j in range(n_cities, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"{title_prefix}: Month x Weekday", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    suffix = f"_{normalize_city_name(city_name)}" if city_name else ""
    out_path = out_dir / f"{out_name}{suffix}.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def plot_calendar_price_heatmap(
    temporal_df: pd.DataFrame,
    output_dir: Path | None = None,
    city_name: str | None = None,
) -> Path:
    """Plot calendar price heatmap (month x weekday)."""
    return _plot_temporal_heatmap(
        temporal_df,
        metric_col="avg_price",
        title_prefix="Average Price",
        out_name="calendar_price_heatmap_month_weekday",
        output_dir=output_dir,
        city_name=city_name,
    )


def plot_calendar_availability_heatmap(
    temporal_df: pd.DataFrame,
    output_dir: Path | None = None,
    city_name: str | None = None,
) -> Path:
    """Plot calendar availability heatmap (month x weekday)."""
    return _plot_temporal_heatmap(
        temporal_df,
        metric_col="availability_rate",
        title_prefix="Availability Rate",
        out_name="calendar_availability_heatmap_month_weekday",
        output_dir=output_dir,
        city_name=city_name,
    )


def plot_selected_listing_price_trends(
    calendar_df: pd.DataFrame,
    output_dir: Path | None = None,
    city_name: str | None = None,
) -> Path:
    """Plot time-wise price trends for selected listings in each city."""
    work = calendar_df.copy()
    if city_name is not None:
        city_key = normalize_city_name(city_name)
        work = work[work["city"].astype(str).str.lower() == city_key].copy()

    selected_parts: list[pd.DataFrame] = []
    for city in sorted(work["city"].unique().tolist()):
        city_df = work[work["city"] == city]
        chosen_ids = _pick_representative_listings(city_df, n_listings=5)
        if not chosen_ids:
            continue

        subset = city_df[city_df["listing_id"].isin(chosen_ids)].copy()
        subset["series"] = subset["city"] + "_" + subset["listing_id"].astype(str)
        selected_parts.append(subset)

    if not selected_parts:
        raise ValueError("No listings available to plot selected listing trends.")

    plot_df = pd.concat(selected_parts, ignore_index=True)

    plt.figure(figsize=(14, 7))
    sns.lineplot(
        data=plot_df,
        x="date",
        y="price",
        hue="series",
        estimator="median",
        errorbar=None,
        linewidth=1.4,
        alpha=0.9,
    )
    plt.title("Calendar Price Variation Over Time (Selected Listings)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend(title="City_Listing", bbox_to_anchor=(1.02, 1), loc="upper left")

    out_dir = output_dir or PLOTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{normalize_city_name(city_name)}" if city_name else ""
    out_path = out_dir / f"calendar_price_variation_selected_listings{suffix}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_city_level_availability_trend(
    calendar_df: pd.DataFrame,
    output_dir: Path | None = None,
    city_name: str | None = None,
) -> Path:
    """Plot monthly availability rate trend by city."""
    trend = calendar_df.copy()
    if city_name is not None:
        city_key = normalize_city_name(city_name)
        trend = trend[trend["city"].astype(str).str.lower() == city_key].copy()
    trend["month"] = trend["date"].dt.to_period("M").dt.to_timestamp()

    monthly = (
        trend.groupby(["city", "month"], as_index=False)["available"]
        .mean()
        .rename(columns={"available": "availability_rate"})
    )

    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=monthly,
        x="month",
        y="availability_rate",
        hue="city",
        marker="o",
    )
    plt.title("Monthly Availability Trend by City")
    plt.xlabel("Month")
    plt.ylabel("Availability Rate")
    plt.ylim(0, 1)

    out_dir = output_dir or PLOTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{normalize_city_name(city_name)}" if city_name else ""
    out_path = out_dir / f"calendar_availability_trend{suffix}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def export_calendar_summary(
    calendar_df: pd.DataFrame,
    output_dir: Path | None = None,
    city_name: str | None = None,
) -> Path:
    """Export compact calendar trend summary for reporting."""
    work = calendar_df.copy()
    if city_name is not None:
        city_key = normalize_city_name(city_name)
        work = work[work["city"].astype(str).str.lower() == city_key].copy()

    rows: list[dict[str, object]] = []
    for city, city_df in work.groupby("city", dropna=False):
        city_df = city_df.copy()
        price_df = city_df
        if "price_source" in city_df.columns:
            non_global = city_df[city_df["price_source"] != "global_fallback"]
            if not non_global.empty:
                price_df = non_global
            else:
                # If all prices are global fallback values, avoid reporting misleading city medians.
                price_df = city_df.iloc[0:0]

        rows.append(
            {
                "city": city,
                "rows": int(len(city_df)),
                "unique_listings": int(city_df["listing_id"].nunique()),
                "date_min": city_df["date"].min(),
                "date_max": city_df["date"].max(),
                "avg_price": float(price_df["price"].mean()) if not price_df.empty else float("nan"),
                "median_price": float(price_df["price"].median()) if not price_df.empty else float("nan"),
                "availability_rate": float(city_df["available"].mean()) if not city_df.empty else float("nan"),
            }
        )

    summary = pd.DataFrame(rows).sort_values("city").reset_index(drop=True)

    summary["availability_rate"] = summary["availability_rate"].round(4)
    summary["avg_price"] = summary["avg_price"].round(2)
    summary["median_price"] = summary["median_price"].round(2)

    out_dir = output_dir or TABLES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{normalize_city_name(city_name)}" if city_name else ""
    out_path = out_dir / f"calendar_summary{suffix}.csv"
    summary.to_csv(out_path, index=False)
    return out_path


def run_calendar_analysis(
    city_name: str | None = None,
    plots_output_dir: Path | None = None,
    tables_output_dir: Path | None = None,
) -> tuple[list[Path], Path]:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    calendar_df = _load_calendar_data(city_name=city_name)
    temporal_df = engineer_calendar_temporal_features(calendar_df, city_name=city_name)

    price_plot = plot_selected_listing_price_trends(
        calendar_df,
        output_dir=plots_output_dir,
        city_name=city_name,
    )
    availability_plot = plot_city_level_availability_trend(
        calendar_df,
        output_dir=plots_output_dir,
        city_name=city_name,
    )
    summary_path = export_calendar_summary(
        calendar_df,
        output_dir=tables_output_dir,
        city_name=city_name,
    )

    temporal_summary_path = export_calendar_temporal_summary(
        temporal_df,
        output_dir=tables_output_dir,
        city_name=city_name,
    )

    price_heatmap = plot_calendar_price_heatmap(
        temporal_df,
        output_dir=plots_output_dir,
        city_name=city_name,
    )
    availability_heatmap = plot_calendar_availability_heatmap(
        temporal_df,
        output_dir=plots_output_dir,
        city_name=city_name,
    )

    print(f"Calendar temporal summary table saved: {temporal_summary_path}")

    return [price_plot, availability_plot, price_heatmap, availability_heatmap], summary_path


def main() -> None:
    plots, summary = run_calendar_analysis()
    print("Calendar analysis plots saved:")
    for path in plots:
        print(path)
    print(f"Calendar summary table saved: {summary}")


if __name__ == "__main__":
    main()
