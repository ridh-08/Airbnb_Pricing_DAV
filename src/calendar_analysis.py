from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PLOTS_DIR = BASE_DIR / "outputs" / "plots"
TABLES_DIR = BASE_DIR / "outputs" / "tables"

CALENDAR_FILES = {
    "newyork": PROCESSED_DIR / "newyork_calendar_sample_clean.csv",
    "chicago": PROCESSED_DIR / "chicago_calendar_sample_clean.csv",
}


def _load_calendar_data() -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for city, path in CALENDAR_FILES.items():
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


def plot_selected_listing_price_trends(calendar_df: pd.DataFrame) -> Path:
    """Plot time-wise price trends for selected listings in each city."""
    selected_parts: list[pd.DataFrame] = []
    for city in sorted(calendar_df["city"].unique().tolist()):
        city_df = calendar_df[calendar_df["city"] == city]
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

    out_path = PLOTS_DIR / "calendar_price_variation_selected_listings.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_city_level_availability_trend(calendar_df: pd.DataFrame) -> Path:
    """Plot monthly availability rate trend by city."""
    trend = calendar_df.copy()
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

    out_path = PLOTS_DIR / "calendar_availability_trend.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def export_calendar_summary(calendar_df: pd.DataFrame) -> Path:
    """Export compact calendar trend summary for reporting."""
    summary = (
        calendar_df.groupby("city", as_index=False)
        .agg(
            rows=("listing_id", "size"),
            unique_listings=("listing_id", "nunique"),
            date_min=("date", "min"),
            date_max=("date", "max"),
            avg_price=("price", "mean"),
            median_price=("price", "median"),
            availability_rate=("available", "mean"),
        )
        .sort_values("city")
    )

    summary["availability_rate"] = summary["availability_rate"].round(4)
    summary["avg_price"] = summary["avg_price"].round(2)
    summary["median_price"] = summary["median_price"].round(2)

    out_path = TABLES_DIR / "calendar_summary.csv"
    summary.to_csv(out_path, index=False)
    return out_path


def run_calendar_analysis() -> tuple[list[Path], Path]:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    calendar_df = _load_calendar_data()

    price_plot = plot_selected_listing_price_trends(calendar_df)
    availability_plot = plot_city_level_availability_trend(calendar_df)
    summary_path = export_calendar_summary(calendar_df)

    return [price_plot, availability_plot], summary_path


def main() -> None:
    plots, summary = run_calendar_analysis()
    print("Calendar analysis plots saved:")
    for path in plots:
        print(path)
    print(f"Calendar summary table saved: {summary}")


if __name__ == "__main__":
    main()
