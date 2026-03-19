from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
CALENDAR_SAMPLE_ROWS = 200000

CITY_FOLDERS = {
    "newyork": "New York",
    "chicago": "Chicago",
}

RELEVANT_COLUMNS = [
    "listing_id",
    "price",
    "room_type",
    "neighbourhood",
    "latitude",
    "longitude",
    "accommodates",
    "number_of_reviews",
    "availability_365",
    "amenities",
]


def _find_dataset_file(city_dir: Path, dataset_name: str) -> Path:
    """Find dataset files across both standard and nested folder layouts."""
    candidates = [
        city_dir / f"{dataset_name}.csv.gz",
        city_dir / f"{dataset_name}.csv",
        city_dir / f"{dataset_name}.csv" / f"{dataset_name}.csv.gz",
        city_dir / f"{dataset_name}.csv" / f"{dataset_name}.csv",
        city_dir / f"{dataset_name} (1).csv" / f"{dataset_name}.csv.gz",
        city_dir / f"{dataset_name} (1).csv" / f"{dataset_name}.csv",
    ]

    for path in candidates:
        if path.exists() and path.is_file():
            return path

    # Fallback: look for any nested folder variant like "listings (2).csv".
    for directory in city_dir.glob(f"{dataset_name}*.csv"):
        if directory.is_dir():
            for ext in ("*.csv.gz", "*.csv"):
                matches = list(directory.glob(ext))
                if matches:
                    return matches[0]

    raise FileNotFoundError(
        f"Could not find {dataset_name}.csv or {dataset_name}.csv.gz under {city_dir}"
    )


def load_tabular_data(file_path: Path, nrows: int | None = None) -> pd.DataFrame:
    """Load regular or compressed CSV data with automatic compression detection."""
    return pd.read_csv(file_path, compression="infer", nrows=nrows)


def load_geojson(file_path: Path) -> gpd.GeoDataFrame:
    """Load neighbourhood boundaries from geojson."""
    return gpd.read_file(file_path)


def load_city_data(
    city_key: str,
    include_calendar: bool = False,
    calendar_sample_rows: int | None = None,
) -> dict[str, Any]:
    """Load all required raw data for a single city."""
    normalized_city = city_key.lower().strip()
    if normalized_city not in CITY_FOLDERS:
        raise ValueError(
            f"Unknown city '{city_key}'. Valid options: {list(CITY_FOLDERS.keys())}"
        )

    city_dir = BASE_DIR / CITY_FOLDERS[normalized_city]
    listings_path = _find_dataset_file(city_dir, "listings")
    reviews_path = _find_dataset_file(city_dir, "reviews")
    geojson_path = city_dir / "neighbourhoods.geojson"

    city_data: dict[str, Any] = {
        "city": normalized_city,
        "listings": load_tabular_data(listings_path),
        "reviews": load_tabular_data(reviews_path),
        "neighbourhoods": load_geojson(geojson_path),
    }

    if include_calendar:
        calendar_path = _find_dataset_file(city_dir, "calendar")
        city_data["calendar"] = load_tabular_data(
            calendar_path, nrows=calendar_sample_rows
        )

    return city_data


def load_all_cities(
    include_calendar: bool = False,
    calendar_sample_rows: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Load city datasets for New York and Chicago using a common interface."""
    return {
        city_key: load_city_data(
            city_key,
            include_calendar=include_calendar,
            calendar_sample_rows=calendar_sample_rows,
        )
        for city_key in CITY_FOLDERS
    }


def _clean_price_column(series: pd.Series) -> pd.Series:
    """Convert Airbnb price strings like '$1,234.00' into float values."""
    return pd.to_numeric(
        series.astype(str).str.replace(r"[$,]", "", regex=True).str.strip(),
        errors="coerce",
    )


def _standardize_listing_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure consistent schema across cities before selecting analysis columns."""
    cleaned = df.copy()

    if "listing_id" not in cleaned.columns and "id" in cleaned.columns:
        cleaned["listing_id"] = cleaned["id"]

    if "neighbourhood_cleansed" in cleaned.columns:
        if "neighbourhood" not in cleaned.columns:
            cleaned = cleaned.rename(columns={"neighbourhood_cleansed": "neighbourhood"})
        else:
            cleaned["neighbourhood"] = cleaned["neighbourhood_cleansed"].fillna(
                cleaned["neighbourhood"]
            )

    for col in RELEVANT_COLUMNS:
        if col not in cleaned.columns:
            cleaned[col] = pd.NA

    return cleaned[RELEVANT_COLUMNS].copy()


def clean_city_listings(
    listings_df: pd.DataFrame,
    city_key: str,
    room_type_price_fallback: dict[str, float],
    global_price_fallback: float,
) -> pd.DataFrame:
    """Apply Step 2 cleaning rules to a city's listings data."""
    cleaned = _standardize_listing_columns(listings_df)

    cleaned["price"] = _clean_price_column(cleaned["price"])
    cleaned["price_observed"] = cleaned["price"].notna()

    numeric_columns = [
        "listing_id",
        "latitude",
        "longitude",
        "accommodates",
        "number_of_reviews",
        "availability_365",
    ]
    for col in numeric_columns:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    cleaned.loc[cleaned["accommodates"] <= 0, "accommodates"] = pd.NA

    cleaned["number_of_reviews"] = cleaned["number_of_reviews"].fillna(0)
    cleaned["availability_365"] = cleaned["availability_365"].fillna(
        cleaned["availability_365"].median()
    )
    cleaned["amenities"] = cleaned["amenities"].fillna("{}")

    # Fill missing prices using room-type medians, then a global median fallback.
    cleaned["price"] = cleaned["price"].fillna(cleaned["room_type"].map(room_type_price_fallback))
    cleaned["price"] = cleaned["price"].fillna(global_price_fallback)
    cleaned["price_imputed"] = ~cleaned["price_observed"]

    required_fields = [
        "listing_id",
        "room_type",
        "neighbourhood",
        "latitude",
        "longitude",
        "accommodates",
    ]
    cleaned = cleaned.dropna(subset=required_fields)
    cleaned["listing_id"] = cleaned["listing_id"].astype("int64")

    cleaned = cleaned[cleaned["price"] > 0].copy()
    price_cap = cleaned["price"].quantile(0.99)
    cleaned = cleaned[cleaned["price"] <= price_cap].copy()

    cleaned["city"] = city_key

    return cleaned.reset_index(drop=True)


def clean_all_cities(raw_data: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Create a cleaned listings table for each city dataset bundle."""
    cleaned_data = raw_data.copy()

    # Build fallback price statistics from all available non-missing prices.
    all_prices: list[pd.DataFrame] = []
    for city_bundle in cleaned_data.values():
        standardized = _standardize_listing_columns(city_bundle["listings"])
        standardized["price"] = _clean_price_column(standardized["price"])
        all_prices.append(standardized[["room_type", "price"]])

    pooled_prices = pd.concat(all_prices, ignore_index=True)
    pooled_prices = pooled_prices[pooled_prices["price"] > 0]

    room_type_price_fallback = (
        pooled_prices.groupby("room_type", dropna=True)["price"].median().to_dict()
    )
    global_price_fallback = float(pooled_prices["price"].median())

    for city_key, city_bundle in cleaned_data.items():
        city_bundle["listings_clean"] = clean_city_listings(
            city_bundle["listings"],
            city_key=city_key,
            room_type_price_fallback=room_type_price_fallback,
            global_price_fallback=global_price_fallback,
        )

        city_bundle["reviews_clean"] = clean_reviews(city_bundle["reviews"], city_key=city_key)
        city_bundle["reviews_summary"] = build_reviews_summary(city_bundle["reviews_clean"])

        if "calendar" in city_bundle:
            listing_price_fallback = build_listing_price_lookup(city_bundle["listings"])
            city_bundle["calendar_clean"] = clean_calendar(
                city_bundle["calendar"],
                city_key,
                listing_price_fallback=listing_price_fallback,
                global_price_fallback=global_price_fallback,
            )

    return cleaned_data


def clean_reviews(reviews_df: pd.DataFrame, city_key: str) -> pd.DataFrame:
    """Clean reviews data for demand-proxy analysis."""
    cleaned = reviews_df.copy()

    if "listing_id" not in cleaned.columns:
        raise KeyError("Expected 'listing_id' column in reviews dataset.")
    if "date" not in cleaned.columns:
        raise KeyError("Expected 'date' column in reviews dataset.")

    cleaned["listing_id"] = pd.to_numeric(cleaned["listing_id"], errors="coerce")
    cleaned["date"] = pd.to_datetime(cleaned["date"], errors="coerce")
    cleaned = cleaned.dropna(subset=["listing_id", "date"])
    cleaned["listing_id"] = cleaned["listing_id"].astype("int64")

    duplicate_subset = [col for col in ["id", "listing_id", "date"] if col in cleaned.columns]
    if duplicate_subset:
        cleaned = cleaned.drop_duplicates(subset=duplicate_subset)

    cleaned["city"] = city_key
    return cleaned.reset_index(drop=True)


def build_reviews_summary(reviews_clean_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate review counts as a listing-level demand proxy."""
    latest_date = reviews_clean_df["date"].max()
    trailing_cutoff = latest_date - pd.Timedelta(days=365)

    total_reviews = (
        reviews_clean_df.groupby("listing_id", as_index=False)
        .size()
        .rename(columns={"size": "reviews_total"})
    )
    trailing_reviews = (
        reviews_clean_df[reviews_clean_df["date"] >= trailing_cutoff]
        .groupby("listing_id", as_index=False)
        .size()
        .rename(columns={"size": "reviews_last_365d"})
    )

    summary = total_reviews.merge(trailing_reviews, on="listing_id", how="left")
    summary["reviews_last_365d"] = summary["reviews_last_365d"].fillna(0).astype("int64")
    return summary


def build_calendar_summary(calendar_clean_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate calendar rows to listing-level temporal metrics."""
    summary = (
        calendar_clean_df.groupby("listing_id", as_index=False)
        .agg(
            calendar_avg_price=("price", "mean"),
            calendar_median_price=("price", "median"),
            calendar_availability_rate=("available", "mean"),
            calendar_days_sampled=("date", "nunique"),
        )
    )
    summary["calendar_availability_rate"] = summary["calendar_availability_rate"].round(4)
    return summary


def build_analysis_ready_dataset(city_bundle: dict[str, Any]) -> pd.DataFrame:
    """Join cleaned city tables into one listing-level analysis dataset."""
    analysis_df = city_bundle["listings_clean"].copy()
    analysis_df = analysis_df.merge(
        city_bundle["reviews_summary"],
        on="listing_id",
        how="left",
    )

    analysis_df[["reviews_total", "reviews_last_365d"]] = analysis_df[
        ["reviews_total", "reviews_last_365d"]
    ].fillna(0)
    analysis_df["reviews_total"] = analysis_df["reviews_total"].astype("int64")
    analysis_df["reviews_last_365d"] = analysis_df["reviews_last_365d"].astype("int64")

    if "calendar_clean" in city_bundle:
        calendar_summary = build_calendar_summary(city_bundle["calendar_clean"])
        analysis_df = analysis_df.merge(calendar_summary, on="listing_id", how="left")

    return analysis_df


def build_listing_price_lookup(listings_df: pd.DataFrame) -> dict[int, float]:
    """Build listing-level median price lookup for calendar price fallback."""
    if "id" not in listings_df.columns or "price" not in listings_df.columns:
        return {}

    listing_prices = listings_df[["id", "price"]].copy()
    listing_prices["id"] = pd.to_numeric(listing_prices["id"], errors="coerce")
    listing_prices["price"] = _clean_price_column(listing_prices["price"])
    listing_prices = listing_prices.dropna(subset=["id", "price"])
    listing_prices = listing_prices[listing_prices["price"] > 0]

    if listing_prices.empty:
        return {}

    lookup_series = listing_prices.groupby("id")["price"].median()
    return {int(k): float(v) for k, v in lookup_series.items()}


def clean_calendar(
    calendar_df: pd.DataFrame,
    city_key: str,
    listing_price_fallback: dict[int, float],
    global_price_fallback: float,
) -> pd.DataFrame:
    """Clean sampled calendar data for temporal price and availability analysis."""
    cleaned = calendar_df.copy()

    required = ["listing_id", "date", "available", "price"]
    for col in required:
        if col not in cleaned.columns:
            raise KeyError(f"Expected '{col}' column in calendar dataset.")

    cleaned["listing_id"] = pd.to_numeric(cleaned["listing_id"], errors="coerce")
    cleaned["date"] = pd.to_datetime(cleaned["date"], errors="coerce")
    cleaned["price"] = _clean_price_column(cleaned["price"])

    # Calendar data can have sparse prices; use listing-level and global fallbacks.
    cleaned["price"] = cleaned["price"].fillna(
        cleaned["listing_id"].map(listing_price_fallback)
    )
    cleaned["price"] = cleaned["price"].fillna(global_price_fallback)

    cleaned["available"] = (
        cleaned["available"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"t": 1, "f": 0, "true": 1, "false": 0})
    )

    cleaned = cleaned.dropna(subset=["listing_id", "date", "price", "available"])
    cleaned["listing_id"] = cleaned["listing_id"].astype("int64")
    cleaned["available"] = cleaned["available"].astype("int64")

    keep_cols = [
        "listing_id",
        "date",
        "available",
        "price",
        "minimum_nights",
        "maximum_nights",
    ]
    for col in keep_cols:
        if col not in cleaned.columns:
            cleaned[col] = pd.NA

    cleaned = cleaned[keep_cols].drop_duplicates(subset=["listing_id", "date"])
    cleaned["city"] = city_key
    return cleaned.reset_index(drop=True)


def save_cleaned_datasets(cleaned_data: dict[str, dict[str, Any]]) -> list[Path]:
    """Persist cleaned listing datasets to data/processed."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    saved_files: list[Path] = []

    for city_key, city_bundle in cleaned_data.items():
        listings_path = PROCESSED_DIR / f"{city_key}_listings_clean.csv"
        city_bundle["listings_clean"].to_csv(listings_path, index=False)
        saved_files.append(listings_path)

        reviews_path = PROCESSED_DIR / f"{city_key}_reviews_clean.csv"
        city_bundle["reviews_clean"].to_csv(reviews_path, index=False)
        saved_files.append(reviews_path)

        summary_path = PROCESSED_DIR / f"{city_key}_reviews_summary.csv"
        city_bundle["reviews_summary"].to_csv(summary_path, index=False)
        saved_files.append(summary_path)

        analysis_ready = build_analysis_ready_dataset(city_bundle)
        analysis_path = PROCESSED_DIR / f"{city_key}_analysis_ready.csv"
        analysis_ready.to_csv(analysis_path, index=False)
        saved_files.append(analysis_path)

        if "calendar_clean" in city_bundle:
            calendar_path = PROCESSED_DIR / f"{city_key}_calendar_sample_clean.csv"
            city_bundle["calendar_clean"].to_csv(calendar_path, index=False)
            saved_files.append(calendar_path)

    return saved_files


def main() -> None:
    all_data = load_all_cities(
        include_calendar=True,
        calendar_sample_rows=CALENDAR_SAMPLE_ROWS,
    )
    all_data = clean_all_cities(all_data)
    saved_files = save_cleaned_datasets(all_data)

    for city_key, data in all_data.items():
        print(f"\n{city_key.upper()} DATA LOADED")
        print(f"Listings: {data['listings'].shape}")
        print(f"Listings (clean): {data['listings_clean'].shape}")
        print(f"Reviews: {data['reviews'].shape}")
        print(f"Neighbourhoods: {data['neighbourhoods'].shape}")
        print(f"Reviews (clean): {data['reviews_clean'].shape}")
        if "calendar_clean" in data:
            print(f"Calendar (clean sample): {data['calendar_clean'].shape}")

    print("\nSaved cleaned files:")
    for file_path in saved_files:
        print(file_path)


if __name__ == "__main__":
    main()