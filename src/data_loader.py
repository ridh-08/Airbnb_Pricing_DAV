from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

import numpy as np
import pandas as pd

from src.preprocessing import load_all_cities, load_geojson, load_tabular_data


def load_project_data(include_calendar: bool = False) -> dict[str, dict[str, Any]]:
    """Convenience wrapper for loading Airbnb city datasets."""
    return load_all_cities(include_calendar=include_calendar)


def _clean_listings_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Apply lightweight cleaning to listings dataframe after load."""
    cleaned = df.copy()

    if "id" in cleaned.columns and "listing_id" not in cleaned.columns:
        cleaned["listing_id"] = cleaned["id"]

    if "price" in cleaned.columns:
        cleaned["price"] = pd.to_numeric(
            cleaned["price"].astype(str).str.replace(r"[$,]", "", regex=True),
            errors="coerce",
        )

    return cleaned


def _clean_calendar_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Apply lightweight cleaning to calendar dataframe after load."""
    cleaned = df.copy()

    if "date" in cleaned.columns:
        cleaned["date"] = pd.to_datetime(cleaned["date"], errors="coerce")

    if "price" in cleaned.columns:
        cleaned["price"] = pd.to_numeric(
            cleaned["price"].astype(str).str.replace(r"[$,]", "", regex=True),
            errors="coerce",
        )

    if "available" in cleaned.columns:
        cleaned["available"] = (
            cleaned["available"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"t": 1, "f": 0, "true": 1, "false": 0})
        )

    return cleaned


def load_multiple_cities(
    data_path_dict: dict[str, dict[str, str | Path]],
) -> dict[str, dict[str, Any]]:
    """Load multiple cities from explicit file paths.

    Parameters
    ----------
    data_path_dict:
        Mapping of city name to input paths, e.g.
        {
            "newyork": {
                "listings": ".../listings.csv",
                "calendar": ".../calendar.csv",
                "neighbourhoods": ".../neighbourhoods.geojson",
            }
        }

    Returns
    -------
    dict[str, dict[str, Any]]
        Per-city payload aligned with load_city_data() schema:
        {
            "city": city_name,
            "listings": DataFrame,
            "calendar": DataFrame (if available),
            "neighbourhoods": GeoDataFrame,
        }
        Cities missing required files are skipped with warnings.
    """
    loaded: dict[str, dict[str, Any]] = {}

    for city_name, paths in data_path_dict.items():
        city_key = city_name.lower().strip().replace(" ", "")

        listings_path = paths.get("listings") or paths.get("listings_csv")
        calendar_path = paths.get("calendar") or paths.get("calendar_csv")
        neighbourhoods_path = (
            paths.get("neighbourhoods")
            or paths.get("neighbourhood")
            or paths.get("geojson")
        )

        missing_required: list[str] = []
        if listings_path is None:
            missing_required.append("listings")
        if neighbourhoods_path is None:
            missing_required.append("neighbourhoods")

        if missing_required:
            warnings.warn(
                f"Skipping city '{city_name}': missing required path(s): {missing_required}"
            )
            continue

        listings_file = Path(listings_path)
        neighbourhoods_file = Path(neighbourhoods_path)
        calendar_file = Path(calendar_path) if calendar_path is not None else None

        if not listings_file.exists() or not listings_file.is_file():
            warnings.warn(
                f"Skipping city '{city_name}': listings file not found: {listings_file}"
            )
            continue
        if not neighbourhoods_file.exists() or not neighbourhoods_file.is_file():
            warnings.warn(
                f"Skipping city '{city_name}': neighbourhood geojson not found: {neighbourhoods_file}"
            )
            continue

        city_payload: dict[str, Any] = {
            "city": city_key,
            "listings": _clean_listings_frame(load_tabular_data(listings_file)),
            "neighbourhoods": load_geojson(neighbourhoods_file),
        }

        if calendar_file is not None:
            if calendar_file.exists() and calendar_file.is_file():
                city_payload["calendar"] = _clean_calendar_frame(
                    load_tabular_data(calendar_file)
                )
            else:
                warnings.warn(
                    f"City '{city_name}': calendar file not found, continuing without calendar: {calendar_file}"
                )

        loaded[city_key] = city_payload

    return loaded


def build_city_summary(
    df_dict: dict[str, pd.DataFrame | dict[str, Any]],
) -> pd.DataFrame:
    """Build per-city summary metrics from city dataframes or loader payloads.

    Metrics include:
    - mean price
    - median price
    - standard deviation of price
    - price skewness
    - availability rate
    - number of listings
    """

    rows: list[dict[str, Any]] = []

    for city_name, payload in df_dict.items():
        if isinstance(payload, dict):
            listings_df = payload.get("listings")
            calendar_df = payload.get("calendar")
        else:
            listings_df = payload
            calendar_df = None

        if listings_df is None or not isinstance(listings_df, pd.DataFrame):
            warnings.warn(f"Skipping city '{city_name}': listings dataframe not found.")
            continue

        listings = listings_df.copy()

        if "price" not in listings.columns:
            warnings.warn(f"Skipping city '{city_name}': missing 'price' column in listings.")
            continue

        price = pd.to_numeric(
            listings["price"].astype(str).str.replace(r"[$,]", "", regex=True),
            errors="coerce",
        )
        price = price[price > 0]

        if "listing_id" in listings.columns:
            listing_count = int(pd.to_numeric(listings["listing_id"], errors="coerce").nunique())
        elif "id" in listings.columns:
            listing_count = int(pd.to_numeric(listings["id"], errors="coerce").nunique())
        else:
            listing_count = int(len(listings))

        availability_rate = np.nan
        if isinstance(calendar_df, pd.DataFrame) and "available" in calendar_df.columns:
            available_series = (
                calendar_df["available"]
                .astype(str)
                .str.strip()
                .str.lower()
                .map({"t": 1, "f": 0, "true": 1, "false": 0})
            )
            available_numeric = pd.to_numeric(available_series, errors="coerce")
            availability_rate = float(available_numeric.mean())
        elif "availability_365" in listings.columns:
            avail365 = pd.to_numeric(listings["availability_365"], errors="coerce")
            avail365 = avail365.clip(lower=0, upper=365)
            availability_rate = float((avail365 / 365.0).mean())

        rows.append(
            {
                "city": city_name.lower().strip().replace(" ", ""),
                "mean_price": float(price.mean()) if not price.empty else np.nan,
                "median_price": float(price.median()) if not price.empty else np.nan,
                "std_deviation": float(price.std(ddof=1)) if len(price) > 1 else np.nan,
                "price_skewness": float(price.skew()) if len(price) > 2 else np.nan,
                "availability_rate": availability_rate,
                "number_of_listings": listing_count,
            }
        )

    summary = pd.DataFrame(rows)
    if summary.empty:
        return pd.DataFrame(
            columns=[
                "mean_price",
                "median_price",
                "std_deviation",
                "price_skewness",
                "availability_rate",
                "number_of_listings",
            ]
        )

    summary = summary.sort_values("city").set_index("city")
    return summary
