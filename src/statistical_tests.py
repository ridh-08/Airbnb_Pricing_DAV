from __future__ import annotations

from itertools import combinations
from pathlib import Path

import pandas as pd
from scipy.stats import ks_2samp, mannwhitneyu

from src.config import TABLES_DIR, normalize_city_name


BASE_DIR = Path(__file__).resolve().parent.parent


def _interpret_pvalue(p_value: float, alpha: float) -> str:
    """Return significance interpretation for a p-value."""
    return "significant" if p_value < alpha else "not significant"


def _price_series(df: pd.DataFrame) -> pd.Series:
    """Return clean positive price values."""
    if "price" not in df.columns:
        raise KeyError("Input dataframe must contain 'price'.")
    series = pd.to_numeric(df["price"], errors="coerce")
    return series[series > 0].dropna()


def run_statistical_comparison(
    df: pd.DataFrame,
    alpha: float = 0.05,
    city_name: str | None = None,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Run price distribution tests across cities and room types and return summary table."""
    required_cols = {"city", "room_type", "price"}
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    work = df.copy()
    work["city"] = work["city"].astype(str).str.lower()
    if city_name is not None:
        city_key = normalize_city_name(city_name)
        work = work[work["city"] == city_key].copy()

    results: list[dict[str, object]] = []

    cities = sorted(work["city"].dropna().unique().tolist())
    for city_1, city_2 in combinations(cities, 2):
        prices_1 = _price_series(work[work["city"] == city_1])
        prices_2 = _price_series(work[work["city"] == city_2])
        if len(prices_1) == 0 or len(prices_2) == 0:
            continue

        mw_stat, mw_p = mannwhitneyu(prices_1, prices_2, alternative="two-sided")
        results.append(
            {
                "test_name": f"Mann-Whitney U ({city_1} vs {city_2})",
                "group_1": city_1,
                "group_2": city_2,
                "statistic": float(mw_stat),
                "p_value": float(mw_p),
                "interpretation": _interpret_pvalue(float(mw_p), alpha),
            }
        )

        ks_stat, ks_p = ks_2samp(prices_1, prices_2, alternative="two-sided", method="auto")
        results.append(
            {
                "test_name": f"Kolmogorov-Smirnov ({city_1} vs {city_2})",
                "group_1": city_1,
                "group_2": city_2,
                "statistic": float(ks_stat),
                "p_value": float(ks_p),
                "interpretation": _interpret_pvalue(float(ks_p), alpha),
            }
        )

    for city in cities:
        city_df = work[work["city"] == city].copy()
        room_types = [
            room
            for room, values in city_df.groupby("room_type")
            if len(_price_series(values)) > 0
        ]

        for rt1, rt2 in combinations(sorted(room_types), 2):
            prices_1 = _price_series(city_df[city_df["room_type"] == rt1])
            prices_2 = _price_series(city_df[city_df["room_type"] == rt2])
            if len(prices_1) == 0 or len(prices_2) == 0:
                continue

            stat, p_value = mannwhitneyu(prices_1, prices_2, alternative="two-sided")
            results.append(
                {
                    "test_name": f"Mann-Whitney U ({city}: room type)",
                    "group_1": f"{city}:{rt1}",
                    "group_2": f"{city}:{rt2}",
                    "statistic": float(stat),
                    "p_value": float(p_value),
                    "interpretation": _interpret_pvalue(float(p_value), alpha),
                }
            )

    summary = pd.DataFrame(results)
    if summary.empty:
        saved_path = save_statistical_summary(summary, output_dir=output_dir)
        print(f"[statistical_tests] Saved statistical summary to: {saved_path}")
        return summary

    summary = summary.sort_values(["test_name", "p_value"]).reset_index(drop=True)
    saved_path = save_statistical_summary(summary, output_dir=output_dir)
    print(f"[statistical_tests] Saved statistical summary to: {saved_path}")
    return summary


def save_statistical_summary(
    summary_df: pd.DataFrame,
    output_dir: Path | None = None,
) -> Path:
    """Save statistical test summary DataFrame to outputs/tables/."""
    out_dir = output_dir or TABLES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "statistical_tests_summary.csv"
    summary_df.to_csv(out_path, index=False)
    return out_path
