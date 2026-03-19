from __future__ import annotations

from itertools import combinations
from pathlib import Path

import pandas as pd
from scipy.stats import ks_2samp, mannwhitneyu


BASE_DIR = Path(__file__).resolve().parent.parent
TABLES_DIR = BASE_DIR / "outputs" / "tables"


def _interpret_pvalue(p_value: float, alpha: float) -> str:
    """Return significance interpretation for a p-value."""
    return "significant" if p_value < alpha else "not significant"


def _price_series(df: pd.DataFrame) -> pd.Series:
    """Return clean positive price values."""
    if "price" not in df.columns:
        raise KeyError("Input dataframe must contain 'price'.")
    series = pd.to_numeric(df["price"], errors="coerce")
    return series[series > 0].dropna()


def run_statistical_comparison(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Run price distribution tests across cities and room types and return summary table."""
    required_cols = {"city", "room_type", "price"}
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    work = df.copy()
    work["city"] = work["city"].astype(str).str.lower()
    results: list[dict[str, object]] = []

    ny = _price_series(work[work["city"] == "newyork"])
    chicago = _price_series(work[work["city"] == "chicago"])

    if len(ny) == 0 or len(chicago) == 0:
        raise ValueError("Both New York and Chicago require at least one valid positive price.")

    mw_stat, mw_p = mannwhitneyu(ny, chicago, alternative="two-sided")
    results.append(
        {
            "test_name": "Mann-Whitney U (NY vs Chicago)",
            "group_1": "newyork",
            "group_2": "chicago",
            "statistic": float(mw_stat),
            "p_value": float(mw_p),
            "interpretation": _interpret_pvalue(float(mw_p), alpha),
        }
    )

    ks_stat, ks_p = ks_2samp(ny, chicago, alternative="two-sided", method="auto")
    results.append(
        {
            "test_name": "Kolmogorov-Smirnov (NY vs Chicago)",
            "group_1": "newyork",
            "group_2": "chicago",
            "statistic": float(ks_stat),
            "p_value": float(ks_p),
            "interpretation": _interpret_pvalue(float(ks_p), alpha),
        }
    )

    for city in ["newyork", "chicago"]:
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
        saved_path = save_statistical_summary(summary)
        print(f"[statistical_tests] Saved statistical summary to: {saved_path}")
        return summary

    summary = summary.sort_values(["test_name", "p_value"]).reset_index(drop=True)
    saved_path = save_statistical_summary(summary)
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
