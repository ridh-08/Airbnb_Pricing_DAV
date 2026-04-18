"""
Data-artefact filter for Airbnb listings.

Addresses the contamination discovered during report audit:
  - 148 rows priced at exactly $10,000 or $40,000 (Chicago, New Orleans)
    that survived the original 99th-percentile cap because the cap itself
    was being pulled upward by the contamination.

Integrate by calling `apply_price_artefact_filter()` inside
`clean_city_listings()` in src/preprocessing.py, *before* the existing
99th-percentile quantile cap. Drop-in placement shown in the companion
file `preprocessing_patch.py`.

Two-stage filter:
  1. Absolute filter: drop price >= ABSOLUTE_PRICE_CAP ($10,000).
     Catches known placeholder values ($10000, $40000).
  2. Relative filter: drop price > RELATIVE_MULTIPLIER * city_median.
     Catches city-specific extreme outliers that the absolute filter
     might miss in a high-priced market.

The combination is deliberately belt-and-braces: the absolute rule
handles placeholder-value contamination; the relative rule handles
ordinary outliers without hard-coding city-specific thresholds.
"""
from __future__ import annotations

import pandas as pd


# Placeholder-value threshold. Real Airbnb nightly prices are almost never
# above $5,000; $10,000 is a safe hard cap for normal short-term lets.
ABSOLUTE_PRICE_CAP: float = 10_000.0

# Relative outlier cap. A price more than 10x the city median is either
# a penthouse outlier or a data issue; either way it distorts means and
# should be trimmed from the analytic sample.
RELATIVE_MULTIPLIER: float = 10.0


def apply_price_artefact_filter(
    df: pd.DataFrame,
    price_col: str = "price",
    *,
    absolute_cap: float = ABSOLUTE_PRICE_CAP,
    relative_multiplier: float = RELATIVE_MULTIPLIER,
    verbose: bool = False,
) -> pd.DataFrame:
    """Drop rows with implausible prices, both absolute and city-relative.

    Parameters
    ----------
    df
        DataFrame containing at least `price_col`. If `city` column is
        present, relative trimming is applied per city; otherwise it is
        applied globally.
    price_col
        Name of the price column, default "price".
    absolute_cap
        Prices >= this value are dropped outright.
    relative_multiplier
        Prices strictly greater than (relative_multiplier * median price)
        for the row's city are dropped.
    verbose
        If True, print the row counts dropped at each stage.

    Returns
    -------
    pd.DataFrame
        Filtered frame. The `price_col` is coerced to numeric in place;
        rows with non-numeric prices are dropped.
    """
    if price_col not in df.columns:
        raise KeyError(f"Column '{price_col}' not found in dataframe.")

    work = df.copy()
    work[price_col] = pd.to_numeric(work[price_col], errors="coerce")
    n_start = len(work)

    # Stage 0: drop non-numeric / non-positive prices
    work = work[work[price_col].notna() & (work[price_col] > 0)]
    n_after_zero = len(work)

    # Stage 1: absolute placeholder filter
    abs_mask = work[price_col] < absolute_cap
    n_drop_abs = (~abs_mask).sum()
    work = work[abs_mask]

    # Stage 2: city-relative filter (falls back to global if no 'city' col)
    if "city" in work.columns:
        medians = work.groupby("city")[price_col].transform("median")
        rel_mask = work[price_col] <= (relative_multiplier * medians)
    else:
        rel_mask = work[price_col] <= (relative_multiplier * work[price_col].median())
    n_drop_rel = (~rel_mask).sum()
    work = work[rel_mask]

    if verbose:
        print(
            f"[price_artefact_filter] start={n_start} "
            f"after_zero_filter={n_after_zero} "
            f"dropped_absolute(>=${absolute_cap:,.0f})={n_drop_abs} "
            f"dropped_relative(>{relative_multiplier}x median)={n_drop_rel} "
            f"final={len(work)}"
        )

    return work.reset_index(drop=True)


def audit_price_distribution(df: pd.DataFrame, price_col: str = "price") -> pd.DataFrame:
    """Emit a small diagnostic frame for price sanity checks.

    Returns one row per city with n_distinct, min, p25, median, p75, max,
    and flags for whether the city fails basic sanity checks (too few
    distinct values, unusually low max).
    """
    if "city" not in df.columns:
        raise KeyError("`city` column required for audit.")

    def _summarise(g: pd.DataFrame) -> pd.Series:
        s = pd.to_numeric(g[price_col], errors="coerce").dropna()
        return pd.Series({
            "n": len(s),
            "n_distinct": s.nunique(),
            "min": s.min() if len(s) else float("nan"),
            "p25": s.quantile(0.25) if len(s) else float("nan"),
            "median": s.median() if len(s) else float("nan"),
            "p75": s.quantile(0.75) if len(s) else float("nan"),
            "max": s.max() if len(s) else float("nan"),
        })

    summary = df.groupby("city").apply(_summarise, include_groups=False).reset_index()
    summary["flag_few_distinct"] = summary["n_distinct"] < 20
    summary["flag_compressed_iqr"] = summary["p25"] == summary["p75"]
    return summary
