"""
Data-quality audit for the processed host-level file.

Standalone diagnostic script — run after the pipeline's host-strategy
step to verify that no city has pathologically compressed price variance
(the original bug that caused LA's XGBoost R² ≈ 1.0).

Usage:
    python scripts/audit_host_prices.py
    python scripts/audit_host_prices.py --path outputs/multi_city/tables/host_strategy_cluster_labels.csv

Exit codes:
    0 — all cities pass sanity checks
    1 — at least one city has n_distinct_prices < 20 OR p25 == p75
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


DEFAULT_PATH = Path("outputs/multi_city/tables/host_strategy_cluster_labels.csv")

# Sanity thresholds. A real Airbnb market should have hundreds of
# distinct prices per city and a non-degenerate interquartile range.
MIN_DISTINCT_PRICES = 20
MIN_IQR_RATIO = 0.1  # (p75 - p25) / median should be >= this


def audit_host_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Return per-city audit table with sanity flags."""
    if "city" not in df.columns or "avg_price" not in df.columns:
        raise KeyError(
            "Input must contain 'city' and 'avg_price' columns "
            "(as produced by cluster_host_strategies)."
        )

    rows = []
    for city, g in df.groupby("city"):
        prices = pd.to_numeric(g["avg_price"], errors="coerce").dropna()
        if len(prices) == 0:
            continue
        median = float(prices.median())
        p25 = float(prices.quantile(0.25))
        p75 = float(prices.quantile(0.75))
        iqr_ratio = (p75 - p25) / median if median > 0 else 0.0
        rows.append({
            "city": city,
            "n": len(prices),
            "n_distinct": int(prices.nunique()),
            "min": float(prices.min()),
            "p25": p25,
            "median": median,
            "p75": p75,
            "max": float(prices.max()),
            "iqr_ratio": round(iqr_ratio, 4),
            "flag_few_distinct": int(prices.nunique()) < MIN_DISTINCT_PRICES,
            "flag_compressed_iqr": iqr_ratio < MIN_IQR_RATIO,
        })
    return pd.DataFrame(rows)


def print_audit_report(audit: pd.DataFrame) -> bool:
    """Print a human-readable audit report; return True if any flags set."""
    any_flag = bool(audit[["flag_few_distinct", "flag_compressed_iqr"]].any().any())

    print("Host-level price audit")
    print("=" * 70)
    print(audit.to_string(index=False))
    print()

    bad_distinct = audit.loc[audit["flag_few_distinct"], "city"].tolist()
    bad_iqr = audit.loc[audit["flag_compressed_iqr"], "city"].tolist()
    if bad_distinct:
        print(f"  FAIL: cities with < {MIN_DISTINCT_PRICES} distinct prices: {bad_distinct}")
        print("        Likely processed-data artefact; exclude from per-city")
        print("        XGBoost and host-price clustering until investigated.")
    if bad_iqr:
        print(f"  FAIL: cities with IQR/median < {MIN_IQR_RATIO}: {bad_iqr}")
        print("        Price distribution is too compressed for meaningful modelling.")
    if not any_flag:
        print("  All cities pass sanity checks.")
    return any_flag


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path",
        type=Path,
        default=DEFAULT_PATH,
        help=f"Path to host_strategy_cluster_labels.csv (default: {DEFAULT_PATH})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the audit table as CSV.",
    )
    args = parser.parse_args()

    if not args.path.exists():
        print(f"ERROR: {args.path} does not exist", file=sys.stderr)
        return 2

    df = pd.read_csv(args.path)
    audit = audit_host_prices(df)

    if args.output:
        audit.to_csv(args.output, index=False)
        print(f"  -> wrote audit to {args.output}")

    any_flag = print_audit_report(audit)
    return 1 if any_flag else 0


if __name__ == "__main__":
    sys.exit(main())
