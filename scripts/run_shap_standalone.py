"""
Standalone SHAP runner for all cities.

Loads each city's *_featured.csv, refits the XGBoost model with the
same design matrix the pipeline uses, computes SHAP values on a
subsample for speed, and saves:

  outputs/<city>/plots/<city>_xgboost_shap_beeswarm_top10.png
  outputs/<city>/tables/<city>_xgboost_shap_summary.csv
  outputs/multi_city/tables/shap_top5_features_by_city.csv

Usage (from project root):
    python scripts/run_shap_standalone.py
    python scripts/run_shap_standalone.py --cities newyork chicago
    python scripts/run_shap_standalone.py --sample 1500      # faster

Tuning for speed:
  --sample 2000   subsample rows before explaining (default 2000).
                  SHAP scales ~linearly in rows; the visual is
                  indistinguishable below ~2000.
  --trees 150     reduce n_estimators for a faster fit (default 250).

Typical runtime with defaults: ~1-2 min per city on a laptop.
LA is skipped by default because its prices are 100% imputed.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CITY_FOLDERS, PROCESSED_DIR, OUTPUTS_DIR

# All cities the pipeline currently processes
DEFAULT_CITIES = tuple(k for k in CITY_FOLDERS.keys() if k != "losangeles")

# Same numeric feature set the pipeline uses for XGBoost
# (see src/regression_analysis.py XGB_NUMERIC_CANDIDATES).
NUMERIC_FEATURES = [
    "accommodates",
    "amenities_count",
    "availability_365",
    "review_density",
    "reviews_per_month",
    "minimum_nights",
    "number_of_reviews",
    "demand_score",
    "latitude",
    "longitude",
]


def build_design_matrix(
    df: pd.DataFrame,
    top_n_neighbourhoods: int = 30,
) -> tuple[pd.DataFrame, pd.Series]:
    """Rebuild the same design matrix XGBoost trained on."""
    work = df.copy()

    # Numeric coercion
    work["price"] = pd.to_numeric(work["price"], errors="coerce")
    work = work[work["price"] > 0].copy()

    # Mirror the pipeline's winsorisation: 1st/99.5th per city
    if len(work) >= 500:
        lo = work["price"].quantile(0.01)
        hi = work["price"].quantile(0.995)
        work = work[(work["price"] >= lo) & (work["price"] <= hi)].copy()

    for col in NUMERIC_FEATURES:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    numeric_cols = [c for c in NUMERIC_FEATURES if c in work.columns]
    work = work.dropna(subset=numeric_cols + ["price"])
    if work.empty:
        raise ValueError("No valid rows after numeric coercion.")

    X_num = work[numeric_cols].astype(float).reset_index(drop=True)

    parts = [X_num]
    if "room_type" in work.columns:
        room = work["room_type"].astype(str).replace({"nan": "unknown"}).fillna("unknown")
        parts.append(pd.get_dummies(room.reset_index(drop=True), prefix="room"))

    if "neighbourhood" in work.columns:
        neigh = work["neighbourhood"].astype(str).fillna("OTHER")
        top = neigh.value_counts().head(top_n_neighbourhoods).index.tolist()
        neigh = neigh.where(neigh.isin(top), other="OTHER")
        parts.append(pd.get_dummies(neigh.reset_index(drop=True), prefix="neigh"))

    X = pd.concat(parts, axis=1).astype(float)
    X = X.loc[:, ~X.columns.duplicated()]

    y = np.log1p(work["price"].astype(float).reset_index(drop=True))
    return X, y


def run_city(city: str, args: argparse.Namespace) -> dict:
    featured = PROCESSED_DIR / f"{city}_featured.csv"
    if not featured.exists():
        print(f"  [{city}] SKIP — {featured} not found")
        return {}

    print(f"\n[{city}] loading {featured.name}")
    df = pd.read_csv(featured)
    print(f"  rows: {len(df):,}")

    X, y = build_design_matrix(df)
    print(f"  design matrix: {X.shape}")

    # Fit XGBoost — match the pipeline's config closely
    from xgboost import XGBRegressor
    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=args.trees,
        learning_rate=0.06,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )
    t0 = time.time()
    model.fit(X, y)
    print(f"  fit done in {time.time()-t0:.1f}s")

    # Subsample before SHAP — the visual is identical, runtime is 10x faster
    n_sample = min(args.sample, len(X))
    X_sample = X.sample(n=n_sample, random_state=42)

    import shap
    # Use model.get_booster() to force TreeExplainer down the fast C++ path
    explainer = shap.TreeExplainer(model.get_booster())

    t0 = time.time()
    shap_vals = explainer.shap_values(X_sample)
    print(f"  SHAP done in {time.time()-t0:.1f}s on {n_sample} rows")

    # Save summary CSV (all features, ranked)
    mean_abs = np.abs(shap_vals).mean(axis=0)
    summary = pd.DataFrame({"feature": X_sample.columns, "mean_abs_shap": mean_abs})
    summary = summary.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    tables_dir = OUTPUTS_DIR / city / "tables"
    plots_dir = OUTPUTS_DIR / city / "plots"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    summary_path = tables_dir / f"{city}_xgboost_shap_summary.csv"
    summary.to_csv(summary_path, index=False)

    # Beeswarm plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(11, 6.5))
    shap.summary_plot(
        shap_vals, X_sample,
        plot_type="dot", max_display=10, show=False,
    )
    plt.title(f"{city.title()} XGBoost SHAP Beeswarm (Top 10 Features)", pad=12)
    plt.tight_layout()
    plot_path = plots_dir / f"{city}_xgboost_shap_beeswarm_top10.png"
    plt.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close()

    print(f"  wrote {summary_path.name}, {plot_path.name}")
    print(f"  top-5 features: {summary['feature'].head(5).tolist()}")

    return {
        "city": city,
        "top5": summary.head(5),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cities", nargs="*", default=list(DEFAULT_CITIES),
                        help="Cities to process (default: all except LA).")
    parser.add_argument("--sample", type=int, default=2000,
                        help="Rows to subsample for SHAP explain (default 2000).")
    parser.add_argument("--trees", type=int, default=250,
                        help="XGBoost n_estimators (default 250).")
    args = parser.parse_args()

    print(f"Processing cities: {args.cities}")
    print(f"SHAP sample size: {args.sample}, XGBoost trees: {args.trees}\n")

    results = []
    for city in args.cities:
        try:
            out = run_city(city, args)
            if out:
                results.append(out)
        except Exception as e:
            print(f"  [{city}] FAILED: {e}")
            import traceback
            traceback.print_exc()

    # Cross-city rank table
    if results:
        rows = []
        for r in results:
            for rank, feat in enumerate(r["top5"]["feature"].tolist(), start=1):
                rows.append({"city": r["city"], "rank": rank, "feature": feat})
        cross_df = pd.DataFrame(rows)

        multi_tables = OUTPUTS_DIR / "multi_city" / "tables"
        multi_tables.mkdir(parents=True, exist_ok=True)

        wide = cross_df.pivot(index="rank", columns="city", values="feature")
        wide.to_csv(multi_tables / "shap_top5_features_by_city.csv")
        print(f"\n✓ Wrote {multi_tables / 'shap_top5_features_by_city.csv'}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
