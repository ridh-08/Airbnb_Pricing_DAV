"""
Standalone host-strategy clustering runner.

Pools all cities' *_featured.csv, then calls the (already-patched)
src.clustering.run_host_strategy_segmentation(). Outputs:

  outputs/multi_city/tables/host_strategy_cluster_labels.csv
  outputs/multi_city/tables/host_strategy_cluster_summary.csv
  outputs/multi_city/tables/host_strategy_silhouette_scan.csv

Usage (from project root):
    python scripts/run_host_strategy_standalone.py

LA is excluded by default (its prices are 100% imputed; see §5.14 of
the report). The $10,000 price cap in run_host_strategy_segmentation
handles any residual artefact rows.

Typical runtime: ~30 seconds.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUTS_DIR = BASE_DIR / "outputs"

CITIES = ("newyork", "chicago", "nashville", "neworleans", "austin", "losangeles")


def main() -> int:
    # Pool all cities' featured files into one long frame
    frames = []
    for city in CITIES:
        path = PROCESSED_DIR / f"{city}_featured.csv"
        if not path.exists():
            print(f"  [{city}] SKIP — {path} not found")
            continue
        df = pd.read_csv(path)
        df["city"] = city  # ensure city column is set
        frames.append(df)
        print(f"  [{city}] loaded {len(df):,} rows")

    if not frames:
        print("No _featured.csv files found under data/processed/.")
        return 1

    pooled = pd.concat(frames, ignore_index=True)
    print(f"\n  pooled total: {len(pooled):,} rows")

    # Call the patched function in src.clustering
    from src.clustering import run_host_strategy_segmentation

    out_dir = OUTPUTS_DIR / "multi_city" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    result = run_host_strategy_segmentation(
        pooled,
        city_name=None,  # multi-city pooled run
        output_dir=out_dir,
        random_state=42,
        exclude_cities=("losangeles",),
        price_cap=10_000.0,
    )

    print(f"\n  chosen k = {result['chosen_k']}")
    print(f"  silhouette scan: {result['silhouette_scan']}")
    print(f"  wrote {result['labels_path'].name}")
    print(f"  wrote {result['summary_path'].name}")
    print(f"  wrote {result['scan_path'].name}")
    print("\n  cluster summary:")
    print(result["summary_stats"].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
