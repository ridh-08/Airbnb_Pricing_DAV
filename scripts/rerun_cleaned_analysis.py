"""
Reproduce all cleaned numbers reported in the rewritten Results section.

This script takes the original pipeline's host-level output
(outputs/multi_city/tables/host_strategy_cluster_labels.csv) and
regenerates:

  - Clean host-strategy clusters (Table 5.3)
  - Clean per-city summary (Figure 5.2 headline numbers)
  - Clean Kruskal-Wallis + pairwise Mann-Whitney with effect sizes (§5.12)

All outputs are written to outputs/cleaned/ alongside a manifest CSV.

Usage:
    python scripts/rerun_cleaned_analysis.py
    python scripts/rerun_cleaned_analysis.py --out outputs/cleaned_v2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# Local imports — assumes this script sits in scripts/ and src/ is on
# the Python path (as in the existing project layout).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_quality_filter import apply_price_artefact_filter  # type: ignore
from src.clustering_patch import cluster_host_strategies  # type: ignore
from src.statistical_tests_patch import (  # type: ignore
    pairwise_price_tests_with_effects,
    effect_size_summary,
)


DEFAULT_INPUT = Path("outputs/multi_city/tables/host_strategy_cluster_labels.csv")
DEFAULT_OUTPUT = Path("outputs/cleaned")

# Cities to exclude from the cleaned analysis. LA is excluded because
# its processed host file contains only 3 distinct prices ($170, $80,
# $39) — see audit_host_prices.py for the sanity check that flags this.
EXCLUDED_CITIES = ("losangeles",)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: {args.input} not found", file=sys.stderr)
        return 2

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Reading {args.input}")
    raw = pd.read_csv(args.input)
    print(f"  rows: {len(raw)}")

    # -----------------------------------------------------------------
    # Stage 1 — Data cleanup
    # -----------------------------------------------------------------
    print("\n[1/3] Applying price-artefact filter and excluding problem cities")
    cleaned = apply_price_artefact_filter(
        raw, price_col="avg_price", verbose=True
    )
    n_before = len(cleaned)
    cleaned = cleaned[~cleaned["city"].isin(EXCLUDED_CITIES)].reset_index(drop=True)
    print(f"  excluded {n_before - len(cleaned)} rows from {EXCLUDED_CITIES}")
    print(f"  final analytic sample: {len(cleaned)}")

    # Per-city summary for Figure 5.2
    city_stats = (
        cleaned.groupby("city")
        .agg(
            n=("host_id", "count"),
            median_price=("avg_price", "median"),
            mean_price=("avg_price", "mean"),
            q25=("avg_price", lambda x: x.quantile(0.25)),
            q75=("avg_price", lambda x: x.quantile(0.75)),
            avg_availability=("availability_365", "mean"),
        )
        .round(2)
        .sort_values("median_price", ascending=False)
    )
    city_stats.to_csv(args.out / "cleaned_city_stats.csv")
    print("\nPer-city cleaned summary:")
    print(city_stats.to_string())

    # -----------------------------------------------------------------
    # Stage 2 — Host strategy clustering on cleaned data
    # -----------------------------------------------------------------
    print("\n[2/3] Re-clustering host strategies (silhouette-optimal k)")
    clustered = cluster_host_strategies(
        cleaned,
        k=None,
        k_range=(2, 7),
        random_state=args.seed,
    )
    print(f"  chosen k = {clustered['chosen_k']}")
    print(f"  silhouette scan: {clustered['silhouette_scan']}")
    clustered["summary_stats"].to_csv(
        args.out / "cleaned_host_clusters.csv", index=False
    )
    pd.DataFrame(
        clustered["silhouette_scan"], columns=["k", "silhouette"]
    ).to_csv(args.out / "cleaned_silhouette_scan.csv", index=False)
    print("\nCleaned cluster summary:")
    print(clustered["summary_stats"].to_string(index=False))

    # -----------------------------------------------------------------
    # Stage 3 — Statistical tests with effect sizes
    # -----------------------------------------------------------------
    print("\n[3/3] Running Kruskal-Wallis + pairwise MW-U with effect sizes")
    global_stats, pairwise = pairwise_price_tests_with_effects(
        cleaned, group_col="city", value_col="avg_price"
    )
    print(f"  Kruskal-Wallis: H = {global_stats['H']:.2f}, p = {global_stats['p']:.2e}")
    print(f"  pairs tested: {len(pairwise)}")
    print(f"  significant after BH: {int(pairwise['significant'].sum())}/{len(pairwise)}")

    pairwise.to_csv(args.out / "cleaned_pairwise_tests.csv", index=False)
    es_summary = effect_size_summary(pairwise)
    es_summary.to_csv(args.out / "cleaned_effect_size_summary.csv", index=False)
    print("\nPairwise results (cleaned):")
    print(pairwise.to_string(index=False))
    print("\nEffect-size buckets:")
    print(es_summary.to_string(index=False))

    # -----------------------------------------------------------------
    # Manifest
    # -----------------------------------------------------------------
    manifest = {
        "input": str(args.input),
        "output_dir": str(args.out),
        "analytic_sample_size": len(cleaned),
        "excluded_cities": list(EXCLUDED_CITIES),
        "chosen_k": clustered["chosen_k"],
        "kruskal_H": global_stats["H"],
        "kruskal_p": global_stats["p"],
        "pairwise_significant": int(pairwise["significant"].sum()),
        "pairwise_total": len(pairwise),
        "max_effect_size": (
            float(pairwise["rbc"].abs().max()) if len(pairwise) else 0.0
        ),
        "median_effect_size": (
            float(pairwise["rbc"].abs().median()) if len(pairwise) else 0.0
        ),
        "seed": args.seed,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest written to {args.out / 'manifest.json'}")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
