from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Allow running this script directly from scripts/ while importing src.*
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CITY_FOLDERS, PROCESSED_DIR, get_city_output_dir
from src.pipeline import _save_regression_outputs
from src.regression_analysis import run_regression_analysis


def main() -> int:
    success = []
    failed = []

    for city in sorted(CITY_FOLDERS.keys()):
        clustered_path = PROCESSED_DIR / f"{city}_clustered.csv"
        if not clustered_path.exists():
            print(f"[shap] SKIP {city}: missing {clustered_path}")
            failed.append((city, "missing_clustered_input"))
            continue

        try:
            print(f"[shap] START {city}", flush=True)
            city_df = pd.read_csv(clustered_path)
            city_output_dir = get_city_output_dir(city)
            plots_dir = city_output_dir / "plots"
            tables_dir = city_output_dir / "tables"
            plots_dir.mkdir(parents=True, exist_ok=True)
            tables_dir.mkdir(parents=True, exist_ok=True)

            result = run_regression_analysis(
                city_df,
                output_dir=plots_dir,
                tables_output_dir=tables_dir,
                city_name=city,
            )
            print(f"[shap] REGRESSION_DONE {city}", flush=True)
            _save_regression_outputs(result, city, tables_dir)

            shap_table = tables_dir / "shap_feature_importance.csv"
            if shap_table.exists():
                print(f"[shap] SHAP_TABLE_WRITTEN {city}: {shap_table.name}", flush=True)

            beeswarm_files = sorted(plots_dir.glob(f"{city}_xgboost_shap_beeswarm_top*.png"))

            if shap_table.exists() and beeswarm_files:
                print(
                    f"[shap] DONE {city}: table={shap_table.name}, "
                    f"beeswarm={beeswarm_files[-1].name}"
                )
                success.append(city)
            else:
                print(
                    f"[shap] WARN {city}: outputs missing "
                    f"(table_exists={shap_table.exists()}, beeswarm_count={len(beeswarm_files)})"
                )
                failed.append((city, "outputs_missing"))

            print(f"[shap] END {city}", flush=True)

        except Exception as exc:  # pragma: no cover
            print(f"[shap] FAIL {city}: {exc}")
            failed.append((city, str(exc)))

    print("\n[shap] SUMMARY")
    print(f"[shap] success_cities={success}")
    print(f"[shap] failed_count={len(failed)}")
    for city, reason in failed:
        print(f"[shap] failed_city={city} reason={reason}")

    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
