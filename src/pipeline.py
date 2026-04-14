from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.calendar_analysis import run_calendar_analysis
from src.clustering import run_clustering
from src.config import CITY_FOLDERS, PROCESSED_DIR, get_city_output_dir
from src.feature_engineering import run_feature_engineering
from src.multi_city_analysis import run_multi_city_analysis
from src.preprocessing import clean_all_cities, load_all_cities, save_cleaned_datasets
from src.regression_analysis import get_clean_ols_summary, run_regression_analysis
from src.spatial_analysis import run_spatial_analysis
from src.statistical_tests import run_statistical_comparison
from src.visualization import run_eda


def _collect_city_tables(city_outputs: dict[str, dict[str, Any]]) -> dict[str, pd.DataFrame]:
    """Collect and merge all per-city table CSVs keyed by table filename stem."""
    combined: dict[str, list[pd.DataFrame]] = {}

    for city_name, payload in city_outputs.items():
        tables_dir = Path(payload["output_dir"]) / "tables"
        if not tables_dir.exists():
            continue

        for csv_path in sorted(tables_dir.glob("*.csv")):
            df = pd.read_csv(csv_path)
            if "city" not in df.columns:
                df.insert(0, "city", city_name)
            df.insert(1, "source_file", csv_path.name)

            table_key = csv_path.stem
            combined.setdefault(table_key, []).append(df)

    merged = {
        key: pd.concat(parts, ignore_index=True)
        for key, parts in combined.items()
        if parts
    }
    return merged


def export_consolidated_summary(
    city_outputs: dict[str, dict[str, Any]],
    output_dir: Path | None = None,
) -> dict[str, Path]:
    """Export consolidated multi-city summary tables to workbook, with CSV fallback."""
    out_dir = output_dir or (Path(__file__).resolve().parent.parent / "outputs" / "consolidated")
    out_dir.mkdir(parents=True, exist_ok=True)

    merged_tables = _collect_city_tables(city_outputs)
    if not merged_tables:
        raise ValueError("No per-city summary tables were found to consolidate.")

    manifest_rows: list[dict[str, Any]] = []
    for table_name, frame in merged_tables.items():
        manifest_rows.append(
            {
                "table_name": table_name,
                "rows": int(len(frame)),
                "columns": int(len(frame.columns)),
            }
        )

    manifest_df = pd.DataFrame(manifest_rows).sort_values("table_name").reset_index(drop=True)

    workbook_path = out_dir / "airbnb_multicity_summary.xlsx"
    try:
        with pd.ExcelWriter(workbook_path) as writer:
            manifest_df.to_excel(writer, sheet_name="manifest", index=False)

            for table_name, frame in sorted(merged_tables.items()):
                sheet_name = table_name[:31]
                frame.to_excel(writer, sheet_name=sheet_name, index=False)

        return {
            "workbook": workbook_path,
            "manifest": workbook_path,
        }
    except ImportError:
        manifest_path = out_dir / "manifest.csv"
        manifest_df.to_csv(manifest_path, index=False)

        for table_name, frame in merged_tables.items():
            frame.to_csv(out_dir / f"{table_name}.csv", index=False)

        return {
            "workbook": manifest_path,
            "manifest": manifest_path,
        }


def load_city_dataframe_dict(suffix: str) -> dict[str, pd.DataFrame]:
    """Load processed city files into a dictionary: {city_name: dataframe}."""
    city_frames: dict[str, pd.DataFrame] = {}
    for city in CITY_FOLDERS:
        path = PROCESSED_DIR / f"{city}_{suffix}.csv"
        if path.exists():
            city_frames[city] = pd.read_csv(path)
    return city_frames


def _save_regression_outputs(result: dict[str, Any], city_name: str, tables_dir: Path) -> None:
    city_result = result["city_results"][city_name]
    ols_raw_path = tables_dir / "regression_ols_summary_raw.csv"
    city_result["ols_summary"].to_csv(ols_raw_path, index=False)

    clean_ols = get_clean_ols_summary(city_result["ols_summary"])
    clean_ols_path = tables_dir / "regression_ols_summary_clean.csv"
    clean_ols.to_csv(clean_ols_path, index=False)

    rf_path = tables_dir / "regression_rf_importance.csv"
    city_result["rf_importance"].to_csv(rf_path, index=False)

    top_cmp_path = tables_dir / "regression_top_feature_comparison.csv"
    result["top_feature_comparison"].to_csv(top_cmp_path, index=False)


def run_full_pipeline_for_city(city_name: str) -> dict[str, Any]:
    """Run EDA, clustering-dependent analyses, regression, and exports for one city."""
    city_key = city_name.lower().strip().replace(" ", "")
    city_output = get_city_output_dir(city_key)
    plots_dir = city_output / "plots"
    tables_dir = city_output / "tables"
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    insights, quality_table = run_eda(
        city_name=city_key,
        plots_output_dir=plots_dir,
        tables_output_dir=tables_dir,
    )

    clustered_path = PROCESSED_DIR / f"{city_key}_clustered.csv"
    clustered_df = pd.read_csv(clustered_path)

    regression_result = run_regression_analysis(
        clustered_df,
        output_dir=plots_dir,
        city_name=city_key,
    )
    _save_regression_outputs(regression_result, city_key, tables_dir)

    run_statistical_comparison(
        clustered_df,
        city_name=city_key,
        output_dir=tables_dir,
    )

    run_spatial_analysis(
        city_name=city_key,
        plots_output_dir=plots_dir,
        tables_output_dir=tables_dir,
    )

    calendar_outputs: dict[str, Any] = {}
    try:
        cal_plots, cal_table = run_calendar_analysis(
            city_name=city_key,
            plots_output_dir=plots_dir,
            tables_output_dir=tables_dir,
        )
        calendar_outputs = {"plots": cal_plots, "table": cal_table}
    except FileNotFoundError:
        calendar_outputs = {"plots": [], "table": None}

    return {
        "city": city_key,
        "output_dir": city_output,
        "quality_table": quality_table,
        "insights": insights,
        "calendar": calendar_outputs,
    }


def run_full_pipeline_all_cities() -> dict[str, Any]:
    """Run the complete Airbnb analytics pipeline for all configured cities."""
    raw_city_data = load_all_cities(include_calendar=True)
    cleaned_city_data = clean_all_cities(raw_city_data)
    saved_clean_paths = save_cleaned_datasets(cleaned_city_data)

    featured_paths = run_feature_engineering()
    clustered_paths, cluster_summary = run_clustering()

    city_dataframes = load_city_dataframe_dict("clustered")

    city_outputs: dict[str, dict[str, Any]] = {}
    for city_name in sorted(city_dataframes.keys()):
        city_outputs[city_name] = run_full_pipeline_for_city(city_name)

    multi_city_outputs = run_multi_city_analysis(city_dataframes)
    consolidated_outputs = export_consolidated_summary(city_outputs)

    return {
        "saved_clean_paths": saved_clean_paths,
        "featured_paths": featured_paths,
        "clustered_paths": clustered_paths,
        "cluster_summary": cluster_summary,
        "city_dataframes": city_dataframes,
        "city_outputs": city_outputs,
        "multi_city_outputs": multi_city_outputs,
        "consolidated_outputs": consolidated_outputs,
    }


def main() -> None:
    results = run_full_pipeline_all_cities()
    print("Pipeline completed for cities:")
    for city in sorted(results["city_outputs"].keys()):
        print(f"- {city}: {results['city_outputs'][city]['output_dir']}")
    print(f"Multi-city outputs: {results['multi_city_outputs']['output_dir']}")
    print(f"Consolidated summary: {results['consolidated_outputs']['workbook']}")


if __name__ == "__main__":
    main()
