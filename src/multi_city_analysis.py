from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from scipy.stats import kruskal, mannwhitneyu

from src.config import CITY_COLORS, OUTPUTS_DIR
from src.regression_analysis import compare_model_performance, train_xgboost_model


MULTI_CITY_OUTPUT_DIR = OUTPUTS_DIR / "multi_city"


def _combine_city_frames(city_dataframes: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Combine city dataframes into one normalized dataframe."""
    parts: list[pd.DataFrame] = []
    for city_name, df in city_dataframes.items():
        part = df.copy()
        if "city" not in part.columns:
            part["city"] = city_name
        part["city"] = part["city"].astype(str).str.lower()
        parts.append(part)

    if not parts:
        raise ValueError("No city dataframes provided for multi-city analysis.")

    return pd.concat(parts, ignore_index=True)


def _bh_adjust(p_values: list[float]) -> list[float]:
    """Benjamini-Hochberg FDR correction."""
    if not p_values:
        return []

    p = np.array(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.empty(n, dtype=float)

    running = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        value = ranked[i] * n / rank
        running = min(running, value)
        adjusted[i] = running

    adjusted = np.clip(adjusted, 0.0, 1.0)
    out = np.empty(n, dtype=float)
    out[order] = adjusted
    return out.tolist()


def plot_city_median_price(df: pd.DataFrame, output_dir: Path) -> Path:
    """Bar chart of median price by city."""
    work = df.copy()
    work["price"] = pd.to_numeric(work["price"], errors="coerce")
    work = work[work["price"] > 0]

    summary = (
        work.groupby("city", as_index=False)["price"]
        .median()
        .rename(columns={"price": "median_price"})
        .sort_values("median_price", ascending=False)
    )

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=summary,
        x="city",
        y="median_price",
        hue="city",
        legend=False,
        palette=[CITY_COLORS.get(c, "#555555") for c in summary["city"]],
    )
    ax.set_title("Median Listing Price by City")
    ax.set_xlabel("City")
    ax.set_ylabel("Median Price")
    plt.xticks(rotation=15)

    out_path = output_dir / "city_median_price.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    return out_path


def plot_cluster_share_by_city(df: pd.DataFrame, output_dir: Path) -> Path:
    """Stacked bar chart of cluster share per city."""
    if "cluster" not in df.columns:
        raise KeyError("Expected 'cluster' column in clustered dataframe.")

    work = df.copy()
    work["cluster"] = pd.to_numeric(work["cluster"], errors="coerce")
    work = work.dropna(subset=["cluster", "city"]) 
    work["cluster"] = work["cluster"].astype(int)

    count = (
        work.groupby(["city", "cluster"], as_index=False)
        .size()
        .rename(columns={"size": "n"})
    )
    total = count.groupby("city")["n"].transform("sum")
    count["share_pct"] = 100 * count["n"] / total

    pivot = count.pivot(index="city", columns="cluster", values="share_pct").fillna(0)
    pivot = pivot.sort_index()

    fig, ax = plt.subplots(figsize=(11, 6))
    bottom = np.zeros(len(pivot))
    for cluster in sorted(pivot.columns.tolist()):
        vals = pivot[cluster].values
        ax.bar(pivot.index, vals, bottom=bottom, label=f"Cluster {cluster}")
        bottom += vals

    ax.set_title("Cluster Composition by City")
    ax.set_ylabel("Percent of Listings")
    ax.set_xlabel("City")
    ax.legend(ncol=2)

    out_path = output_dir / "cluster_composition_by_city.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def plot_city_feature_heatmap(df: pd.DataFrame, output_dir: Path) -> Path:
    """Heatmap of city-level standardized medians across selected features."""
    candidate_cols = [
        "price",
        "availability_365",
        "number_of_reviews",
        "amenities_count",
        "accommodates",
        "price_per_person",
    ]
    use_cols = [c for c in candidate_cols if c in df.columns]
    if not use_cols:
        raise ValueError("No numeric features available for city feature heatmap.")

    work = df.copy()
    for col in use_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    summary = work.groupby("city", as_index=True)[use_cols].median(numeric_only=True)
    z = (summary - summary.mean()) / summary.std(ddof=0)
    z = z.fillna(0.0)

    plt.figure(figsize=(10, 6))
    sns.heatmap(z, cmap="RdBu_r", center=0, annot=True, fmt=".2f")
    plt.title("City Feature Profile (Z-scored Medians)")
    plt.xlabel("Feature")
    plt.ylabel("City")

    out_path = output_dir / "city_feature_profile_heatmap.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    return out_path


def run_pooled_regressions(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Run pooled fixed-effects and quantile regressions across cities."""
    base_candidates = [
        "accommodates",
        "availability_365",
        "number_of_reviews",
        "amenities_count",
        "price_per_person",
    ]
    base_cols = [c for c in base_candidates if c in df.columns]
    if not base_cols:
        raise ValueError("No numeric predictors available for pooled regression.")

    work = df.copy()
    work["price"] = pd.to_numeric(work["price"], errors="coerce")
    work = work[work["price"] > 0].copy()

    for col in base_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    work = work.dropna(subset=["price", "city"] + base_cols)
    if "room_type" in work.columns:
        work = work.dropna(subset=["room_type"])

    work["city"] = work["city"].astype(str).str.lower()

    # Fixed-effects pooled model with city and room_type controls.
    rhs = " + ".join(base_cols)
    terms = [rhs, "C(city)"]
    if "room_type" in work.columns:
        terms.append("C(room_type)")

    interaction_cols = [c for c in ["accommodates", "amenities_count"] if c in base_cols]
    for col in interaction_cols:
        terms.append(f"C(city):{col}")

    formula = "np.log(price) ~ " + " + ".join(terms)
    ols_model = smf.ols(formula=formula, data=work).fit()

    ols_summary = pd.DataFrame(
        {
            "term": ols_model.params.index,
            "coefficient": ols_model.params.values,
            "p_value": ols_model.pvalues.values,
        }
    )
    ols_summary["r_squared"] = float(ols_model.rsquared)

    quantile_rows: list[dict[str, Any]] = []
    for q in [0.25, 0.5, 0.75]:
        q_model = smf.quantreg(formula=formula, data=work).fit(q=q)
        q_df = pd.DataFrame(
            {
                "quantile": q,
                "term": q_model.params.index,
                "coefficient": q_model.params.values,
                "p_value": q_model.pvalues.values,
            }
        )
        quantile_rows.extend(q_df.to_dict("records"))

    quantile_summary = pd.DataFrame(quantile_rows)
    return {
        "pooled_fixed_effects": ols_summary,
        "pooled_quantile": quantile_summary,
    }


def run_cross_city_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Run Kruskal-Wallis and pairwise Mann-Whitney tests across cities."""
    work = df.copy()
    work["price"] = pd.to_numeric(work["price"], errors="coerce")
    work = work[work["price"] > 0].copy()
    work["city"] = work["city"].astype(str).str.lower()

    cities = sorted(work["city"].dropna().unique().tolist())
    series_map = {
        city: work.loc[work["city"] == city, "price"].dropna().astype(float)
        for city in cities
    }
    series_map = {k: v for k, v in series_map.items() if len(v) > 0}

    if len(series_map) < 2:
        raise ValueError("Need at least two cities with valid prices for cross-city tests.")

    rows: list[dict[str, Any]] = []

    kw_stat, kw_p = kruskal(*series_map.values())
    rows.append(
        {
            "test": "Kruskal-Wallis (all cities)",
            "group_1": "all",
            "group_2": "all",
            "statistic": float(kw_stat),
            "p_value": float(kw_p),
            "p_value_bh": float(kw_p),
        }
    )

    pair_rows: list[dict[str, Any]] = []
    raw_p: list[float] = []
    for city_a, city_b in combinations(sorted(series_map.keys()), 2):
        stat, p_val = mannwhitneyu(series_map[city_a], series_map[city_b], alternative="two-sided")
        raw_p.append(float(p_val))
        pair_rows.append(
            {
                "test": "Mann-Whitney U (city pair)",
                "group_1": city_a,
                "group_2": city_b,
                "statistic": float(stat),
                "p_value": float(p_val),
            }
        )

    adj = _bh_adjust(raw_p)
    for row, adj_p in zip(pair_rows, adj):
        row["p_value_bh"] = float(adj_p)
        rows.append(row)

    out = pd.DataFrame(rows)
    out["significant_0_05_bh"] = out["p_value_bh"] < 0.05
    return out


def export_multi_city_workbook(tables: dict[str, pd.DataFrame], output_dir: Path) -> Path:
    """Export multi-city summary workbook with one sheet per table."""
    workbook_path = output_dir / "multi_city_analysis_summary.xlsx"
    try:
        with pd.ExcelWriter(workbook_path) as writer:
            for table_name, df in tables.items():
                df.to_excel(writer, sheet_name=table_name[:31], index=False)
        return workbook_path
    except ImportError:
        for table_name, df in tables.items():
            df.to_csv(output_dir / f"{table_name}.csv", index=False)
        return output_dir / "multi_city_analysis_summary.csv"


def run_multi_city_analysis(
    city_dataframes: dict[str, pd.DataFrame],
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run cross-city charts, pooled regressions, and comparative tests."""
    out_dir = output_dir or MULTI_CITY_OUTPUT_DIR
    plots_dir = out_dir / "plots"
    tables_dir = out_dir / "tables"
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")
    combined = _combine_city_frames(city_dataframes)

    plot_paths = [
        plot_city_median_price(combined, output_dir=plots_dir),
        plot_cluster_share_by_city(combined, output_dir=plots_dir),
        plot_city_feature_heatmap(combined, output_dir=plots_dir),
    ]

    regression_tables = run_pooled_regressions(combined)
    tests_table = run_cross_city_tests(combined)

    xgboost_metrics_df = pd.DataFrame()
    xgboost_r2_plot_path: Path | None = None
    xgboost_city_models: dict[str, Any] = {}

    for city_name, city_df in city_dataframes.items():
        try:
            xgb_result = train_xgboost_model(city_df, target="price", random_state=42)
            xgboost_city_models[city_name] = xgb_result
        except (ImportError, KeyError, ValueError):
            # Skip city-level XGBoost if dependency is missing or required fields are absent.
            continue

    if xgboost_city_models:
        xgboost_metrics_df = compare_model_performance(
            xgboost_city_models,
            output_dir=plots_dir,
        )
        xgboost_r2_plot_path = plots_dir / "model_r2_comparison_by_city.png"

    all_tables = {
        "pooled_fixed_effects": regression_tables["pooled_fixed_effects"],
        "pooled_quantile": regression_tables["pooled_quantile"],
        "cross_city_tests": tests_table,
    }
    if not xgboost_metrics_df.empty:
        all_tables["xgboost_city_metrics"] = xgboost_metrics_df

    table_paths: dict[str, Path] = {}
    for name, table in all_tables.items():
        path = tables_dir / f"{name}.csv"
        table.to_csv(path, index=False)
        table_paths[name] = path

    workbook_path = export_multi_city_workbook(all_tables, output_dir=tables_dir)

    return {
        "output_dir": out_dir,
        "plots": plot_paths,
        "tables": table_paths,
        "workbook": workbook_path,
        "xgboost_r2_plot": xgboost_r2_plot_path,
    }
