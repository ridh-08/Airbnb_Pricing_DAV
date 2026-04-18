from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from src.config import CITY_COLORS, CITY_FOLDERS

BASE_DIR = Path(__file__).resolve().parent
OUTPUTS = BASE_DIR / "outputs"
NETLIFY_DIR = BASE_DIR / "netlify_static"
MAPS_DIR = NETLIFY_DIR / "maps"

CITY_KEYS = ["newyork", "chicago", "nashville", "neworleans", "austin", "losangeles"]
DATA_QUALITY_ISSUES: dict[str, str] = {
    "losangeles": (
        "Processed host-level data contains only 3 distinct price values. "
        "XGBoost metrics and live predictor are unavailable for LA until the "
        "upstream snapshot is re-ingested."
    ),
}


def _read(path: Path) -> list[dict[str, Any]]:
    try:
        return pd.read_csv(path).to_dict("records")
    except Exception:
        return []


def _read_city_table_bundle(filename_template: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for city in CITY_KEYS:
        path = OUTPUTS / city / "tables" / filename_template.format(city=city)
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
            if "city" not in df.columns:
                df["city"] = city
            rows.extend(df.to_dict("records"))
        except Exception:
            continue
    return rows


def _coerce_wide_shap_table(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not records:
        return []
    first = records[0]
    if "city" in first and "feature" in first:
        return records
    if "rank" not in first:
        return []

    rows: list[dict[str, Any]] = []
    for rec in records:
        rank = int(rec.get("rank", 0) or 0)
        for city in CITY_KEYS:
            feature = rec.get(city)
            if not feature:
                continue
            rows.append(
                {
                    "city": city,
                    "feature": str(feature),
                    "mean_abs_shap": float(max(1, 6 - rank)),
                    "rank": rank,
                }
            )
    return rows


def _load_shap_from_city_files() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    shap_rows: list[dict[str, Any]] = []
    dominance_rows: list[dict[str, Any]] = []

    for city in CITY_KEYS:
        path = OUTPUTS / city / "tables" / f"{city}_xgboost_shap_summary.csv"
        if not path.exists():
            continue

        try:
            df = pd.read_csv(path)
        except Exception:
            continue

        if "feature" not in df.columns:
            continue

        if "mean_abs_shap" not in df.columns:
            candidate = [c for c in df.columns if "shap" in c.lower()]
            if candidate:
                df = df.rename(columns={candidate[0]: "mean_abs_shap"})

        if "mean_abs_shap" not in df.columns:
            continue

        top = df.sort_values("mean_abs_shap", ascending=False).head(5).reset_index(drop=True)
        top["rank"] = top.index + 1

        for _, row in top.iterrows():
            shap_rows.append(
                {
                    "city": city,
                    "feature": str(row["feature"]),
                    "mean_abs_shap": float(row["mean_abs_shap"]),
                    "rank": int(row["rank"]),
                }
            )

        total = float(top["mean_abs_shap"].sum()) if len(top) else 0.0
        if total <= 0:
            continue

        amen = top[top["feature"] == "amenities_count"]
        neigh = top[top["feature"].astype(str).str.startswith("neigh_")]
        top_row = top.iloc[0]
        amen_val = float(amen.iloc[0]["mean_abs_shap"]) if not amen.empty else 0.0
        amen_rank = int(amen.iloc[0]["rank"]) if not amen.empty else None
        neigh_val = float(neigh["mean_abs_shap"].sum()) if not neigh.empty else 0.0

        dominance_rows.append(
            {
                "city": city,
                "top_feature": str(top_row["feature"]),
                "top_value": float(top_row["mean_abs_shap"]),
                "amenities_value": amen_val,
                "amenities_rank": amen_rank,
                "amenities_share_pct": 100.0 * amen_val / total,
                "neighbourhood_share_pct": 100.0 * neigh_val / total,
            }
        )

    return shap_rows, dominance_rows


def _load_dashboard_data() -> dict[str, list[dict[str, Any]]]:
    data: dict[str, list[dict[str, Any]]] = {}
    data["r2"] = _read(OUTPUTS / "multi_city/tables/xgboost_city_metrics.csv")
    data["xgb_progression"] = _read(OUTPUTS / "multi_city/tables/xgboost_progression_history.csv")
    data["xgb_tuning"] = _read(OUTPUTS / "multi_city/tables/xgboost_tuning_summary.csv")
    data["shap"] = _coerce_wide_shap_table(_read(OUTPUTS / "multi_city/tables/shap_top5_features_by_city.csv"))
    data["shap_dominance"] = _read(OUTPUTS / "multi_city/tables/shap_city_dominance_checks.csv")
    data["cross_tests"] = _read(OUTPUTS / "multi_city/tables/cross_city_tests.csv")
    data["host_clusters"] = _read(OUTPUTS / "multi_city/tables/host_strategy_cluster_summary.csv")
    data["host_silhouette"] = _read(OUTPUTS / "multi_city/tables/host_strategy_silhouette_scan.csv")
    data["cluster_comp"] = _read(OUTPUTS / "tables/pooled_cluster_composition.csv")
    data["calendar"] = _read(OUTPUTS / "tables/calendar_summary.csv")
    data["spatial"] = _read(OUTPUTS / "tables/spatial_summary.csv")
    data["fixed_effects"] = _read(OUTPUTS / "multi_city/tables/pooled_fixed_effects.csv")

    if not data["calendar"]:
        data["calendar"] = _read_city_table_bundle("calendar_summary_{city}.csv")
    if not data["spatial"]:
        data["spatial"] = _read_city_table_bundle("spatial_summary_{city}.csv")
    if not data["shap"]:
        shap_rows, shap_dom = _load_shap_from_city_files()
        data["shap"] = shap_rows
        if not data["shap_dominance"]:
            data["shap_dominance"] = shap_dom

    city_price_fallback: dict[str, dict[str, float]] = {}
    try:
        df = pd.read_csv(OUTPUTS / "tables/calendar_temporal_summary.csv")
    except Exception:
        fallback_temporal = _read_city_table_bundle("calendar_temporal_summary_{city}.csv")
        df = pd.DataFrame(fallback_temporal) if fallback_temporal else pd.DataFrame()

    if not df.empty and {"city", "month_num", "month", "avg_price", "availability_rate"}.issubset(df.columns):
        monthly = (
            df.groupby(["city", "month_num", "month"])[["avg_price", "availability_rate"]]
            .mean()
            .reset_index()
            .sort_values(["city", "month_num"])
        )
        data["calendar_monthly"] = monthly.to_dict("records")

        fallback_df = (
            df.groupby("city", as_index=False)["avg_price"]
            .agg(avg_price="mean", median_price="median")
        )
        for row in fallback_df.to_dict("records"):
            city = str(row.get("city", "")).strip()
            if not city:
                continue
            city_price_fallback[city] = {
                "avg_price": float(row.get("avg_price", 0.0) or 0.0),
                "median_price": float(row.get("median_price", 0.0) or 0.0),
            }
    else:
        data["calendar_monthly"] = []

    for row in data.get("calendar", []):
        city = str(row.get("city", "")).strip()
        if city not in city_price_fallback:
            continue

        avg_price = pd.to_numeric(row.get("avg_price"), errors="coerce")
        median_price = pd.to_numeric(row.get("median_price"), errors="coerce")

        if pd.isna(avg_price):
            row["avg_price"] = round(city_price_fallback[city]["avg_price"], 2)
        if pd.isna(median_price):
            row["median_price"] = round(city_price_fallback[city]["median_price"], 2)

    # Keep city price KPIs aligned with modeling inputs by taking price center from
    # each city's featured dataset when available.
    featured_price_stats: dict[str, dict[str, float]] = {}
    for city in CITY_KEYS:
        featured_path = BASE_DIR / "data" / "processed" / f"{city}_featured.csv"
        if not featured_path.exists():
            continue
        try:
            fdf = pd.read_csv(featured_path, usecols=["price"])
            fdf["price"] = pd.to_numeric(fdf["price"], errors="coerce")
            fdf = fdf[fdf["price"] > 0]
            if fdf.empty:
                continue
            featured_price_stats[city] = {
                "avg_price": float(fdf["price"].mean()),
                "median_price": float(fdf["price"].median()),
            }
        except Exception:
            continue

    for row in data.get("calendar", []):
        city = str(row.get("city", "")).strip()
        stats = featured_price_stats.get(city)
        if not stats:
            continue
        row["avg_price"] = round(stats["avg_price"], 2)
        row["median_price"] = round(stats["median_price"], 2)

    return data


def _build_city_cards(data: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    for city in CITY_KEYS:
        cal = next((r for r in data.get("calendar", []) if r.get("city") == city), {})
        r2 = next((r for r in data.get("r2", []) if r.get("city") == city), {})
        issue = DATA_QUALITY_ISSUES.get(city)
        r2_value = float(r2.get("r2", 0.0) or 0.0)
        cards.append(
            {
                "city": city,
                "label": CITY_FOLDERS.get(city, city.title()),
                "color": CITY_COLORS.get(city, "#666666"),
                "median_price": cal.get("median_price"),
                "r2": None if issue else r2_value,
                "availability_rate": float(cal.get("availability_rate", 0.0) or 0.0),
                "data_quality_issue": issue,
                "has_predictor": issue is None,
            }
        )
    return cards


def _copy_maps() -> None:
    MAPS_DIR.mkdir(parents=True, exist_ok=True)
    for city in CITY_KEYS:
        for suffix in ("price_choropleth", "cluster_map"):
            src_old = OUTPUTS / "plots" / f"{city}_{suffix}.html"
            src_new = OUTPUTS / city / "plots" / f"{city}_{suffix}.html"
            src = src_new if src_new.exists() else src_old
            if src.exists():
                shutil.copy2(src, MAPS_DIR / src.name)


def export_dashboard() -> None:
    NETLIFY_DIR.mkdir(parents=True, exist_ok=True)
    api_base_url = os.getenv("API_BASE_URL", "").strip().rstrip("/")

    data = _load_dashboard_data()
    city_cards = _build_city_cards(data)

    env = Environment(
        loader=FileSystemLoader(str(BASE_DIR / "templates")),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = env.get_template("dashboard.html")
    rendered = template.render(
        initial_data=data,
        initial_predictor_meta={},
        city_cards=city_cards,
        has_api=bool(api_base_url),
        api_base_url=api_base_url,
    )

    (NETLIFY_DIR / "index.html").write_text(rendered, encoding="utf-8")
    _copy_maps()

    print(f"Static dashboard exported to: {NETLIFY_DIR}")


if __name__ == "__main__":
    export_dashboard()
