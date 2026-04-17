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


def _read(path: Path) -> list[dict[str, Any]]:
    try:
        return pd.read_csv(path).to_dict("records")
    except Exception:
        return []


def _load_dashboard_data() -> dict[str, list[dict[str, Any]]]:
    data: dict[str, list[dict[str, Any]]] = {}
    data["r2"] = _read(OUTPUTS / "multi_city/tables/xgboost_city_metrics.csv")
    data["xgb_progression"] = _read(OUTPUTS / "multi_city/tables/xgboost_progression_history.csv")
    data["xgb_tuning"] = _read(OUTPUTS / "multi_city/tables/xgboost_tuning_summary.csv")
    data["shap"] = _read(OUTPUTS / "multi_city/tables/shap_top5_features_by_city.csv")
    data["shap_dominance"] = _read(OUTPUTS / "multi_city/tables/shap_city_dominance_checks.csv")
    data["cross_tests"] = _read(OUTPUTS / "multi_city/tables/cross_city_tests.csv")
    data["host_clusters"] = _read(OUTPUTS / "multi_city/tables/host_strategy_cluster_summary.csv")
    data["cluster_comp"] = _read(OUTPUTS / "tables/pooled_cluster_composition.csv")
    data["calendar"] = _read(OUTPUTS / "tables/calendar_summary.csv")
    data["spatial"] = _read(OUTPUTS / "tables/spatial_summary.csv")
    data["fixed_effects"] = _read(OUTPUTS / "multi_city/tables/pooled_fixed_effects.csv")

    city_price_fallback: dict[str, dict[str, float]] = {}
    try:
        df = pd.read_csv(OUTPUTS / "tables/calendar_temporal_summary.csv")
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
    except Exception:
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

    return data


def _build_city_cards(data: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    for city in CITY_KEYS:
        cal = next((row for row in data.get("calendar", []) if row.get("city") == city), {})
        r2 = next((row for row in data.get("r2", []) if row.get("city") == city), {})
        cards.append(
            {
                "city": city,
                "label": CITY_FOLDERS.get(city, city.title()),
                "color": CITY_COLORS.get(city, "#666666"),
                "median_price": cal.get("median_price"),
                "r2": float(r2.get("r2", 0.0) or 0.0),
                "availability_rate": float(cal.get("availability_rate", 0.0) or 0.0),
            }
        )
    return cards


def _copy_maps() -> None:
    MAPS_DIR.mkdir(parents=True, exist_ok=True)
    for city in CITY_KEYS:
        for suffix in ("price_choropleth", "cluster_map"):
            src = OUTPUTS / "plots" / f"{city}_{suffix}.html"
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
