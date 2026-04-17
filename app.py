from __future__ import annotations

import os
from pathlib import Path
from typing import Any
import pandas as pd
from flask import Flask, jsonify, render_template, send_file, abort, request
from flask_cors import CORS
from sklearn.model_selection import train_test_split

from src.config import CITY_COLORS, CITY_FOLDERS

app = Flask(__name__)
CORS(app)
BASE_DIR = Path(__file__).resolve().parent
OUTPUTS  = BASE_DIR / "outputs"
PROCESSED = BASE_DIR / "data" / "processed"
API_BASE_URL = os.getenv("API_BASE_URL", "").strip().rstrip("/")

CITY_KEYS = ["newyork", "chicago", "nashville", "neworleans", "austin", "losangeles"]


def _normalize_city_key(value: str) -> str:
    return str(value).strip().lower().replace(" ", "").replace("_", "").replace("-", "")


def _read(path: Path) -> list[dict]:
    try:
        return pd.read_csv(path).to_dict("records")
    except Exception:
        return []


def _load() -> dict:
    d: dict = {}
    d["r2"]             = _read(OUTPUTS / "multi_city/tables/xgboost_city_metrics.csv")
    d["xgb_progression"] = _read(OUTPUTS / "multi_city/tables/xgboost_progression_history.csv")
    d["xgb_tuning"]      = _read(OUTPUTS / "multi_city/tables/xgboost_tuning_summary.csv")
    d["shap"]           = _read(OUTPUTS / "multi_city/tables/shap_top5_features_by_city.csv")
    d["shap_dominance"] = _read(OUTPUTS / "multi_city/tables/shap_city_dominance_checks.csv")
    d["cross_tests"]    = _read(OUTPUTS / "multi_city/tables/cross_city_tests.csv")
    d["host_clusters"]  = _read(OUTPUTS / "multi_city/tables/host_strategy_cluster_summary.csv")
    d["cluster_comp"]   = _read(OUTPUTS / "tables/pooled_cluster_composition.csv")
    d["calendar"]       = _read(OUTPUTS / "tables/calendar_summary.csv")
    d["spatial"]        = _read(OUTPUTS / "tables/spatial_summary.csv")
    d["fixed_effects"]  = _read(OUTPUTS / "multi_city/tables/pooled_fixed_effects.csv")
    city_price_fallback: dict[str, dict[str, float]] = {}
    try:
        df = pd.read_csv(OUTPUTS / "tables/calendar_temporal_summary.csv")
        monthly = (
            df.groupby(["city","month_num","month"])[["avg_price","availability_rate"]]
            .mean().reset_index().sort_values(["city","month_num"])
        )
        d["calendar_monthly"] = monthly.to_dict("records")

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
        d["calendar_monthly"] = []

    for row in d.get("calendar", []):
        city = str(row.get("city", "")).strip()
        if city not in city_price_fallback:
            continue

        avg_price = pd.to_numeric(row.get("avg_price"), errors="coerce")
        median_price = pd.to_numeric(row.get("median_price"), errors="coerce")

        if pd.isna(avg_price):
            row["avg_price"] = round(city_price_fallback[city]["avg_price"], 2)
        if pd.isna(median_price):
            row["median_price"] = round(city_price_fallback[city]["median_price"], 2)

    return d


def _build_city_cards() -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    for city in CITY_KEYS:
        cal = next((row for row in DATA.get("calendar", []) if row.get("city") == city), {})
        r2 = next((row for row in DATA.get("r2", []) if row.get("city") == city), {})
        cards.append(
            {
                "city": city,
                "label": CITY_FOLDERS.get(city, city.replace("newyork", "New York").title()),
                "color": CITY_COLORS.get(city, "#666666"),
                "median_price": cal.get("median_price"),
                "r2": float(r2.get("r2", 0.0) or 0.0),
                "availability_rate": float(cal.get("availability_rate", 0.0) or 0.0),
            }
        )
    return cards


def _build_predictor_meta() -> dict[str, dict[str, Any]]:
    meta: dict[str, dict[str, Any]] = {}
    for city, model_info in PREDICTORS.items():
        meta[city] = {
            "ranges": model_info.get("ranges", {}),
            "top_neighbourhoods": model_info.get("top_neighbourhoods", []),
            "default_neighbourhood": model_info.get("default_neighbourhood", "OTHER"),
            "sample_size": model_info.get("sample_size", 0),
        }
    return meta


def _build_predictor_cache() -> dict[str, dict[str, Any]]:
    """Train lightweight city-level XGBoost models for interactive dashboard scenarios."""
    try:
        from xgboost import XGBRegressor
    except Exception:
        return {}

    cache: dict[str, dict[str, Any]] = {}
    numeric_features = ["accommodates", "amenities_count", "availability_365", "review_density"]

    for city in CITY_KEYS:
        path = PROCESSED / f"{city}_featured.csv"
        if not path.exists():
            continue

        try:
            df = pd.read_csv(path)
        except Exception:
            continue

        required = numeric_features + ["price", "neighbourhood"]
        if any(col not in df.columns for col in required):
            continue

        work = df.copy()
        for col in numeric_features + ["price"]:
            work[col] = pd.to_numeric(work[col], errors="coerce")
        work["neighbourhood"] = work["neighbourhood"].astype(str)
        work = work.dropna(subset=required)
        work = work[(work["price"] > 0) & (work["price"] < work["price"].quantile(0.995))]
        if len(work) < 200:
            continue

        # Keep top neighbourhoods to make the live model stable and fast.
        top_n = 25
        top_neighs = work["neighbourhood"].value_counts().head(top_n).index.tolist()
        work["neighbourhood_model"] = work["neighbourhood"].where(
            work["neighbourhood"].isin(top_neighs),
            "OTHER",
        )

        X_num = work[numeric_features].copy()
        X_neigh = pd.get_dummies(work["neighbourhood_model"], prefix="neigh")
        X = pd.concat([X_num, X_neigh], axis=1).astype(float)
        y = work["price"].astype(float)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
        )

        model = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=250,
            learning_rate=0.06,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=2,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        rmse = float(((y_test - y_pred) ** 2).mean() ** 0.5)

        ranges = {
            col: {
                "min": float(work[col].quantile(0.05)),
                "max": float(work[col].quantile(0.95)),
                "default": float(work[col].median()),
            }
            for col in numeric_features
        }

        cache[city] = {
            "model": model,
            "feature_columns": X.columns.tolist(),
            "top_neighbourhoods": sorted(top_neighs),
            "default_neighbourhood": top_neighs[0] if top_neighs else "OTHER",
            "ranges": ranges,
            "rmse": rmse,
            "price_floor": float(work["price"].quantile(0.01)),
            "price_cap": float(work["price"].quantile(0.99)),
            "sample_size": int(len(work)),
        }

    return cache


def _predict_price(city: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    model_info = PREDICTORS.get(city)
    if not model_info:
        return None

    ranges = model_info["ranges"]
    row = {
        "accommodates": float(payload.get("accommodates", ranges["accommodates"]["default"])),
        "amenities_count": float(payload.get("amenities_count", ranges["amenities_count"]["default"])),
        "availability_365": float(payload.get("availability_365", ranges["availability_365"]["default"])),
        "review_density": float(payload.get("review_density", ranges["review_density"]["default"])),
    }

    # Keep the UI predictor inside realistic training bounds.
    for col, val in row.items():
        row[col] = max(ranges[col]["min"], min(ranges[col]["max"], float(val)))

    neighbourhood = str(payload.get("neighbourhood", model_info["default_neighbourhood"]))
    if neighbourhood not in model_info["top_neighbourhoods"]:
        neighbourhood = "OTHER"

    X_pred = pd.DataFrame([row])
    for col in model_info["feature_columns"]:
        if col.startswith("neigh_"):
            label = col.replace("neigh_", "")
            X_pred[col] = 1.0 if label == neighbourhood else 0.0

    X_pred = X_pred.reindex(columns=model_info["feature_columns"], fill_value=0.0)
    pred = float(model_info["model"].predict(X_pred)[0])

    low = max(0.0, pred - model_info["rmse"])
    high = pred + model_info["rmse"]

    return {
        "city": city,
        "prediction": round(pred, 2),
        "band_low": round(low, 2),
        "band_high": round(high, 2),
        "rmse": round(float(model_info["rmse"]), 2),
        "sample_size": model_info["sample_size"],
        "used_neighbourhood": neighbourhood,
    }


def _predict_sweep(city: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    """Return a one-feature sensitivity curve for dashboard what-if analysis."""
    model_info = PREDICTORS.get(city)
    if not model_info:
        return None

    feature = str(payload.get("feature", "accommodates"))
    if feature not in model_info["ranges"]:
        return None

    points = int(payload.get("points", 12))
    points = max(6, min(points, 30))

    feature_min = float(model_info["ranges"][feature]["min"])
    feature_max = float(model_info["ranges"][feature]["max"])
    if feature_max <= feature_min:
        return None

    step = (feature_max - feature_min) / (points - 1)
    values = [feature_min + i * step for i in range(points)]

    preds: list[float] = []
    for val in values:
        scenario = dict(payload)
        scenario[feature] = val
        out = _predict_price(city, scenario)
        if out is None:
            return None
        preds.append(float(out["prediction"]))

    return {
        "city": city,
        "feature": feature,
        "x": [round(v, 4) for v in values],
        "y": [round(v, 2) for v in preds],
    }


DATA = _load()
PREDICTORS = _build_predictor_cache()


@app.route("/")
def index():
    return render_template(
        "dashboard.html",
        initial_data=DATA,
        initial_predictor_meta=_build_predictor_meta(),
        city_cards=_build_city_cards(),
        has_api=True,
        api_base_url=API_BASE_URL,
    )


@app.route("/api/health")
def api_health():
    return jsonify({"status": "ok", "cities_with_models": sorted(PREDICTORS.keys())})

@app.route("/api/data")
def api_data():
    return jsonify(DATA)

@app.route("/api/city/<city>")
def api_city(city: str):
    stats: dict = {"city": city}
    r2  = next((d for d in DATA["r2"]      if d["city"]==city), None)
    cal = next((d for d in DATA["calendar"] if d["city"]==city), None)
    sp  = next((d for d in DATA["spatial"]  if d["city"]==city), None)
    sh1 = next((d for d in DATA["shap"]     if d["city"]==city and d["rank"]==1), None)
    if r2:  stats["r2"] = round(r2["r2"],4)
    if cal: stats.update({"avg_price":cal.get("avg_price"),"median_price":cal.get("median_price"),
                           "availability_pct":round(cal.get("availability_rate",0)*100,1),
                           "unique_listings":cal.get("unique_listings")})
    if sh1: stats["top_feature"] = sh1["feature"].replace("neigh_","")
    if sp:  stats.update({"top_neighbourhood":sp.get("top_priced_neighbourhood"),
                           "top_neigh_price":round(sp.get("top_avg_price",0)),
                           "morans_i":round(sp.get("morans_i",0),3),
                           "coverage":round(sp.get("coverage_rate",0)*100,1)})
    return jsonify(stats)


@app.route("/api/predict/meta")
def api_predict_meta():
    payload: dict[str, Any] = {}
    for city, model_info in PREDICTORS.items():
        payload[city] = {
            "ranges": model_info["ranges"],
            "top_neighbourhoods": model_info["top_neighbourhoods"],
            "default_neighbourhood": model_info["default_neighbourhood"],
            "sample_size": model_info["sample_size"],
        }
    return jsonify(payload)


@app.route("/api/predict/<city>", methods=["POST"])
def api_predict_city(city: str):
    if city not in PREDICTORS:
        return jsonify({"error": "City model unavailable"}), 404

    payload = request.get_json(silent=True) or {}
    result = _predict_price(city, payload)
    if result is None:
        return jsonify({"error": "Prediction failed"}), 400
    return jsonify(result)


@app.route("/api/predict/sweep/<city>", methods=["POST"])
def api_predict_sweep(city: str):
    if city not in PREDICTORS:
        return jsonify({"error": "City model unavailable"}), 404

    payload = request.get_json(silent=True) or {}
    result = _predict_sweep(city, payload)
    if result is None:
        return jsonify({"error": "Sweep failed"}), 400
    return jsonify(result)

@app.route("/map/<city>/<map_type>")
def serve_map(city: str, map_type: str):
    city_key = _normalize_city_key(city)
    if city_key not in set(CITY_KEYS):
        abort(404)
    if map_type not in {"choropleth","cluster"}: abort(404)
    suffix = "price_choropleth" if map_type=="choropleth" else "cluster_map"
    path = OUTPUTS / f"plots/{city_key}_{suffix}.html"
    if not path.exists(): abort(404)
    return send_file(path)

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5050"))
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug, host="0.0.0.0", port=port)