from __future__ import annotations

from pathlib import Path
import pandas as pd
from flask import Flask, jsonify, render_template, send_file, abort

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent
OUTPUTS  = BASE_DIR / "outputs"


def _read(path: Path) -> list[dict]:
    try:
        return pd.read_csv(path).to_dict("records")
    except Exception:
        return []


def _load() -> dict:
    d: dict = {}
    d["r2"]             = _read(OUTPUTS / "multi_city/tables/xgboost_city_metrics.csv")
    d["shap"]           = _read(OUTPUTS / "multi_city/tables/shap_top5_features_by_city.csv")
    d["shap_dominance"] = _read(OUTPUTS / "multi_city/tables/shap_city_dominance_checks.csv")
    d["cross_tests"]    = _read(OUTPUTS / "multi_city/tables/cross_city_tests.csv")
    d["host_clusters"]  = _read(OUTPUTS / "multi_city/tables/host_strategy_cluster_summary.csv")
    d["cluster_comp"]   = _read(OUTPUTS / "tables/pooled_cluster_composition.csv")
    d["calendar"]       = _read(OUTPUTS / "tables/calendar_summary.csv")
    d["spatial"]        = _read(OUTPUTS / "tables/spatial_summary.csv")
    d["fixed_effects"]  = _read(OUTPUTS / "multi_city/tables/pooled_fixed_effects.csv")
    try:
        df = pd.read_csv(OUTPUTS / "tables/calendar_temporal_summary.csv")
        monthly = (
            df.groupby(["city","month_num","month"])[["avg_price","availability_rate"]]
            .mean().reset_index().sort_values(["city","month_num"])
        )
        d["calendar_monthly"] = monthly.to_dict("records")
    except Exception:
        d["calendar_monthly"] = []
    return d


DATA = _load()


@app.route("/")
def index():
    return render_template("dashboard.html")

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

@app.route("/map/<city>/<map_type>")
def serve_map(city: str, map_type: str):
    if city not in {"newyork","chicago","nashville","neworleans","austin","losangeles"}: abort(404)
    if map_type not in {"choropleth","cluster"}: abort(404)
    suffix = "price_choropleth" if map_type=="choropleth" else "cluster_map"
    path = OUTPUTS / f"plots/{city}_{suffix}.html"
    if not path.exists(): abort(404)
    return send_file(path)

if __name__ == "__main__":
    app.run(debug=True, port=5050)