from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor


BASE_DIR = Path(__file__).resolve().parent.parent
PLOTS_DIR = BASE_DIR / "outputs" / "plots"

CITY_COLORS = {
    "newyork": "#FF5A5F",
    "chicago": "#00A699",
}

BASE_FEATURES = [
    "accommodates",
    "amenities_count",
    "availability_365",
    "reviews_per_month",
    "minimum_nights",
]


def _clean_city_frame(df: pd.DataFrame, city: str) -> pd.DataFrame:
    """Return city-specific rows with required columns and valid numeric values."""
    required = BASE_FEATURES + ["log_price", "room_type", "neighbourhood"]
    missing = [col for col in required + ["city"] if col not in df.columns]
    if missing:
        raise KeyError(f"Missing columns for regression: {missing}")

    city_df = df[df["city"].astype(str).str.lower() == city.lower()].copy()
    if city_df.empty:
        raise ValueError(f"No rows found for city={city}")

    for col in BASE_FEATURES + ["log_price"]:
        city_df[col] = pd.to_numeric(city_df[col], errors="coerce")

    city_df = city_df.dropna(subset=required)
    city_df = city_df[city_df["log_price"].notna()].copy()
    if city_df.empty:
        raise ValueError(f"No valid rows after cleaning for city={city}")

    return city_df


def _build_model_matrix(city_df: pd.DataFrame, top_n_neighbourhoods: int = 10) -> tuple[pd.DataFrame, pd.Series]:
    """Build OLS/RF design matrix with room and top-neighbourhood dummies."""
    top_neighbourhoods = (
        city_df["neighbourhood"].value_counts().head(top_n_neighbourhoods).index.tolist()
    )

    model_df = city_df.copy()
    model_df["neighbourhood_top"] = model_df["neighbourhood"].where(
        model_df["neighbourhood"].isin(top_neighbourhoods),
        other="OTHER",
    )

    X_base = model_df[BASE_FEATURES].copy()
    room_dummies = pd.get_dummies(model_df["room_type"], prefix="room", drop_first=True)
    neigh_dummies = pd.get_dummies(
        model_df["neighbourhood_top"],
        prefix="neigh",
        drop_first=True,
    )

    X = pd.concat([X_base, room_dummies, neigh_dummies], axis=1)
    X = X.astype(float)
    y = model_df["log_price"].astype(float)
    return X, y


def run_city_regression(
    df: pd.DataFrame,
    city: str,
    top_n_neighbourhoods: int = 10,
    random_state: int = 42,
) -> dict[str, Any]:
    """Run OLS and Random Forest regression for one city and return model artifacts."""
    city_df = _clean_city_frame(df, city=city)
    X, y = _build_model_matrix(city_df, top_n_neighbourhoods=top_n_neighbourhoods)

    X_const = sm.add_constant(X)
    ols_model = sm.OLS(y, X_const).fit()

    ols_summary = pd.DataFrame(
        {
            "feature": ols_model.params.index,
            "coefficient": ols_model.params.values,
            "p_value": ols_model.pvalues.values,
        }
    )
    ols_summary["r_squared"] = ols_model.rsquared

    rf = RandomForestRegressor(
        n_estimators=400,
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X, y)
    rf_importance = (
        pd.DataFrame({"feature": X.columns, "importance": rf.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "city": city.lower(),
        "ols_model": ols_model,
        "ols_summary": ols_summary,
        "r_squared": float(ols_model.rsquared),
        "rf_model": rf,
        "rf_importance": rf_importance,
        "n_rows": int(len(X)),
    }


def plot_rf_feature_importances(
    results_by_city: dict[str, dict[str, Any]],
    output_dir: Path | None = None,
) -> Path:
    """Plot Random Forest feature importances as horizontal bars by city."""
    out_dir = output_dir or PLOTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    cities = [city for city in ["newyork", "chicago"] if city in results_by_city]
    if not cities:
        raise ValueError("No city results were provided for plotting.")

    fig, axes = plt.subplots(1, len(cities), figsize=(8 * len(cities), 6), sharex=False)
    if len(cities) == 1:
        axes = [axes]

    for ax, city in zip(axes, cities):
        imp = results_by_city[city]["rf_importance"].head(15).copy()
        imp = imp.sort_values("importance", ascending=True)

        sns.barplot(
            data=imp,
            x="importance",
            y="feature",
            color=CITY_COLORS.get(city, "#333333"),
            ax=ax,
        )
        ax.set_title(f"{city.title()} Random Forest Feature Importance")
        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")

    out_path = out_dir / "regression_rf_feature_importance_by_city.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def compare_top_feature_importance(
    results_by_city: dict[str, dict[str, Any]],
    top_n: int = 5,
) -> pd.DataFrame:
    """Build a side-by-side top-N feature-importance comparison table."""
    rows: list[dict[str, Any]] = []
    for city, result in results_by_city.items():
        top = result["rf_importance"].head(top_n).reset_index(drop=True)
        for rank, row in top.iterrows():
            rows.append(
                {
                    "city": city,
                    "rank": rank + 1,
                    "feature": row["feature"],
                    "importance": row["importance"],
                }
            )

    table = pd.DataFrame(rows)
    if table.empty:
        return table

    pivot = table.pivot(index="rank", columns="city", values="feature")
    pivot.columns = [f"{col}_feature" for col in pivot.columns]

    imp_pivot = table.pivot(index="rank", columns="city", values="importance")
    imp_pivot.columns = [f"{col}_importance" for col in imp_pivot.columns]

    comparison = pd.concat([pivot, imp_pivot], axis=1).reset_index()
    return comparison.sort_values("rank").reset_index(drop=True)


def get_clean_ols_summary(ols_summary: pd.DataFrame) -> pd.DataFrame:
    """Return filtered, presentation-ready OLS summary excluding const and neighbourhood dummies."""
    required_cols = {"feature", "coefficient", "p_value", "r_squared"}
    missing = sorted(required_cols - set(ols_summary.columns))
    if missing:
        raise KeyError(f"Missing required OLS summary columns: {missing}")

    clean = ols_summary.copy()
    clean = clean[clean["feature"] != "const"]
    clean = clean[~clean["feature"].astype(str).str.startswith("neigh_")]

    room_mask = clean["feature"].astype(str).str.startswith("room_")
    base_mask = clean["feature"].isin(BASE_FEATURES)
    clean = clean[room_mask | base_mask].copy()

    clean = clean.rename(
        columns={
            "feature": "Feature",
            "coefficient": "Coefficient",
            "p_value": "P-Value",
            "r_squared": "R²",
        }
    )

    clean["Coefficient"] = clean["Coefficient"].round(4)
    clean["P-Value"] = clean["P-Value"].round(4)
    clean["R²"] = clean["R²"].round(4)
    clean["Significant"] = clean["P-Value"] < 0.05

    order = ["Feature", "Coefficient", "P-Value", "R²", "Significant"]
    return clean[order].reset_index(drop=True)


def run_regression_analysis(
    df: pd.DataFrame,
    output_dir: Path | None = None,
    top_n_neighbourhoods: int = 10,
    random_state: int = 42,
) -> dict[str, Any]:
    """Run full regression workflow for New York and Chicago and print clean summaries."""
    sns.set_theme(style="whitegrid")

    results: dict[str, dict[str, Any]] = {}
    for city in ["newyork", "chicago"]:
        results[city] = run_city_regression(
            df,
            city=city,
            top_n_neighbourhoods=top_n_neighbourhoods,
            random_state=random_state,
        )

    print("\n[regression] OLS Summary Tables")
    for city in ["newyork", "chicago"]:
        city_summary = results[city]["ols_summary"].copy()
        clean_summary = get_clean_ols_summary(city_summary)
        print(f"\n[regression] City: {city.title()} | R-squared: {results[city]['r_squared']:.4f}")
        print(clean_summary)

    importance_plot = plot_rf_feature_importances(results, output_dir=output_dir)
    comparison = compare_top_feature_importance(results, top_n=5)

    print("\n[regression] Top 5 Feature Importance Comparison (Side-by-Side)")
    print(comparison)

    return {
        "city_results": results,
        "rf_importance_plot": importance_plot,
        "top_feature_comparison": comparison,
    }
