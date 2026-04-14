from __future__ import annotations

from pathlib import Path
from typing import Any
import importlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.config import CITY_COLORS, PLOTS_DIR, normalize_city_name


BASE_DIR = Path(__file__).resolve().parent.parent

BASE_FEATURES = [
    "accommodates",
    "amenities_count",
    "availability_365",
    "reviews_per_month",
    "minimum_nights",
]


def train_xgboost_model(
    df: pd.DataFrame,
    target: str = "price",
    random_state: int = 42,
) -> dict[str, Any]:
    """Train an XGBoost regressor using selected listing features.

    Steps implemented:
    - Uses features: accommodates, amenities_count, availability_365, review_density,
      neighbourhood (one-hot encoded).
    - Applies an 80-20 train-test split.
    - Trains XGBoost regressor with deterministic random_state.
    - Returns model, predictions, and evaluation metrics (R2, RMSE).
    """
    try:
        xgboost_module = importlib.import_module("xgboost")
        XGBRegressor = getattr(xgboost_module, "XGBRegressor")
    except Exception as exc:
        raise ImportError(
            "xgboost is required for train_xgboost_model(). Install with: pip install xgboost"
        ) from exc

    required = ["accommodates", "amenities_count", "availability_365", "review_density", "neighbourhood", target]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for XGBoost training: {missing}")

    work = df.copy()
    numeric_cols = ["accommodates", "amenities_count", "availability_365", "review_density", target]
    for col in numeric_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    work["neighbourhood"] = work["neighbourhood"].astype(str)
    work = work.dropna(subset=required)

    if work.empty:
        raise ValueError("No valid rows available after cleaning for XGBoost training.")

    X_num = work[["accommodates", "amenities_count", "availability_365", "review_density"]].copy()
    X_neigh = pd.get_dummies(work["neighbourhood"], prefix="neigh", drop_first=True)
    X = pd.concat([X_num, X_neigh], axis=1).astype(float)
    y = work[target].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
    )

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    r2 = float(r2_score(y_test, predictions))
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))

    return {
        "model": model,
        "predictions": predictions,
        "metrics": {
            "r2": r2,
            "rmse": rmse,
        },
        "X_test": X_test,
        "y_test": y_test,
    }


def compute_shap_feature_importance(
    model: Any,
    X: pd.DataFrame,
) -> pd.DataFrame:
    """Compute SHAP values for a trained tree model and return feature-importance summary.

    Uses shap.TreeExplainer and ranks features by mean absolute SHAP value.
    """
    try:
        shap_module = importlib.import_module("shap")
        TreeExplainer = getattr(shap_module, "TreeExplainer")
    except Exception as exc:
        raise ImportError(
            "shap is required for compute_shap_feature_importance(). Install with: pip install shap"
        ) from exc

    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame with feature columns.")
    if X.empty:
        raise ValueError("X is empty; SHAP values cannot be computed.")

    explainer = TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap_array = np.asarray(shap_values)

    # Handle both (n_samples, n_features) and multi-output shapes.
    if shap_array.ndim == 3:
        shap_array = np.mean(np.abs(shap_array), axis=0)
    elif shap_array.ndim == 2:
        shap_array = np.abs(shap_array)
    else:
        raise ValueError(f"Unexpected SHAP values shape: {shap_array.shape}")

    summary_df = pd.DataFrame(
        {
            "feature": X.columns,
            "mean_abs_shap": shap_array.mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)

    return summary_df.reset_index(drop=True)


def save_shap_beeswarm_plot(
    model: Any,
    X: pd.DataFrame,
    city_name: str,
    output_dir: Path,
    top_n: int = 10,
) -> Path:
    """Generate and save a SHAP beeswarm plot for a trained tree model.

    The beeswarm highlights top_n features and saves a clean labeled PNG.
    """
    try:
        shap_module = importlib.import_module("shap")
        TreeExplainer = getattr(shap_module, "TreeExplainer")
        summary_plot = getattr(shap_module, "summary_plot")
    except Exception as exc:
        raise ImportError(
            "shap is required for save_shap_beeswarm_plot(). Install with: pip install shap"
        ) from exc

    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame with feature columns.")
    if X.empty:
        raise ValueError("X is empty; SHAP beeswarm plot cannot be generated.")

    city_key = normalize_city_name(city_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    explainer = TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    plt.figure(figsize=(11, 6.5))
    summary_plot(
        shap_values,
        X,
        plot_type="dot",
        max_display=top_n,
        show=False,
    )
    plt.title(f"{city_key.title()} XGBoost SHAP Beeswarm (Top {top_n} Features)", pad=12)
    plt.xlabel("SHAP value (impact on model output)")
    plt.tight_layout()

    out_path = output_dir / f"{city_key}_xgboost_shap_beeswarm_top{top_n}.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    return out_path


def compare_model_performance(
    city_models_dict: dict[str, Any],
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Compare model performance across cities using R2 and plot a bar chart.

    Parameters
    ----------
    city_models_dict:
        Mapping city -> trained model payload. Supported shapes per city value:
        - {"metrics": {"r2": <float>}}
        - {"r2": <float>}
        - <float> (direct R2)
    output_dir:
        Optional directory to save chart. Defaults to outputs/plots.

    Returns
    -------
    pd.DataFrame
        Dataframe with columns: city, r2 sorted by descending r2.
    """
    rows: list[dict[str, Any]] = []

    for city, payload in city_models_dict.items():
        r2_value: float | None = None

        if isinstance(payload, (float, int)):
            r2_value = float(payload)
        elif isinstance(payload, dict):
            if "metrics" in payload and isinstance(payload["metrics"], dict):
                metric_r2 = payload["metrics"].get("r2")
                if metric_r2 is not None:
                    r2_value = float(metric_r2)
            elif "r2" in payload:
                r2_value = float(payload["r2"])

        if r2_value is None:
            raise KeyError(
                f"Could not extract R2 for city '{city}'. Expected float, {{'r2': ...}}, or {{'metrics': {{'r2': ...}}}}."
            )

        rows.append({"city": str(city).lower(), "r2": r2_value})

    comparison_df = pd.DataFrame(rows)
    if comparison_df.empty:
        raise ValueError("No city model performance entries provided.")

    comparison_df = comparison_df.sort_values("r2", ascending=False).reset_index(drop=True)

    out_dir = output_dir or PLOTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=comparison_df,
        x="city",
        y="r2",
        hue="city",
        legend=False,
        palette=[CITY_COLORS.get(c, "#555555") for c in comparison_df["city"]],
    )
    ax.set_title("Model R2 Comparison Across Cities (Higher R2 = More Predictable Market)")
    ax.set_xlabel("City")
    ax.set_ylabel("R2")
    ax.set_ylim(bottom=min(-0.05, float(comparison_df["r2"].min()) - 0.05), top=1.0)
    plt.xticks(rotation=15)

    out_path = out_dir / "model_r2_comparison_by_city.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

    return comparison_df


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

    cities = sorted(results_by_city.keys())
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
    city_name: str | None = None,
) -> dict[str, Any]:
    """Run full regression workflow for all cities or one selected city."""
    sns.set_theme(style="whitegrid")

    results: dict[str, dict[str, Any]] = {}
    work = df.copy()
    work["city"] = work["city"].astype(str).str.lower()
    if city_name is not None:
        city_key = normalize_city_name(city_name)
        work = work[work["city"] == city_key].copy()

    cities = sorted(work["city"].dropna().unique().tolist())
    for city in cities:
        results[city] = run_city_regression(
            work,
            city=city,
            top_n_neighbourhoods=top_n_neighbourhoods,
            random_state=random_state,
        )

    if not results:
        raise ValueError("No valid city rows available for regression analysis.")

    print("\n[regression] OLS Summary Tables")
    for city in cities:
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
