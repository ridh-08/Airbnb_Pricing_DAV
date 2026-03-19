from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

# Force non-interactive backend for plot generation in script runs.
import matplotlib

matplotlib.use("Agg")


BASE_DIR = Path(__file__).resolve().parent
VALIDATION_OUTPUTS = BASE_DIR / "outputs" / "validation"
VALIDATION_PLOTS = VALIDATION_OUTPUTS / "plots"
VALIDATION_TABLES = VALIDATION_OUTPUTS / "tables"

MODULE_NAMES = [
    "src.preprocessing",
    "src.feature_engineering",
    "src.visualization",
    "src.clustering",
    "src.spatial_analysis",
    "src.statistical_tests",
    "src.regression_analysis",
    "src.calendar_analysis",
    "src.data_loader",
]


def build_dummy_df(n_per_city: int = 80) -> pd.DataFrame:
    """Build a synthetic dataframe with required columns for fix validation."""
    rng = np.random.default_rng(42)
    cities = np.array(["newyork"] * n_per_city + ["chicago"] * n_per_city)
    n_total = len(cities)

    room_type_choices = np.array(["Entire home/apt", "Private room", "Shared room"])
    room_type = room_type_choices[np.arange(n_total) % len(room_type_choices)]

    neighbourhood = np.array(
        [f"{city}_neigh_{(i % 12) + 1}" for i, city in enumerate(cities)],
        dtype=object,
    )

    accommodates = rng.integers(1, 7, size=n_total)
    amenities_count = rng.integers(5, 35, size=n_total)
    availability_365 = rng.integers(10, 365, size=n_total)
    reviews_per_month = rng.uniform(0.05, 6.0, size=n_total)
    number_of_reviews = rng.integers(1, 350, size=n_total)
    minimum_nights = rng.integers(1, 30, size=n_total)

    city_price_base = np.where(cities == "newyork", 180.0, 120.0)
    price = (
        city_price_base
        + accommodates * 18
        + amenities_count * 1.5
        - availability_365 * 0.03
        + rng.normal(0, 15, size=n_total)
    )
    price = np.clip(price, 30, None)
    log_price = np.log(price)

    latitude = np.where(
        cities == "newyork",
        rng.normal(40.73, 0.05, size=n_total),
        rng.normal(41.88, 0.05, size=n_total),
    )
    longitude = np.where(
        cities == "newyork",
        rng.normal(-73.98, 0.06, size=n_total),
        rng.normal(-87.63, 0.06, size=n_total),
    )

    # Demand score with the requested weighted normalization structure.
    rpm_norm = (reviews_per_month - reviews_per_month.min()) / (
        reviews_per_month.max() - reviews_per_month.min()
    )
    nor_norm = (number_of_reviews - number_of_reviews.min()) / (
        number_of_reviews.max() - number_of_reviews.min()
    )
    avail_norm = (availability_365 - availability_365.min()) / (
        availability_365.max() - availability_365.min()
    )
    demand_score = 0.4 * rpm_norm + 0.3 * nor_norm + 0.3 * (1.0 - avail_norm)

    df = pd.DataFrame(
        {
            "listing_id": np.arange(1, n_total + 1),
            "city": cities,
            "price": price.astype(float),
            "log_price": log_price.astype(float),
            "room_type": room_type,
            "neighbourhood": neighbourhood,
            "accommodates": accommodates.astype(float),
            "amenities_count": amenities_count.astype(float),
            "availability_365": availability_365.astype(float),
            "reviews_per_month": reviews_per_month.astype(float),
            "number_of_reviews": number_of_reviews.astype(float),
            "minimum_nights": minimum_nights.astype(float),
            "latitude": latitude.astype(float),
            "longitude": longitude.astype(float),
            "demand_score": demand_score.astype(float),
            "price_imputed": False,
            "amenities": "{Wifi,Kitchen,Heating}",
        }
    )

    # Pre-populate cluster labels for demand plot validation.
    df["cluster_kmeans"] = np.arange(n_total) % 3
    df["cluster"] = df["cluster_kmeans"]
    return df


def run_check(name: str, fn: Callable[[], Any]) -> tuple[bool, str]:
    """Run one validation check and return status + message."""
    try:
        fn()
        return True, f"PASS: {name}"
    except Exception as exc:  # pylint: disable=broad-except
        return False, f"FAIL: {name} -> {type(exc).__name__}: {exc}"


def main() -> None:
    """Run smoke validations for all fixed functions with a dummy dataframe."""
    VALIDATION_PLOTS.mkdir(parents=True, exist_ok=True)
    VALIDATION_TABLES.mkdir(parents=True, exist_ok=True)

    checks: list[tuple[str, Callable[[], Any]]] = []

    imported_modules: dict[str, Any] = {}

    def _import_all_modules() -> None:
        for mod_name in MODULE_NAMES:
            imported_modules[mod_name] = import_module(mod_name)

    checks.append(("import_all_src_modules", _import_all_modules))

    dummy_df = build_dummy_df(n_per_city=80)

    def _run_statistical_comparison() -> None:
        statistical_tests = import_module("src.statistical_tests")
        summary_df = statistical_tests.run_statistical_comparison(dummy_df)
        statistical_tests.save_statistical_summary(summary_df, output_dir=VALIDATION_TABLES)

    checks.append(("run_statistical_comparison", _run_statistical_comparison))

    def _save_statistical_summary() -> None:
        statistical_tests = import_module("src.statistical_tests")
        sample_summary = pd.DataFrame(
            {
                "test_name": ["dummy"],
                "statistic": [1.0],
                "p_value": [0.04],
                "interpretation": ["significant"],
            }
        )
        statistical_tests.save_statistical_summary(sample_summary, output_dir=VALIDATION_TABLES)

    checks.append(("save_statistical_summary", _save_statistical_summary))

    def _run_dbscan_spatial_clustering() -> None:
        clustering = import_module("src.clustering")
        result = clustering.run_dbscan_spatial_clustering(dummy_df, city="newyork")
        _ = result["noise_count"], result["noise_pct"]

    checks.append(("run_dbscan_spatial_clustering", _run_dbscan_spatial_clustering))

    def _run_clustering_on_dataframe() -> None:
        clustering = import_module("src.clustering")
        clustering.run_clustering_on_dataframe(
            dummy_df,
            output_tables_dir=VALIDATION_TABLES,
            output_plots_dir=VALIDATION_PLOTS,
        )

    checks.append(("run_clustering_on_dataframe", _run_clustering_on_dataframe))

    def _plot_demand_score_vs_price() -> None:
        feature_engineering = import_module("src.feature_engineering")
        feature_engineering.plot_demand_score_vs_price(
            dummy_df,
            output_dir=VALIDATION_PLOTS,
            cluster_col="cluster_kmeans",
        )

    checks.append(("plot_demand_score_vs_price", _plot_demand_score_vs_price))

    def _run_regression_analysis_and_clean_summary() -> None:
        regression_analysis = import_module("src.regression_analysis")
        result = regression_analysis.run_regression_analysis(
            dummy_df,
            output_dir=VALIDATION_PLOTS,
            top_n_neighbourhoods=10,
            random_state=42,
        )
        ny_summary = result["city_results"]["newyork"]["ols_summary"]
        regression_analysis.get_clean_ols_summary(ny_summary)

    checks.append(("run_regression_analysis", _run_regression_analysis_and_clean_summary))
    checks.append(("get_clean_ols_summary", _run_regression_analysis_and_clean_summary))

    def _run_neighbourhood_inequality_analysis() -> None:
        visualization = import_module("src.visualization")
        visualization.run_neighbourhood_inequality_analysis(dummy_df, output_dir=VALIDATION_PLOTS)

    checks.append(("run_neighbourhood_inequality_analysis", _run_neighbourhood_inequality_analysis))

    print("validate_fixes.py :: starting validation")
    print(f"validation output dir: {VALIDATION_OUTPUTS}")

    pass_count = 0
    fail_count = 0

    for name, fn in checks:
        ok, msg = run_check(name, fn)
        print(msg)
        if ok:
            pass_count += 1
        else:
            fail_count += 1

    print("-")
    print(f"Summary: PASS={pass_count}, FAIL={fail_count}")
    if fail_count:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
