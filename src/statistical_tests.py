from __future__ import annotations

from itertools import combinations
from pathlib import Path

import pandas as pd
from scipy.stats import ks_2samp, mannwhitneyu

from src.config import TABLES_DIR, normalize_city_name


BASE_DIR = Path(__file__).resolve().parent.parent


def _interpret_pvalue(p_value: float, alpha: float) -> str:
    """Return significance interpretation for a p-value."""
    return "significant" if p_value < alpha else "not significant"


def _price_series(df: pd.DataFrame) -> pd.Series:
    """Return clean positive price values."""
    if "price" not in df.columns:
        raise KeyError("Input dataframe must contain 'price'.")
    series = pd.to_numeric(df["price"], errors="coerce")
    return series[series > 0].dropna()

EFFECT_SIZE_THRESHOLDS: dict[str, float] = {
    "negligible": 0.0,
    "small": 0.1,
    "medium": 0.3,
    "large": 0.5,
}


def _classify_effect_size(rbc: float) -> str:
    """Return Cohen's qualitative bucket for |rbc|."""
    magnitude = abs(rbc)
    label = "negligible"
    for name, thresh in EFFECT_SIZE_THRESHOLDS.items():
        if magnitude >= thresh:
            label = name
    return label


def pairwise_price_tests_with_effects(
    df: pd.DataFrame,
    group_col: str = "city",
    value_col: str = "price",
    alpha: float = 0.05,
    multitest_method: str = "fdr_bh",
) -> tuple[dict[str, float], pd.DataFrame]:
    """Run Kruskal–Wallis + all pairwise Mann–Whitney U with effect sizes.

    Parameters
    ----------
    df
        Long frame with one row per observation.
    group_col
        Column holding group labels (e.g. city).
    value_col
        Column holding the continuous value (e.g. price).
    alpha
        Significance threshold for the flag column.
    multitest_method
        Any value accepted by statsmodels multipletests (default is
        Benjamini–Hochberg FDR).

    Returns
    -------
    global_stats : dict
        Kruskal–Wallis {'H', 'p', 'n_groups'}.
    pairwise : pd.DataFrame
        One row per group pair, with columns group_1, group_2, U_stat,
        p_raw, p_adj, rbc, effect_size, significant.
    """
    work = df[[group_col, value_col]].copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.dropna()

    groups = sorted(work[group_col].unique())
    if len(groups) < 2:
        raise ValueError(f"Need at least 2 groups in {group_col}; got {len(groups)}.")

    # Global Kruskal–Wallis
    samples = [work.loc[work[group_col] == g, value_col].values for g in groups]
    H, p_kw = kruskal(*samples)
    global_stats = {"H": float(H), "p": float(p_kw), "n_groups": len(groups)}

    # Pairwise Mann–Whitney U with rank-biserial correlation
    rows: list[dict[str, object]] = []
    for a, b in combinations(groups, 2):
        xa = work.loc[work[group_col] == a, value_col].values
        xb = work.loc[work[group_col] == b, value_col].values
        u_stat, p_raw = mannwhitneyu(xa, xb, alternative="two-sided")

        # Rank-biserial correlation = 1 - 2U / (n1 * n2)
        # Sign convention: positive rbc means group A tends to exceed group B.
        n1, n2 = len(xa), len(xb)
        rbc = 1.0 - (2.0 * u_stat) / (n1 * n2)
        rows.append({
            "group_1": a,
            "group_2": b,
            "n_1": n1,
            "n_2": n2,
            "U_stat": float(u_stat),
            "p_raw": float(p_raw),
            "rbc": float(rbc),
            "effect_size": _classify_effect_size(rbc),
        })

    pairwise = pd.DataFrame(rows)
    if len(pairwise):
        _, p_adj, _, _ = multipletests(pairwise["p_raw"].values, method=multitest_method)
        pairwise["p_adj"] = p_adj
        pairwise["significant"] = pairwise["p_adj"] < alpha

    # Sensible column order for the output CSV
    col_order = [
        "group_1", "group_2", "n_1", "n_2",
        "U_stat", "p_raw", "p_adj",
        "rbc", "effect_size", "significant",
    ]
    pairwise = pairwise[[c for c in col_order if c in pairwise.columns]]

    return global_stats, pairwise


def effect_size_summary(pairwise_df: pd.DataFrame) -> pd.DataFrame:
    """Roll up effect-size counts (how many pairs land in each bucket)."""
    if pairwise_df.empty:
        return pd.DataFrame(columns=["bucket", "n_pairs", "pct"])
    counts = pairwise_df["effect_size"].value_counts().reindex(
        ["negligible", "small", "medium", "large"], fill_value=0
    )
    total = int(counts.sum())
    out = pd.DataFrame({
        "bucket": counts.index,
        "n_pairs": counts.values.astype(int),
        "pct": (counts.values / total * 100).round(1) if total else 0.0,
    })
    return out.reset_index(drop=True)


if __name__ == "__main__":
    # Smoke test — fabricate three groups with mild location shifts.
    rng = np.random.default_rng(42)
    toy = pd.concat([
        pd.DataFrame({"city": "a", "price": rng.lognormal(5.0, 0.5, 1000)}),
        pd.DataFrame({"city": "b", "price": rng.lognormal(5.2, 0.5, 1000)}),
        pd.DataFrame({"city": "c", "price": rng.lognormal(5.1, 0.5, 1000)}),
    ], ignore_index=True)
    global_stats, pairs = pairwise_price_tests_with_effects(toy)
    print("Global:", global_stats)
    print(pairs.to_string(index=False))
    print("\nEffect size roll-up:")
    print(effect_size_summary(pairs).to_string(index=False))



def run_statistical_comparison(
    df: pd.DataFrame,
    alpha: float = 0.05,
    city_name: str | None = None,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Run price distribution tests across cities and room types and return summary table."""
    required_cols = {"city", "room_type", "price"}
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    work = df.copy()
    work["city"] = work["city"].astype(str).str.lower()
    if city_name is not None:
        city_key = normalize_city_name(city_name)
        work = work[work["city"] == city_key].copy()

    results: list[dict[str, object]] = []

    cities = sorted(work["city"].dropna().unique().tolist())
    for city_1, city_2 in combinations(cities, 2):
        prices_1 = _price_series(work[work["city"] == city_1])
        prices_2 = _price_series(work[work["city"] == city_2])
        if len(prices_1) == 0 or len(prices_2) == 0:
            continue

        mw_stat, mw_p = mannwhitneyu(prices_1, prices_2, alternative="two-sided")
        results.append(
            {
                "test_name": f"Mann-Whitney U ({city_1} vs {city_2})",
                "group_1": city_1,
                "group_2": city_2,
                "statistic": float(mw_stat),
                "p_value": float(mw_p),
                "interpretation": _interpret_pvalue(float(mw_p), alpha),
            }
        )

        ks_stat, ks_p = ks_2samp(prices_1, prices_2, alternative="two-sided", method="auto")
        results.append(
            {
                "test_name": f"Kolmogorov-Smirnov ({city_1} vs {city_2})",
                "group_1": city_1,
                "group_2": city_2,
                "statistic": float(ks_stat),
                "p_value": float(ks_p),
                "interpretation": _interpret_pvalue(float(ks_p), alpha),
            }
        )

    for city in cities:
        city_df = work[work["city"] == city].copy()
        room_types = [
            room
            for room, values in city_df.groupby("room_type")
            if len(_price_series(values)) > 0
        ]

        for rt1, rt2 in combinations(sorted(room_types), 2):
            prices_1 = _price_series(city_df[city_df["room_type"] == rt1])
            prices_2 = _price_series(city_df[city_df["room_type"] == rt2])
            if len(prices_1) == 0 or len(prices_2) == 0:
                continue

            stat, p_value = mannwhitneyu(prices_1, prices_2, alternative="two-sided")
            results.append(
                {
                    "test_name": f"Mann-Whitney U ({city}: room type)",
                    "group_1": f"{city}:{rt1}",
                    "group_2": f"{city}:{rt2}",
                    "statistic": float(stat),
                    "p_value": float(p_value),
                    "interpretation": _interpret_pvalue(float(p_value), alpha),
                }
            )

    summary = pd.DataFrame(results)
    if summary.empty:
        saved_path = save_statistical_summary(summary, output_dir=output_dir)
        print(f"[statistical_tests] Saved statistical summary to: {saved_path}")
        return summary

    summary = summary.sort_values(["test_name", "p_value"]).reset_index(drop=True)
    saved_path = save_statistical_summary(summary, output_dir=output_dir)
    print(f"[statistical_tests] Saved statistical summary to: {saved_path}")
    return summary


def save_statistical_summary(
    summary_df: pd.DataFrame,
    output_dir: Path | None = None,
) -> Path:
    """Save statistical test summary DataFrame to outputs/tables/."""
    out_dir = output_dir or TABLES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "statistical_tests_summary.csv"
    summary_df.to_csv(out_path, index=False)
    return out_path
