# Airbnb Pricing and Neighbourhood Dynamics (CSE520)

Comprehensive multi-city Airbnb analytics pipeline covering data cleaning, feature engineering, EDA, clustering, spatial analysis, calendar analysis, statistical testing, and predictive modeling.

## 1. Project Scope

- Course: CSE520 (Data Analytics and Visualization)
- Team:
	- Riddhi Bhargava (AU2220028)
	- Vaishnavi Dahihande (AU2220080)
- Cities analyzed:
	- New York
	- Chicago
	- Austin
	- Los Angeles
	- Nashville
	- New Orleans

The repository is designed as an end-to-end research pipeline: raw files are ingested, transformed into analysis-ready city datasets, analyzed per city and across cities, and exported as publication-ready figures/tables.

## 2. Repository Layout

- `src/`: production analysis modules
- `data/raw/`: raw-source files (if kept locally)
- `data/processed/`: cleaned, featured, clustered city datasets
- `outputs/`: generated artifacts (plots, CSVs, HTML maps, workbooks)
- `notebooks/`: stage-wise exploratory notebooks
- `templates/`, `app.py`: interactive dashboard layer
- `run_pipeline.py`: single entrypoint for full pipeline execution
- `validate_fixes.py`: smoke validation script for key analysis functions

## 3. Pipeline Stages Implemented

### 3.1 Data Loading and Cleaning

Module: `src/preprocessing.py`, `src/data_loader.py`

Implemented:
- Multi-city file discovery and standardized loading
- Price cleaning and numeric coercion
- Missing value handling/imputation strategy
- Export of cleaned datasets per city

Core outputs (examples):
- `data/processed/*_listings_clean.csv`
- `data/processed/*_reviews_clean.csv`
- `data/processed/*_calendar_sample_clean.csv`
- `data/processed/*_analysis_ready.csv`

### 3.2 Feature Engineering

Module: `src/feature_engineering.py`

Implemented:
- `log_price`
- `price_per_person`
- `amenities_count`
- `review_density`
- `demand_score` (weighted normalized composite)
- Feature-enriched exports for all cities

Core outputs:
- `data/processed/*_featured.csv`

### 3.3 EDA and Visualization

Module: `src/visualization.py`

Implemented charts include:
- Price distribution
- Correlation heatmap
- Price vs reviews
- Price vs availability
- Room-type binned comparisons
- Neighbourhood inequality visuals

Core outputs (examples):
- `outputs/plots/price_distribution.png`
- `outputs/plots/correlation_heatmap.png`
- `outputs/plots/scatter_price_vs_reviews.png`
- `outputs/plots/scatter_price_vs_availability.png`
- `outputs/plots/price_vs_reviews_roomtype_binned.png`
- `outputs/plots/price_vs_availability_roomtype_binned.png`

### 3.4 Clustering and Host Segmentation

Module: `src/clustering.py`

Implemented:
- Pooled and city-level KMeans analyses
- PCA scatter visual diagnostics
- Cluster composition tables
- Host strategy segmentation outputs

Core outputs (examples):
- `outputs/plots/*_kmeans_pca_scatter.png`
- `outputs/plots/pooled_kmeans_cluster_composition_by_city.png`
- `outputs/tables/cluster_summary.csv`
- `outputs/tables/pooled_cluster_composition.csv`
- `outputs/multi_city/tables/host_strategy_cluster_summary.csv`

### 3.5 Spatial Analysis

Module: `src/spatial_analysis.py`

Implemented:
- Neighbourhood-level price surfaces
- Moran's I spatial autocorrelation summaries
- Choropleth maps by city
- Cluster-map exports

Core outputs (examples):
- `outputs/plots/*_price_choropleth.html`
- `outputs/plots/*_cluster_map.html`
- `outputs/tables/spatial_summary.csv`
- `outputs/tables/spatial_clustering_classification.csv`

### 3.6 Calendar and Temporal Analysis

Module: `src/calendar_analysis.py`

Implemented:
- Monthly trend analysis for price and availability
- Calendar-derived city summaries
- Temporal heatmaps

Core outputs (examples):
- `outputs/plots/calendar_availability_trend.png`
- `outputs/plots/calendar_availability_heatmap_month_weekday.png`
- `outputs/plots/calendar_price_heatmap_month_weekday.png`
- `outputs/tables/calendar_summary.csv`
- `outputs/tables/calendar_temporal_summary.csv`

### 3.7 Statistical Testing

Module: `src/statistical_tests.py`

Implemented:
- Pairwise city distribution comparisons
- Multiple-testing corrected significance reporting
- Statistical summary exports

Core outputs:
- `outputs/tables/statistical_tests_summary.csv`
- `outputs/multi_city/tables/cross_city_tests.csv`

### 3.8 Predictive Modeling (Enhanced)

Modules: `src/regression_analysis.py`, `src/multi_city_analysis.py`

Implemented model stack:
- OLS regression with interpretable coefficients
- Random Forest feature importance
- Enhanced XGBoost city models

Recent predictive enhancements:
- Richer predictor space (numeric + categorical)
- Room-type and neighbourhood encoding
- Log1p target modeling for price stability
- Lightweight city-specific hyperparameter tuning
- Expanded metric tracking: R2, RMSE, MAE, MAPE, validation R2
- Diagnostic chart: actual vs predicted by city

Core outputs (examples):
- `outputs/plots/model_r2_comparison_by_city.png`
- `outputs/multi_city/plots/model_r2_comparison_by_city.png`
- `outputs/multi_city/plots/xgboost_actual_vs_predicted_by_city.png`
- `outputs/multi_city/tables/xgboost_city_metrics.csv`
- `outputs/multi_city/tables/xgboost_tuning_summary.csv`
- `outputs/multi_city/tables/pooled_fixed_effects.csv`
- `outputs/multi_city/tables/pooled_quantile.csv`
- `outputs/*/plots/*_xgboost_shap_beeswarm_top10.png`

## 4. Multi-City Comparative Outputs

Module: `src/multi_city_analysis.py`

Cross-city comparative artifacts include:
- City median price comparison
- Cluster composition by city
- City feature profile heatmap
- Pooled fixed-effects and quantile regression summaries
- Cross-city non-parametric statistical tests

Key files:
- `outputs/multi_city/plots/city_median_price.png`
- `outputs/multi_city/plots/cluster_composition_by_city.png`
- `outputs/multi_city/plots/city_feature_profile_heatmap.png`
- `outputs/multi_city/tables/multi_city_analysis_summary.xlsx`

## 5. End-to-End Execution

Run full pipeline:

```powershell
C:/Users/Riddhi/AppData/Local/Programs/Python/Python311/python.exe run_pipeline.py
```

Run validation smoke checks:

```powershell
C:/Users/Riddhi/AppData/Local/Programs/Python/Python311/python.exe validate_fixes.py
```

Launch dashboard (after outputs are available):

```powershell
C:/Users/Riddhi/AppData/Local/Programs/Python/Python311/python.exe app.py
```

## 6. Important Generated Folders

- City-level outputs:
	- `outputs/austin/`
	- `outputs/chicago/`
	- `outputs/losangeles/`
	- `outputs/nashville/`
	- `outputs/neworleans/`
	- `outputs/newyork/`
- Shared aggregate outputs:
	- `outputs/plots/`
	- `outputs/tables/`
	- `outputs/multi_city/plots/`
	- `outputs/multi_city/tables/`
	- `outputs/consolidated/airbnb_multicity_summary.xlsx`

## 7. Notes for Final Report Writing

Recommended structure for your written submission:
- Problem statement and city coverage
- Data sources and cleaning assumptions
- Feature engineering rationale
- EDA findings
- Cluster and host segmentation findings
- Spatial findings (Moran's I + neighbourhood leaders)
- Calendar/seasonality findings
- Statistical significance findings
- Predictive model comparison and enhanced XGBoost results
- Limitations and future work

All required figures/tables for those sections are already produced under `outputs/` and `outputs/multi_city/`.
