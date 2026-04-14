# Project File-by-File Analysis Catalog

This document was generated to preserve a detailed project inventory and the analytical role of each file.

Scope: all project files under this workspace excluding .git, .venv, cache folders, and notebook checkpoints.

Total files cataloged: 171

## High-Level Analysis Modules

- `src/preprocessing.py`: cleaning and normalized city datasets.
- `src/feature_engineering.py`: engineered predictors and demand metrics.
- `src/visualization.py`: EDA plots and quality summaries.
- `src/clustering.py`: KMeans/DBSCAN segmentation and diagnostics.
- `src/spatial_analysis.py`: geospatial summaries and map artifacts.
- `src/calendar_analysis.py`: temporal pricing/availability analytics.
- `src/statistical_tests.py`: significance testing and test tables.
- `src/regression_analysis.py`: OLS/RF/XGBoost modeling and diagnostics.
- `src/multi_city_analysis.py`: cross-city pooled comparisons and exports.

## File Catalog

| File | Purpose | Analysis/Role |
|---|---|---|
| .gitattributes | Git LFS and attribute configuration for large data files. | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| .gitignore | Git ignore rules controlling tracked vs local-only files. | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| .vscode\settings.json | Project file used in data, analysis, or reporting workflow. | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| app.py | Flask API/server for dashboard data and prediction endpoints. | Serves aggregated analysis APIs and scenario prediction endpoints for interactive exploration. |
| Austin\calendar.csv\calendar.csv | CSV data artifact (source, intermediate, or result table). | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| Austin\listings (1).csv\listings.csv | CSV data artifact (source, intermediate, or result table). | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| Austin\neighbourhoods (1).geojson | Geospatial boundary/input dataset. | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| Austin\reviews.csv\reviews.csv | CSV data artifact (source, intermediate, or result table). | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| Chicago\calendar.csv\calendar.csv | CSV data artifact (source, intermediate, or result table). | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| Chicago\listings (1).csv\listings.csv | CSV data artifact (source, intermediate, or result table). | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| Chicago\neighbourhoods.geojson | Geospatial boundary/input dataset. | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| Chicago\reviews.csv\reviews.csv | CSV data artifact (source, intermediate, or result table). | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| data\processed\austin_analysis_ready.csv | CSV data artifact (source, intermediate, or result table). | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| data\processed\austin_calendar_sample_clean.csv | CSV data artifact (source, intermediate, or result table). | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| data\processed\austin_clustered.csv | CSV data artifact (source, intermediate, or result table). | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| data\processed\austin_featured.csv | CSV data artifact (source, intermediate, or result table). | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| data\processed\austin_listings_clean.csv | CSV data artifact (source, intermediate, or result table). | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| data\processed\austin_reviews_clean.csv | CSV data artifact (source, intermediate, or result table). | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| data\processed\austin_reviews_summary.csv | CSV data artifact (source, intermediate, or result table). | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| data\processed\chicago_analysis_ready.csv | CSV data artifact (source, intermediate, or result table). | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| data\processed\chicago_calendar_sample_clean.csv | CSV data artifact (source, intermediate, or result table). | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| data\processed\chicago_clustered.csv | CSV data artifact (source, intermediate, or result table). | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| data\processed\chicago_featured.csv | CSV data artifact (source, intermediate, or result table). | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| data\processed\chicago_listings_clean.csv | CSV data artifact (source, intermediate, or result table). | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| data\processed\chicago_reviews_clean.csv | CSV data artifact (source, intermediate, or result table). | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| data\processed\chicago_reviews_summary.csv | CSV data artifact (source, intermediate, or result table). | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| data\processed\losangeles_analysis_ready.csv | CSV data artifact (source, intermediate, or result table). | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| data\processed\losangeles_calendar_sample_clean.csv | CSV data artifact (source, intermediate, or result table). | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| data\processed\losangeles_clustered.csv | CSV data artifact (source, intermediate, or result table). | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| data\processed\losangeles_featured.csv | CSV data artifact (source, intermediate, or result table). | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| data\processed\losangeles_listings_clean.csv | CSV data artifact (source, intermediate, or result table). | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| data\processed\losangeles_reviews_clean.csv | CSV data artifact (source, intermediate, or result table). | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| data\processed\losangeles_reviews_summary.csv | CSV data artifact (source, intermediate, or result table). | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| data\processed\nashville_analysis_ready.csv | CSV data artifact (source, intermediate, or result table). | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| data\processed\nashville_calendar_sample_clean.csv | CSV data artifact (source, intermediate, or result table). | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| data\processed\nashville_clustered.csv | CSV data artifact (source, intermediate, or result table). | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| data\processed\nashville_featured.csv | CSV data artifact (source, intermediate, or result table). | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| data\processed\nashville_listings_clean.csv | CSV data artifact (source, intermediate, or result table). | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| data\processed\nashville_reviews_clean.csv | CSV data artifact (source, intermediate, or result table). | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| data\processed\nashville_reviews_summary.csv | CSV data artifact (source, intermediate, or result table). | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| data\processed\neworleans_analysis_ready.csv | CSV data artifact (source, intermediate, or result table). | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| data\processed\neworleans_calendar_sample_clean.csv | CSV data artifact (source, intermediate, or result table). | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| data\processed\neworleans_clustered.csv | CSV data artifact (source, intermediate, or result table). | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| data\processed\neworleans_featured.csv | CSV data artifact (source, intermediate, or result table). | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| data\processed\neworleans_listings_clean.csv | CSV data artifact (source, intermediate, or result table). | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| data\processed\neworleans_reviews_clean.csv | CSV data artifact (source, intermediate, or result table). | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| data\processed\neworleans_reviews_summary.csv | CSV data artifact (source, intermediate, or result table). | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| data\processed\newyork_analysis_ready.csv | CSV data artifact (source, intermediate, or result table). | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| data\processed\newyork_calendar_sample_clean.csv | CSV data artifact (source, intermediate, or result table). | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| data\processed\newyork_clustered.csv | CSV data artifact (source, intermediate, or result table). | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| data\processed\newyork_featured.csv | CSV data artifact (source, intermediate, or result table). | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| data\processed\newyork_listings_clean.csv | CSV data artifact (source, intermediate, or result table). | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| data\processed\newyork_reviews_clean.csv | CSV data artifact (source, intermediate, or result table). | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| data\processed\newyork_reviews_summary.csv | CSV data artifact (source, intermediate, or result table). | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| data\processed\ny_clustered.csv | CSV data artifact (source, intermediate, or result table). | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| feature_engineering.py | Python source or utility script in project workflow. | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| instructions.py | Python source or utility script in project workflow. | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| Los Angeles\calendar.csv\calendar.csv | CSV data artifact (source, intermediate, or result table). | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| Los Angeles\listings (1).csv\listings.csv | CSV data artifact (source, intermediate, or result table). | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| Los Angeles\neighbourhoods.geojson | Geospatial boundary/input dataset. | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| Los Angeles\reviews.csv\reviews.csv | CSV data artifact (source, intermediate, or result table). | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| Nashville\calendar (1).csv\calendar.csv | CSV data artifact (source, intermediate, or result table). | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| Nashville\listings (2).csv\listings.csv | CSV data artifact (source, intermediate, or result table). | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| Nashville\neighbourhoods.geojson | Geospatial boundary/input dataset. | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| Nashville\reviews (1).csv\reviews.csv | CSV data artifact (source, intermediate, or result table). | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| New Orleans\calendar.csv\calendar.csv | CSV data artifact (source, intermediate, or result table). | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| New Orleans\listings (1).csv\listings.csv | CSV data artifact (source, intermediate, or result table). | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| New Orleans\neighbourhoods.geojson | Geospatial boundary/input dataset. | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| New Orleans\reviews.csv\reviews.csv | CSV data artifact (source, intermediate, or result table). | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| New York\calendar.csv\calendar.csv | CSV data artifact (source, intermediate, or result table). | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| New York\listings (3).csv\listings.csv | CSV data artifact (source, intermediate, or result table). | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| New York\neighbourhoods.geojson | Geospatial boundary/input dataset. | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| New York\reviews.csv\reviews.csv | CSV data artifact (source, intermediate, or result table). | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| notebooks\01_data_cleaning.ipynb | Notebook-based analysis and experimentation file. | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| notebooks\02_feature_engineering.ipynb | Notebook-based analysis and experimentation file. | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| notebooks\03_eda_analysis.ipynb | Notebook-based analysis and experimentation file. | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| notebooks\04_clustering.ipynb | Notebook-based analysis and experimentation file. | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| notebooks\05_spatial_analysis.ipynb | Notebook-based analysis and experimentation file. | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| notebooks\06_calendar_analysis.ipynb | Notebook-based analysis and experimentation file. | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| outputs\austin\plots\austin_xgboost_shap_beeswarm_top10.png | Generated visualization image for analysis/reporting. | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| outputs\chicago\plots\chicago_xgboost_shap_beeswarm_top10.png | Generated visualization image for analysis/reporting. | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| outputs\losangeles\plots\losangeles_xgboost_shap_beeswarm_top10.png | Generated visualization image for analysis/reporting. | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| outputs\multi_city\plots\city_feature_profile_heatmap.png | Generated visualization image for analysis/reporting. | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| outputs\multi_city\plots\city_median_price.png | Generated visualization image for analysis/reporting. | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| outputs\multi_city\plots\cluster_composition_by_city.png | Generated visualization image for analysis/reporting. | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| outputs\multi_city\plots\model_r2_comparison_by_city.png | Generated visualization image for analysis/reporting. | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| outputs\multi_city\plots\shap_top5_rank_heatmap.png | Generated visualization image for analysis/reporting. | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| outputs\multi_city\tables\cross_city_tests.csv | CSV data artifact (source, intermediate, or result table). | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| outputs\multi_city\tables\host_strategy_cluster_labels.csv | CSV data artifact (source, intermediate, or result table). | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| outputs\multi_city\tables\host_strategy_cluster_summary.csv | CSV data artifact (source, intermediate, or result table). | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| outputs\multi_city\tables\multi_city_analysis_summary.xlsx | Project file used in data, analysis, or reporting workflow. | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| outputs\multi_city\tables\newyork_xgboost_shap_summary.csv | CSV data artifact (source, intermediate, or result table). | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| outputs\multi_city\tables\pooled_fixed_effects.csv | CSV data artifact (source, intermediate, or result table). | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| outputs\multi_city\tables\pooled_quantile.csv | CSV data artifact (source, intermediate, or result table). | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| outputs\multi_city\tables\shap_city_dominance_checks.csv | CSV data artifact (source, intermediate, or result table). | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| outputs\multi_city\tables\shap_top5_comparison_table.csv | CSV data artifact (source, intermediate, or result table). | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| outputs\multi_city\tables\shap_top5_features_by_city.csv | CSV data artifact (source, intermediate, or result table). | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| outputs\multi_city\tables\xgboost_city_metrics.csv | CSV data artifact (source, intermediate, or result table). | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| outputs\multi_city\tables\xgboost_progression_history.csv | CSV data artifact (source, intermediate, or result table). | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| outputs\multi_city\tables\xgboost_progression_log.md | Project file used in data, analysis, or reporting workflow. | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| outputs\nashville\plots\nashville_xgboost_shap_beeswarm_top10.png | Generated visualization image for analysis/reporting. | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| outputs\neworleans\plots\neworleans_xgboost_shap_beeswarm_top10.png | Generated visualization image for analysis/reporting. | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| outputs\newyork\plots\newyork_xgboost_shap_beeswarm_top10.png | Generated visualization image for analysis/reporting. | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| outputs\plots\austin_cluster_map.html | Interactive HTML artifact for map/dashboard usage. | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| outputs\plots\austin_kmeans_pca_scatter.png | Generated visualization image for analysis/reporting. | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| outputs\plots\austin_price_choropleth.html | Interactive HTML artifact for map/dashboard usage. | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| outputs\plots\boxplot_price_by_room_type.png | Generated visualization image for analysis/reporting. | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| outputs\plots\calendar_availability_heatmap_month_weekday.png | Generated visualization image for analysis/reporting. | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| outputs\plots\calendar_availability_trend.png | Generated visualization image for analysis/reporting. | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| outputs\plots\calendar_price_heatmap_month_weekday.png | Generated visualization image for analysis/reporting. | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| outputs\plots\calendar_price_variation_selected_listings.png | Generated visualization image for analysis/reporting. | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| outputs\plots\chicago_cluster_map.html | Interactive HTML artifact for map/dashboard usage. | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| outputs\plots\chicago_kmeans_pca_scatter.png | Generated visualization image for analysis/reporting. | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| outputs\plots\chicago_price_choropleth.html | Interactive HTML artifact for map/dashboard usage. | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| outputs\plots\correlation_heatmap.png | Generated visualization image for analysis/reporting. | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| outputs\plots\losangeles_cluster_map.html | Interactive HTML artifact for map/dashboard usage. | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| outputs\plots\losangeles_kmeans_pca_scatter.png | Generated visualization image for analysis/reporting. | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| outputs\plots\losangeles_price_choropleth.html | Interactive HTML artifact for map/dashboard usage. | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| outputs\plots\model_r2_comparison_by_city.png | Generated visualization image for analysis/reporting. | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| outputs\plots\nashville_cluster_map.html | Interactive HTML artifact for map/dashboard usage. | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| outputs\plots\nashville_kmeans_pca_scatter.png | Generated visualization image for analysis/reporting. | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| outputs\plots\nashville_price_choropleth.html | Interactive HTML artifact for map/dashboard usage. | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| outputs\plots\neworleans_cluster_map.html | Interactive HTML artifact for map/dashboard usage. | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| outputs\plots\neworleans_kmeans_pca_scatter.png | Generated visualization image for analysis/reporting. | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| outputs\plots\neworleans_price_choropleth.html | Interactive HTML artifact for map/dashboard usage. | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| outputs\plots\newyork_cluster_map.html | Interactive HTML artifact for map/dashboard usage. | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| outputs\plots\newyork_kmeans_pca_scatter.png | Generated visualization image for analysis/reporting. | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| outputs\plots\newyork_price_choropleth.html | Interactive HTML artifact for map/dashboard usage. | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| outputs\plots\pooled_kmeans_cluster_composition_by_city.png | Generated visualization image for analysis/reporting. | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| outputs\plots\price_distribution.png | Generated visualization image for analysis/reporting. | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| outputs\plots\price_vs_availability_roomtype_binned.png | Generated visualization image for analysis/reporting. | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| outputs\plots\price_vs_reviews_roomtype_binned.png | Generated visualization image for analysis/reporting. | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| outputs\plots\scatter_price_vs_availability.png | Generated visualization image for analysis/reporting. | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| outputs\plots\scatter_price_vs_reviews.png | Generated visualization image for analysis/reporting. | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| outputs\tables\calendar_summary.csv | CSV data artifact (source, intermediate, or result table). | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| outputs\tables\calendar_temporal_summary.csv | CSV data artifact (source, intermediate, or result table). | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| outputs\tables\cluster_summary.csv | CSV data artifact (source, intermediate, or result table). | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| outputs\tables\data_quality_summary.csv | CSV data artifact (source, intermediate, or result table). | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| outputs\tables\pooled_cluster_composition.csv | CSV data artifact (source, intermediate, or result table). | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| outputs\tables\spatial_clustering_classification.csv | CSV data artifact (source, intermediate, or result table). | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| outputs\tables\spatial_summary.csv | CSV data artifact (source, intermediate, or result table). | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| outputs\tables\statistical_tests_summary.csv | CSV data artifact (source, intermediate, or result table). | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| outputs\validation\plots\chicago_kmeans_pca_scatter.png | Generated visualization image for analysis/reporting. | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| outputs\validation\plots\demand_score_vs_price_by_cluster_city_facet.png | Generated visualization image for analysis/reporting. | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| outputs\validation\plots\neighbourhood_mean_price_vs_cv_scatter.png | Generated visualization image for analysis/reporting. | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| outputs\validation\plots\neighbourhood_top15_expensive_by_city.png | Generated visualization image for analysis/reporting. | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| outputs\validation\plots\newyork_kmeans_pca_scatter.png | Generated visualization image for analysis/reporting. | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| outputs\validation\plots\pooled_kmeans_cluster_composition_by_city.png | Generated visualization image for analysis/reporting. | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| outputs\validation\plots\regression_rf_feature_importance_by_city.png | Generated visualization image for analysis/reporting. | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| outputs\validation\plots\violin_price_by_room_type_city_comparison.png | Generated visualization image for analysis/reporting. | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| outputs\validation\tables\cluster_summary.csv | CSV data artifact (source, intermediate, or result table). | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| outputs\validation\tables\pooled_cluster_composition.csv | CSV data artifact (source, intermediate, or result table). | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| outputs\validation\tables\statistical_tests_summary.csv | CSV data artifact (source, intermediate, or result table). | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| README.md | Project overview and execution/report documentation. | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| requirements.txt | Python dependency list for reproducible environment setup. | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| run_pipeline.py | CLI entrypoint to execute the complete pipeline. | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| src\__init__.py | Python source or utility script in project workflow. | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| src\calendar_analysis.py | Python source or utility script in project workflow. | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| src\clustering.py | Python source or utility script in project workflow. | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| src\config.py | Python source or utility script in project workflow. | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| src\data_loader.py | Python source or utility script in project workflow. | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| src\feature_engineering.py | Python source or utility script in project workflow. | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| src\multi_city_analysis.py | Python source or utility script in project workflow. | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| src\pipeline.py | Python source or utility script in project workflow. | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| src\preprocessing.py | Python source or utility script in project workflow. | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| src\regression_analysis.py | Python source or utility script in project workflow. | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| src\spatial_analysis.py | Python source or utility script in project workflow. | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| src\statistical_tests.py | Python source or utility script in project workflow. | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| src\visualization.py | Python source or utility script in project workflow. | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| templates\dashboard.html | Interactive HTML artifact for map/dashboard usage. | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |
| validate_fixes.py | Smoke-test script validating core analytics modules. | Supports the overall analytical workflow either as input, intermediate artifact, or final output evidence. |

