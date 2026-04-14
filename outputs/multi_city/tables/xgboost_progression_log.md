# XGBoost Model Progression Log

This file preserves model progression without overwriting prior outputs.

## Stages

- Stage 0: previously saved metrics from outputs/multi_city/tables/xgboost_city_metrics.csv

- Stage 1: enhanced feature/tuning pipeline with random split (listing_id removed to force random split)

- Stage 2: enhanced pipeline with grouped split by listing_id (leakage-resistant)

## R2 Progression

city,stage0_old_r2,stage1_random_r2,stage2_grouped_r2,delta_old_to_stage1,delta_old_to_stage2
austin,0.4163328917460372,0.5597987896300313,0.559543485022298,0.14346589788399405,0.1432105932762608
chicago,0.6835510594477411,0.5870169385912626,0.5604173027644561,-0.0965341208564785,-0.12313375668328497
losangeles,0.4780225671563355,0.999999983950146,0.9999999713728623,0.5219774167938105,0.5219774042165268
nashville,0.2894823891058588,0.4045823445321324,0.4077590931196716,0.1150999554262736,0.11827670401381285
neworleans,0.7880986814770712,0.8701822599807514,0.7192021624309215,0.0820835785036802,-0.06889651904614968
newyork,0.3927486038386534,0.5676052203715681,0.5662145365859748,0.17485661653291473,0.17346593274732136

## Notes

- Existing output tables/plots were not overwritten by this log.

- Los Angeles remains anomalously high and should be investigated for city-specific leakage or target proxy behavior in source features.
