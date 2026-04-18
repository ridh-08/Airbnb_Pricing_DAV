from pathlib import Path
import pandas as pd
from src.config import CITY_FOLDERS, PROCESSED_DIR, get_city_output_dir
from src.regression_analysis import run_regression_analysis
from src.pipeline import _save_regression_outputs

for city in sorted(CITY_FOLDERS.keys()):
    path = PROCESSED_DIR / f"{city}_clustered.csv"
    if not path.exists():
        print(f"SKIP {city}: missing {path}")
        continue
    df = pd.read_csv(path)
    out = get_city_output_dir(city)
    plots = out / "plots"
    tables = out / "tables"
    plots.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    res = run_regression_analysis(df, output_dir=plots, tables_output_dir=tables, city_name=city)
    _save_regression_outputs(res, city, tables)
    print(f"DONE {city}")

print("SHAP regression generation done")
