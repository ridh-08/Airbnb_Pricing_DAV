from __future__ import annotations

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = BASE_DIR / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"
TABLES_DIR = OUTPUTS_DIR / "tables"

CITY_FOLDERS: dict[str, str] = {
    "newyork": "New York",
    "chicago": "Chicago",
    "nashville": "Nashville",
    "neworleans": "New Orleans",
    "austin": "Austin",
    "losangeles": "Los Angeles",
}

CITY_COLORS: dict[str, str] = {
    "newyork": "#FF5A5F",
    "chicago": "#00A699",
    "nashville": "#FC642D",
    "neworleans": "#7B0051",
    "austin": "#007A87",
    "losangeles": "#484848",
}


def normalize_city_name(city_name: str) -> str:
    return city_name.lower().strip().replace(" ", "")


def get_city_output_dir(city_name: str, root_dir: Path | None = None) -> Path:
    root = root_dir or OUTPUTS_DIR
    city_key = normalize_city_name(city_name)
    out_dir = root / city_key
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def discover_processed_city_files(suffix: str) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for city_key in CITY_FOLDERS:
        path = PROCESSED_DIR / f"{city_key}_{suffix}.csv"
        if path.exists():
            files[city_key] = path
    return files
