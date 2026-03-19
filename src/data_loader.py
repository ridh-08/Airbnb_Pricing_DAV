from __future__ import annotations

from typing import Any

from src.preprocessing import load_all_cities


def load_project_data(include_calendar: bool = False) -> dict[str, dict[str, Any]]:
    """Convenience wrapper for loading Airbnb city datasets."""
    return load_all_cities(include_calendar=include_calendar)
