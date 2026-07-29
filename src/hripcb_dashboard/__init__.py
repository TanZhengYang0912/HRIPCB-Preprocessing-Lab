"""Generic static dashboard helpers for preprocessing experiments."""

from .dashboard import write_dashboard_html
from .filtering import (
    best_by_module,
    best_experiment,
    collapse_shared_baseline,
    comparison_records,
    filter_records,
    is_combined_record,
    normalize_selection,
    option_values,
    reset_selection_state,
)

__all__ = [
    "best_by_module",
    "best_experiment",
    "collapse_shared_baseline",
    "comparison_records",
    "filter_records",
    "is_combined_record",
    "normalize_selection",
    "option_values",
    "reset_selection_state",
    "write_dashboard_html",
]
