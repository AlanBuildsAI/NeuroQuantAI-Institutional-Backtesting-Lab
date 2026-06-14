"""NeuroQuant analytics case study package.

A small, reproducible Python analytics workflow built around a synthetic
time series. It demonstrates data validation, experiment design, KPI
metric construction, scenario comparison, and decision-ready reporting.

This is an ANALYTICS DEMONSTRATION using synthetic data only. It is not a
trading system and contains no financial advice or market predictions.
"""

from .data import generate_synthetic_series
from .validation import (
    validate_price_frame,
    validate_window_config,
    ValidationError,
)
from .backtest import (
    run_backtest,
    run_parameter_sweep,
    BacktestConfig,
)
from .metrics import compute_kpis

__version__ = "1.0.0"

# Note: `pipeline` is intentionally NOT imported here. Eagerly importing it
# would load the module during package init, which triggers a RuntimeWarning
# when running `python -m neuroquant.pipeline`. Import it directly instead:
#   from neuroquant.pipeline import run_pipeline

__all__ = [
    "generate_synthetic_series",
    "validate_price_frame",
    "validate_window_config",
    "ValidationError",
    "run_backtest",
    "run_parameter_sweep",
    "BacktestConfig",
    "compute_kpis",
]
