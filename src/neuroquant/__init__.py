"""NeuroQuantAI — Synthetic Quant Research & Analytics Lab.

A reproducible Python research workflow built around a synthetic signal
series. It demonstrates data validation, signal evaluation, benchmark
comparison, in/out-of-sample testing, walk-forward validation, Monte Carlo
robustness analysis, KPI reporting, and decision-ready visual reporting.

This is a quant-inspired RESEARCH DEMONSTRATION on synthetic data by default
(optional local CSV research data is supported). It is not a trading system,
not financial advice, not an investment recommendation, and makes no
performance guarantees.
"""

from .data import generate_synthetic_series, load_csv_series
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
from .research import (
    split_train_test,
    walk_forward_validation,
    monte_carlo_bootstrap,
)

__version__ = "1.1.0"

# Note: `pipeline` is intentionally NOT imported here. Eagerly importing it
# would load the module during package init, which triggers a RuntimeWarning
# when running `python -m neuroquant.pipeline`. Import it directly instead:
#   from neuroquant.pipeline import run_pipeline

__all__ = [
    "generate_synthetic_series",
    "load_csv_series",
    "validate_price_frame",
    "validate_window_config",
    "ValidationError",
    "run_backtest",
    "run_parameter_sweep",
    "BacktestConfig",
    "compute_kpis",
    "split_train_test",
    "walk_forward_validation",
    "monte_carlo_bootstrap",
]
