"""NeuroQuantAI — Synthetic Quant Research & Analytics Lab.

A reproducible Python research workflow built around a synthetic signal
series. It demonstrates feature engineering, multiple candidate signal
families, data validation, benchmark comparison, in/out-of-sample testing,
walk-forward validation, Monte Carlo robustness analysis, regime-aware
attribution, cost-sensitivity and stress diagnostics, KPI reporting, and
decision-ready visual reporting.

This is a quant-inspired RESEARCH DEMONSTRATION on synthetic data by default
(optional local CSV research data is supported). It is not a trading system,
not financial advice, not an investment recommendation, and makes no
performance guarantees.
"""

from .data import (
    generate_synthetic_series,
    load_csv_series,
    load_csv_frame,
    data_quality_report,
)
from .validation import (
    validate_price_frame,
    validate_window_config,
    ValidationError,
)
from .features import build_feature_frame, add_regime_labels
from .signals import build_signal_frame, SIGNAL_FAMILIES
from .costs import CostRates, resolve_cost_rates
from .sizing import compute_exposure, SIZING_METHODS
from .risk import (
    apply_risk_controls,
    apply_drawdown_guard,
    rolling_volatility,
    rolling_sharpe,
    rolling_drawdown,
)
from .backtest import (
    run_backtest,
    run_parameter_sweep,
    run_config_sweep,
    build_candidate_configs,
    BacktestConfig,
)
from .metrics import compute_kpis, compute_extended_kpis
from .research import (
    split_train_test,
    walk_forward_validation,
    monte_carlo_bootstrap,
    overfit_gap,
    robustness_score,
)
from .regime import summarize_by_regime
from .stress import cost_sensitivity_analysis, stress_test_summary

__version__ = "2.0.0"

# Note: `pipeline` is intentionally NOT imported here. Eagerly importing it
# would load the module during package init, which triggers a RuntimeWarning
# when running `python -m neuroquant.pipeline`. Import it directly instead:
#   from neuroquant.pipeline import run_pipeline

__all__ = [
    "generate_synthetic_series",
    "load_csv_series",
    "load_csv_frame",
    "data_quality_report",
    "validate_price_frame",
    "validate_window_config",
    "ValidationError",
    "build_feature_frame",
    "add_regime_labels",
    "build_signal_frame",
    "SIGNAL_FAMILIES",
    "CostRates",
    "resolve_cost_rates",
    "compute_exposure",
    "SIZING_METHODS",
    "apply_risk_controls",
    "apply_drawdown_guard",
    "rolling_volatility",
    "rolling_sharpe",
    "rolling_drawdown",
    "run_backtest",
    "run_parameter_sweep",
    "run_config_sweep",
    "build_candidate_configs",
    "BacktestConfig",
    "compute_kpis",
    "compute_extended_kpis",
    "split_train_test",
    "walk_forward_validation",
    "monte_carlo_bootstrap",
    "overfit_gap",
    "robustness_score",
    "summarize_by_regime",
    "cost_sensitivity_analysis",
    "stress_test_summary",
]
