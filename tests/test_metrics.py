"""Tests for the KPI metrics."""

import pytest

from neuroquant.backtest import BacktestConfig, generate_signals
from neuroquant.metrics import compute_kpis

EXPECTED_KEYS = {
    "total_return",
    "baseline_return",
    "excess_return",
    "annualized_return",
    "annualized_volatility",
    "sharpe_ratio",
    "sortino_ratio",
    "calmar_ratio",
    "information_ratio",
    "max_drawdown",
    "active_days",
    "trade_count",
    "turnover",
    "correlation_to_baseline",
    "win_loss_ratio",
}


def test_metrics_contains_expected_keys(sample_data):
    signals = generate_signals(sample_data, BacktestConfig(20, 60))
    kpis = compute_kpis(signals)
    assert set(kpis.keys()) == EXPECTED_KEYS


def test_metrics_values_are_numeric(sample_data):
    signals = generate_signals(sample_data, BacktestConfig(20, 60))
    kpis = compute_kpis(signals)
    for value in kpis.values():
        assert isinstance(value, (int, float))


def test_max_drawdown_is_non_positive(sample_data):
    signals = generate_signals(sample_data, BacktestConfig(20, 60))
    kpis = compute_kpis(signals)
    assert kpis["max_drawdown"] <= 0.0


def test_excess_return_is_strategy_minus_baseline(sample_data):
    signals = generate_signals(sample_data, BacktestConfig(20, 60))
    kpis = compute_kpis(signals)
    assert kpis["excess_return"] == pytest.approx(
        kpis["total_return"] - kpis["baseline_return"]
    )


def test_extended_kpis_adds_quant_metrics(sample_data):
    from neuroquant.metrics import compute_extended_kpis

    signals = generate_signals(sample_data, BacktestConfig(20, 60, cost_per_trade=0.005))
    ext = compute_extended_kpis(signals)
    for key in (
        "cagr", "exposure_avg", "exposure_max", "best_day", "worst_day",
        "win_rate", "gross_return", "cost_drag", "rolling_sharpe_median",
        "benchmark_return", "benchmark_excess",
    ):
        assert key in ext
    # With positive costs, gross return exceeds net and cost drag is positive.
    assert ext["gross_return"] >= ext["total_return"] - 1e-9
    assert ext["cost_drag"] >= -1e-9


def test_extended_kpis_benchmark_nan_without_benchmark(sample_data):
    from neuroquant.metrics import compute_extended_kpis

    signals = generate_signals(sample_data, BacktestConfig(20, 60))
    ext = compute_extended_kpis(signals)
    assert ext["benchmark_return"] != ext["benchmark_return"]  # NaN
