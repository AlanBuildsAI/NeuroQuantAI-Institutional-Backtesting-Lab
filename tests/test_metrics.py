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
