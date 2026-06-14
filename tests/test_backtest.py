"""Tests for the backtest engine and parameter sweep."""

import pytest

from neuroquant.backtest import (
    BacktestConfig,
    generate_signals,
    run_parameter_sweep,
)
from neuroquant.validation import ValidationError

SWEEP_COLUMNS = {"short_window", "long_window", "cost_per_trade"}


def test_backtest_rejects_invalid_window_config(sample_data):
    with pytest.raises(ValidationError, match="strictly less"):
        generate_signals(sample_data, BacktestConfig(short_window=60, long_window=20))


def test_signals_are_shifted_to_avoid_lookahead(sample_data):
    signals = generate_signals(sample_data, BacktestConfig(20, 60))
    # The first non-NaN crossover cannot produce a position before the slow
    # window is filled; the very first position must be flat (shifted).
    assert signals["position"].iloc[0] == 0.0


def test_parameter_sweep_returns_expected_columns(sample_data):
    summary = run_parameter_sweep(
        sample_data, short_windows=[10, 20], long_windows=[60, 90]
    )
    assert SWEEP_COLUMNS.issubset(summary.columns)
    assert {"sharpe_ratio", "total_return", "max_drawdown"}.issubset(summary.columns)


def test_parameter_sweep_skips_invalid_combinations(sample_data):
    # short=60 >= long=40 should be skipped, leaving only valid 20/60.
    summary = run_parameter_sweep(
        sample_data, short_windows=[20, 60], long_windows=[40, 60]
    )
    # Only combos where short < long: 20/40, 20/60. (60/40, 60/60 skipped.)
    assert len(summary) == 2


def test_parameter_sweep_sorted_by_sharpe(sample_data):
    summary = run_parameter_sweep(
        sample_data, short_windows=[5, 10, 20], long_windows=[60, 90, 120]
    )
    sharpes = summary["sharpe_ratio"].tolist()
    assert sharpes == sorted(sharpes, reverse=True)
