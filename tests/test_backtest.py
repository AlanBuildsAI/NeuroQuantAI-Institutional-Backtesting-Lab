"""Tests for the backtest engine and parameter sweep."""

import pytest

from neuroquant.backtest import (
    BacktestConfig,
    build_candidate_configs,
    config_from_row,
    generate_signals,
    run_config_sweep,
    run_parameter_sweep,
)
from neuroquant.signals import SIGNAL_FAMILIES
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


def test_signals_contain_expected_columns(sample_data):
    signals = generate_signals(sample_data, BacktestConfig(20, 60))
    expected = {
        "position",
        "market_return",
        "strategy_return",
        "baseline_return",
        "strategy_equity",
        "baseline_equity",
        "cost",
    }
    assert expected.issubset(signals.columns)


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


@pytest.mark.parametrize("family", SIGNAL_FAMILIES)
def test_backtest_runs_each_signal_family(sample_data, family):
    signals = generate_signals(sample_data, BacktestConfig(signal_family=family))
    # First position is flat (shifted) and equity is well-formed.
    assert signals["position"].iloc[0] == 0.0
    assert signals["strategy_equity"].notna().all()


def test_config_sweep_spans_families_and_sorts(sample_data):
    configs = build_candidate_configs()
    summary = run_config_sweep(sample_data, configs)
    assert {"label", "signal_family", "sharpe_ratio"}.issubset(summary.columns)
    # More than one family is represented in the candidate set.
    assert summary["signal_family"].nunique() >= 3
    sharpes = summary["sharpe_ratio"].tolist()
    assert sharpes == sorted(sharpes, reverse=True)


def test_config_from_row_roundtrip(sample_data):
    configs = build_candidate_configs()
    summary = run_config_sweep(sample_data, configs)
    best = config_from_row(summary.iloc[0])
    assert best.signal_family in SIGNAL_FAMILIES
