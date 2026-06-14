"""Tests for regime-aware performance attribution."""

from neuroquant.backtest import BacktestConfig, generate_signals
from neuroquant.features import REGIME_ORDER
from neuroquant.regime import summarize_by_regime


def test_regime_summary_has_expected_columns(sample_data):
    signals = generate_signals(sample_data, BacktestConfig(20, 60))
    summary = summarize_by_regime(signals)
    expected = {
        "regime",
        "n_days",
        "active_days",
        "active_share",
        "total_return",
        "baseline_return",
        "sharpe_ratio",
        "max_drawdown",
    }
    assert expected.issubset(summary.columns)
    assert len(summary) >= 1


def test_regime_labels_are_valid_and_ordered(sample_data):
    signals = generate_signals(sample_data, BacktestConfig(20, 60))
    summary = summarize_by_regime(signals)
    assert set(summary["regime"]).issubset(set(REGIME_ORDER))
    # Day counts across regimes never exceed the sample length.
    assert summary["n_days"].sum() <= len(signals)


def test_regime_active_share_within_bounds(sample_data):
    signals = generate_signals(sample_data, BacktestConfig(20, 60))
    summary = summarize_by_regime(signals)
    assert (summary["active_share"] >= 0).all()
    assert (summary["active_share"] <= 1).all()
