"""Tests for the candidate signal families."""

import pytest

from neuroquant.backtest import BacktestConfig
from neuroquant.signals import (
    SIGNAL_FAMILIES,
    build_signal_frame,
)
from neuroquant.validation import ValidationError


@pytest.mark.parametrize("family", SIGNAL_FAMILIES)
def test_each_family_returns_target_position(sample_data, family):
    config = BacktestConfig(signal_family=family)
    frame = build_signal_frame(sample_data, config)
    assert "target_position" in frame.columns
    # Target position is long-or-flat (0/1) where defined.
    defined = frame["target_position"].dropna()
    assert set(defined.unique()).issubset({0.0, 1.0})


def test_composite_combines_multiple_signals(sample_data):
    config = BacktestConfig(signal_family="composite")
    frame = build_signal_frame(sample_data, config)
    for col in ("trend_signal", "momentum_signal", "mean_reversion_signal",
                "composite_score"):
        assert col in frame.columns
    # The composite score is the mean of three 0/1 signals -> in {0,1/3,2/3,1}.
    score = frame["composite_score"].dropna()
    assert score.min() >= 0.0 and score.max() <= 1.0


def test_volatility_filter_adds_risk_on_and_reduces_exposure(sample_data):
    base = BacktestConfig(signal_family="trend", use_volatility_filter=False)
    filtered = BacktestConfig(signal_family="trend", use_volatility_filter=True)
    base_frame = build_signal_frame(sample_data, base)
    filt_frame = build_signal_frame(sample_data, filtered)
    assert "risk_on" in filt_frame.columns
    # Filtering can only remove exposure, never add it.
    assert filt_frame["target_position"].sum() <= base_frame["target_position"].sum()


def test_unknown_family_rejected(sample_data):
    with pytest.raises(ValidationError, match="signal_family"):
        build_signal_frame(sample_data, BacktestConfig(signal_family="nope"))
