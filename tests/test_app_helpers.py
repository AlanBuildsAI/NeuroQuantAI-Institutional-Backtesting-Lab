"""Tests for the Streamlit demo's UI-agnostic helper logic."""

import io

import pandas as pd
import pytest

from neuroquant.app_helpers import (
    ResearchParams,
    drawdown_frame,
    equity_frame,
    load_uploaded_csv,
    make_synthetic,
    run_research,
    scorecard_frame,
)

RESULT_KEYS = {
    "config",
    "label",
    "n_obs",
    "train_size",
    "test_size",
    "full_signals",
    "full_kpis",
    "in_sample_kpis",
    "out_of_sample_kpis",
    "walk_forward",
    "walk_forward_message",
    "monte_carlo",
    "regime_summary",
    "cost_sensitivity",
    "stress_summary",
    "takeaway",
}


def test_params_to_config_roundtrip():
    params = ResearchParams(signal_family="momentum", momentum_window=30)
    config = params.to_config()
    assert config.signal_family == "momentum"
    assert config.momentum_window == 30


def test_run_research_returns_expected_structure():
    frame = make_synthetic(n_days=600, seed=1)
    result = run_research(frame, ResearchParams(mc_simulations=150))
    assert RESULT_KEYS.issubset(result.keys())
    # Monte Carlo respects the requested simulation count.
    assert len(result["monte_carlo"]["total_returns"]) == 150
    # Walk-forward ran (enough data) and split sizes add up.
    assert result["walk_forward"] is not None
    assert result["train_size"] + result["test_size"] == result["n_obs"]


@pytest.mark.parametrize("family", ["trend", "momentum", "mean_reversion", "composite"])
def test_run_research_each_family(family):
    frame = make_synthetic(n_days=500, seed=3)
    result = run_research(frame, ResearchParams(signal_family=family, mc_simulations=100))
    assert result["config"].signal_family == family
    assert not result["cost_sensitivity"].empty


def test_run_research_too_short_raises():
    frame = make_synthetic(n_days=70, seed=2)
    with pytest.raises(ValueError):
        run_research(frame, ResearchParams(long_window=60, train_fraction=0.7))


def test_walk_forward_message_when_short_but_runnable():
    # 320 rows: full run works, walk-forward still produces folds.
    frame = make_synthetic(n_days=320, seed=5)
    result = run_research(frame, ResearchParams(mc_simulations=100))
    assert result["walk_forward"] is not None or result["walk_forward_message"]


def test_scorecard_frame_columns():
    frame = make_synthetic(n_days=500, seed=4)
    result = run_research(frame, ResearchParams(mc_simulations=100))
    table = scorecard_frame(result["in_sample_kpis"], result["out_of_sample_kpis"])
    assert list(table.columns) == ["Metric", "In-sample", "Out-of-sample"]
    assert len(table) >= 8


def test_equity_and_drawdown_frames():
    frame = make_synthetic(n_days=400, seed=6)
    result = run_research(frame, ResearchParams(mc_simulations=100))
    eq = equity_frame(result["full_signals"])
    dd = drawdown_frame(result["full_signals"])
    assert list(eq.columns) == ["Strategy", "Baseline"]
    assert list(dd.columns) == ["Drawdown"]
    assert len(eq) == result["n_obs"]
    # Drawdown is never positive.
    assert (dd["Drawdown"] <= 1e-9).all()


def test_load_uploaded_csv_valid():
    buffer = io.StringIO(
        "date,close\n2021-01-01,100\n2021-01-02,101\n2021-01-03,102\n"
    )
    frame = load_uploaded_csv(buffer)
    assert "close" in frame.columns
    assert isinstance(frame.index, pd.DatetimeIndex)
    assert len(frame) == 3


def test_load_uploaded_csv_rejects_missing_close():
    buffer = io.StringIO("date,price\n2021-01-01,100\n2021-01-02,101\n")
    with pytest.raises(ValueError, match="close"):
        load_uploaded_csv(buffer)
