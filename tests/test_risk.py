"""Tests for risk diagnostics and optional risk controls."""

import numpy as np
import pandas as pd

from neuroquant.backtest import BacktestConfig
from neuroquant.risk import (
    apply_drawdown_guard,
    apply_risk_controls,
    apply_volatility_cap,
    rolling_drawdown,
    rolling_sharpe,
    rolling_volatility,
)


def _series(values):
    idx = pd.bdate_range("2021-01-01", periods=len(values))
    return pd.Series(values, index=idx)


def test_rolling_diagnostics_shapes():
    rng = np.random.default_rng(0)
    returns = _series(rng.normal(0, 0.01, 100))
    assert len(rolling_volatility(returns, 20)) == 100
    assert len(rolling_sharpe(returns, 30)) == 100
    equity = (1 + returns).cumprod()
    dd = rolling_drawdown(equity)
    assert (dd <= 1e-9).all()


def test_risk_controls_off_by_default_is_noop():
    exposure = _series([1.0] * 50)
    returns = _series([0.02] * 50)
    out = apply_risk_controls(exposure, returns, BacktestConfig())
    assert out.equals(exposure)


def test_volatility_cap_only_reduces_exposure():
    exposure = _series([1.0] * 60)
    rng = np.random.default_rng(2)
    returns = _series(rng.normal(0, 0.04, 60))
    capped = apply_volatility_cap(exposure, returns, vol_cap=0.15, lookback=20)
    assert (capped <= exposure + 1e-9).all()


def test_drawdown_guard_flattens_after_breach_and_is_monotone():
    # A steadily declining market: the guard should flatten exposure.
    position = _series([1.0] * 12)
    market = _series([-0.05] * 12)
    guarded = apply_drawdown_guard(position, market, level=0.10)
    # Guard never adds exposure beyond the base.
    assert (guarded <= position + 1e-9).all()
    # After the drawdown breach the later exposure is flat.
    assert guarded.iloc[-1] == 0.0


def test_drawdown_guard_noop_when_no_breach():
    position = _series([1.0] * 10)
    market = _series([0.01] * 10)  # rising — never breaches
    guarded = apply_drawdown_guard(position, market, level=0.20)
    assert guarded.tolist() == position.tolist()
