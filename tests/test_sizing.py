"""Tests for position-sizing methods (causal, conservative)."""

import numpy as np
import pandas as pd

from neuroquant.backtest import BacktestConfig
from neuroquant.sizing import compute_exposure, fixed_fraction, fixed_unit


def _series(values):
    idx = pd.bdate_range("2021-01-01", periods=len(values))
    return pd.Series(values, index=idx)


def test_fixed_unit_is_identity():
    target = _series([0.0, 1.0, 1.0, 0.0])
    assert fixed_unit(target).equals(target)


def test_fixed_fraction_scales():
    target = _series([0.0, 1.0, 1.0])
    assert fixed_fraction(target, 0.5).tolist() == [0.0, 0.5, 0.5]


def test_default_sizing_matches_raw_signal(sample_data):
    """fixed_unit default keeps the original 0/1 exposure (clipped no-op)."""
    target = _series([0.0, 1.0, 1.0, 0.0, 1.0])
    returns = _series([0.0, 0.01, -0.02, 0.0, 0.03])
    out = compute_exposure(target, returns, BacktestConfig())
    assert out.tolist() == target.tolist()


def test_max_exposure_and_no_short_clip():
    target = _series([1.0, 1.0, -1.0])
    returns = _series([0.0, 0.0, 0.0])
    cfg = BacktestConfig(sizing_method="fixed_fraction", fixed_fraction=2.0,
                         max_exposure=1.0, allow_short=False)
    out = compute_exposure(target, returns, cfg)
    # Clipped to [0, 1]: no leverage, no shorting.
    assert out.max() <= 1.0 + 1e-9
    assert out.min() >= 0.0


def test_volatility_target_reduces_in_high_vol():
    target = _series([1.0] * 80)
    # Higher-volatility returns should pull target exposure below 1.
    rng = np.random.default_rng(0)
    returns = _series(rng.normal(0, 0.03, 80))
    cfg = BacktestConfig(sizing_method="volatility_target", target_volatility=0.10,
                         vol_lookback=20, max_exposure=1.0)
    out = compute_exposure(target, returns, cfg).dropna()
    assert (out <= 1.0 + 1e-9).all()
    assert out.mean() < 1.0


def test_volatility_target_is_causal():
    """Exposure on a prefix matches the full series (no look-ahead)."""
    rng = np.random.default_rng(1)
    returns = _series(rng.normal(0, 0.02, 200))
    target = _series([1.0] * 200)
    cfg = BacktestConfig(sizing_method="volatility_target")
    full = compute_exposure(target, returns, cfg)
    prefix = compute_exposure(target.iloc[:120], returns.iloc[:120], cfg)
    a, b = full.iloc[:120], prefix
    mask = a.notna() & b.notna()
    assert mask.sum() > 0
    assert np.allclose(a[mask].to_numpy(), b[mask].to_numpy())
