"""Feature engineering for the research lab.

These are small, explainable, **causal** features: every value at time ``t``
is computed from information available up to and including ``t`` (rolling and
backward-looking only). No feature uses a forward shift, so building the
feature frame introduces no look-ahead. When a feature is later used to drive a
position it is shifted by one bar in the backtest engine, adding a further
safety margin.

The features are deliberately interpretable (returns, trend distance,
momentum, volatility, drawdown, volatility regime) rather than opaque — the
point of the lab is honest, auditable research, not a black box.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .validation import validate_price_frame

DEFAULT_TREND_WINDOWS = (10, 20, 60)
DEFAULT_MOMENTUM_WINDOWS = (10, 20, 60)
DEFAULT_VOL_WINDOWS = (20, 60)

# Ordered volatility-regime labels, calm → turbulent.
REGIME_ORDER = ("low", "normal", "high", "stress")


def add_return_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add simple and log one-bar returns of ``close``."""
    validate_price_frame(frame, min_rows=2)
    out = frame.copy()
    out["return_1d"] = out["close"].pct_change()
    out["log_return"] = np.log(out["close"]).diff()
    return out


def add_trend_features(
    frame: pd.DataFrame, windows: tuple[int, ...] = DEFAULT_TREND_WINDOWS
) -> pd.DataFrame:
    """Add moving averages, price-to-MA distance, and rolling z-scores.

    For each window ``w``:
      * ``ma_{w}`` — simple moving average of ``close``,
      * ``price_to_ma_{w}`` — ``close / ma_{w} - 1`` (how far above/below trend),
      * ``zscore_{w}`` — ``(close - ma_{w}) / rolling_std(close, w)``.
    """
    out = frame.copy()
    for w in windows:
        ma = out["close"].rolling(w).mean()
        std = out["close"].rolling(w).std(ddof=0)
        out[f"ma_{w}"] = ma
        out[f"price_to_ma_{w}"] = out["close"] / ma - 1.0
        out[f"zscore_{w}"] = (out["close"] - ma) / std.replace(0.0, np.nan)
    return out


def add_momentum_features(
    frame: pd.DataFrame, windows: tuple[int, ...] = DEFAULT_MOMENTUM_WINDOWS
) -> pd.DataFrame:
    """Add multi-horizon momentum (``close`` return over each window)."""
    out = frame.copy()
    for w in windows:
        out[f"momentum_{w}"] = out["close"].pct_change(w)
    return out


def add_volatility_features(
    frame: pd.DataFrame, windows: tuple[int, ...] = DEFAULT_VOL_WINDOWS
) -> pd.DataFrame:
    """Add rolling volatility and drawdown-from-rolling-high features.

    For each window ``w``:
      * ``vol_{w}`` — rolling standard deviation of one-bar returns,
      * ``drawdown_from_high_{w}`` — ``close / rolling_max(close, w) - 1``.
    """
    out = frame.copy()
    returns = out["close"].pct_change()
    for w in windows:
        out[f"vol_{w}"] = returns.rolling(w).std(ddof=0)
        roll_high = out["close"].rolling(w, min_periods=1).max()
        out[f"drawdown_from_high_{w}"] = out["close"] / roll_high - 1.0
    return out


def add_regime_labels(
    frame: pd.DataFrame, volatility_window: int = 60
) -> pd.DataFrame:
    """Label each bar with a volatility regime: low / normal / high / stress.

    Regimes are derived from the rolling volatility of returns using
    **full-series quantile cutoffs** (33rd, 66th, 90th percentiles). This is a
    *descriptive* labelling of the realised series for post-hoc performance
    attribution — it is not used as a trading signal, so the full-series
    quantiles are appropriate. (The tradeable volatility *filter* in
    :mod:`neuroquant.signals` instead uses a trailing, leakage-free threshold.)

    Adds a ``vol_regime`` categorical column ordered low → stress. Warm-up rows
    without enough data to compute volatility are left as ``NaN``.
    """
    out = frame.copy()
    returns = out["close"].pct_change()
    vol = returns.rolling(volatility_window).std(ddof=0)
    out["regime_vol"] = vol

    valid = vol.dropna()
    if valid.empty:
        out["vol_regime"] = pd.Categorical(
            [np.nan] * len(out), categories=REGIME_ORDER, ordered=True
        )
        return out

    q33, q66, q90 = valid.quantile([0.33, 0.66, 0.90])

    def _label(v: float):
        if np.isnan(v):
            return np.nan
        if v <= q33:
            return "low"
        if v <= q66:
            return "normal"
        if v <= q90:
            return "high"
        return "stress"

    out["vol_regime"] = pd.Categorical(
        vol.map(_label), categories=REGIME_ORDER, ordered=True
    )
    return out


def build_feature_frame(
    frame: pd.DataFrame,
    trend_windows: tuple[int, ...] = DEFAULT_TREND_WINDOWS,
    momentum_windows: tuple[int, ...] = DEFAULT_MOMENTUM_WINDOWS,
    vol_windows: tuple[int, ...] = DEFAULT_VOL_WINDOWS,
    regime_window: int = 60,
) -> pd.DataFrame:
    """Compose every feature block into one chronological feature frame.

    The returned frame keeps the original ``close`` and index and adds all
    return, trend, momentum, volatility, drawdown and regime columns. Early
    rows contain ``NaN`` warm-up values by construction; downstream code is
    responsible for respecting the warm-up period.
    """
    validate_price_frame(frame, min_rows=2)
    out = add_return_features(frame)
    out = add_trend_features(out, trend_windows)
    out = add_momentum_features(out, momentum_windows)
    out = add_volatility_features(out, vol_windows)
    out = add_regime_labels(out, regime_window)
    return out
