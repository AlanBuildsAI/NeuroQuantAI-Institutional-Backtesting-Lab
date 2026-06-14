"""Candidate signal families.

Each family turns a price/feature frame into an **unshifted target position**
in ``{0, 1}`` (long-or-flat). The backtest engine shifts that target by one bar
before it is traded, so look-ahead bias is handled centrally in one place.

Families implemented:

* ``trend``          — moving-average crossover (fast above slow ⇒ long),
* ``momentum``       — positive trailing momentum ⇒ long,
* ``mean_reversion`` — long when oversold (z-score below a negative threshold),
* ``composite``      — average of the three signals above, thresholded.

An optional **volatility filter** can damp exposure during turbulent periods.
Its high-volatility threshold is an *expanding* (trailing) quantile, so it uses
only past and present data — no future information leaks into the decision.

"Composite" is used in the literal sense: it combines several real, simple
signals into a transparent score. There is no machine-learning model here, and
nothing in this module is described as one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .validation import ValidationError, validate_price_frame

SIGNAL_FAMILIES = ("trend", "momentum", "mean_reversion", "composite")


def required_warmup(config) -> int:
    """Largest look-back (in bars) a config needs before signals are valid."""
    windows = [config.long_window, config.momentum_window, config.zscore_window]
    if getattr(config, "use_volatility_filter", False):
        windows.append(config.vol_filter_window)
    return int(max(windows))


def _trend_position(frame: pd.DataFrame, config) -> pd.DataFrame:
    """Fast/slow moving-average crossover (long when fast > slow)."""
    out = pd.DataFrame(index=frame.index)
    out["fast_ma"] = frame["close"].rolling(config.short_window).mean()
    out["slow_ma"] = frame["close"].rolling(config.long_window).mean()
    out["target_position"] = (out["fast_ma"] > out["slow_ma"]).astype(float)
    # Flatten the warm-up region where either MA is undefined.
    out.loc[out["slow_ma"].isna(), "target_position"] = np.nan
    return out


def _momentum_position(frame: pd.DataFrame, config) -> pd.DataFrame:
    """Long when trailing momentum over ``momentum_window`` is positive."""
    out = pd.DataFrame(index=frame.index)
    out["momentum"] = frame["close"].pct_change(config.momentum_window)
    out["target_position"] = (out["momentum"] > 0).astype(float)
    out.loc[out["momentum"].isna(), "target_position"] = np.nan
    return out


def _mean_reversion_position(frame: pd.DataFrame, config) -> pd.DataFrame:
    """Long when the price is oversold: z-score below ``-zscore_entry``."""
    out = pd.DataFrame(index=frame.index)
    ma = frame["close"].rolling(config.zscore_window).mean()
    std = frame["close"].rolling(config.zscore_window).std(ddof=0)
    out["zscore"] = (frame["close"] - ma) / std.replace(0.0, np.nan)
    out["target_position"] = (out["zscore"] < -config.zscore_entry).astype(float)
    out.loc[out["zscore"].isna(), "target_position"] = np.nan
    return out


def _composite_position(frame: pd.DataFrame, config) -> pd.DataFrame:
    """Average the three base signals into a 0–1 score, then threshold at 0.5.

    Each base family contributes its own ``{0, 1}`` target; the composite score
    is their mean (so it lives in ``{0, 1/3, 2/3, 1}``) and the final position
    goes long when at least half of the base signals agree.
    """
    trend = _trend_position(frame, config)["target_position"]
    momentum = _momentum_position(frame, config)["target_position"]
    reversion = _mean_reversion_position(frame, config)["target_position"]

    out = pd.DataFrame(index=frame.index)
    out["trend_signal"] = trend
    out["momentum_signal"] = momentum
    out["mean_reversion_signal"] = reversion
    # Score is defined only once every base signal has warmed up.
    components = pd.concat([trend, momentum, reversion], axis=1)
    out["composite_score"] = components.mean(axis=1)
    out["target_position"] = (out["composite_score"] >= 0.5).astype(float)
    out.loc[out["composite_score"].isna(), "target_position"] = np.nan
    return out


_DISPATCH = {
    "trend": _trend_position,
    "momentum": _momentum_position,
    "mean_reversion": _mean_reversion_position,
    "composite": _composite_position,
}


def volatility_filter_mask(frame: pd.DataFrame, config) -> pd.Series:
    """Return a ``{0, 1}`` 'risk-on' mask using a trailing volatility threshold.

    A bar is risk-off (``0``) when its rolling volatility exceeds an expanding
    quantile of past volatility (``vol_filter_quantile``); otherwise risk-on
    (``1``). The expanding quantile uses only data up to the current bar, so the
    filter never peeks ahead. Warm-up bars (no threshold yet) are risk-on.
    """
    returns = frame["close"].pct_change()
    vol = returns.rolling(config.vol_filter_window).std(ddof=0)
    threshold = vol.expanding(min_periods=config.vol_filter_window).quantile(
        config.vol_filter_quantile
    )
    risk_on = (vol <= threshold).astype(float)
    risk_on = risk_on.where(threshold.notna(), other=1.0)
    return risk_on


def build_signal_frame(frame: pd.DataFrame, config) -> pd.DataFrame:
    """Build the explainable signal frame (component columns + target position).

    The returned frame always contains a ``target_position`` column in
    ``{0, 1}`` (NaN during warm-up). When ``use_volatility_filter`` is set, the
    target is additionally forced flat during high-volatility bars and a
    ``risk_on`` column records the filter state.

    The target is **not** shifted here; the backtest engine applies the one-bar
    lag so the look-ahead guard lives in exactly one place.
    """
    family = getattr(config, "signal_family", "trend")
    if family not in _DISPATCH:
        raise ValidationError(
            f"Unknown signal_family '{family}'. "
            f"Choose one of: {', '.join(SIGNAL_FAMILIES)}."
        )
    validate_price_frame(frame, min_rows=required_warmup(config) + 2)

    out = _DISPATCH[family](frame, config)

    if getattr(config, "use_volatility_filter", False):
        risk_on = volatility_filter_mask(frame, config)
        out["risk_on"] = risk_on
        # Keep NaN warm-up bars NaN; otherwise damp exposure when risk-off.
        target = out["target_position"]
        out["target_position"] = target.where(target.isna(), target * risk_on)

    return out
