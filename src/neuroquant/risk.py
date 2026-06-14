"""Optional risk controls and rolling risk diagnostics.

These are research diagnostics and *optional* exposure controls — not trading
advice. The controls are causal (they only use trailing information) and, like
sizing, the engine applies the one-bar lag before returns are realised.

Defaults leave every control off so the base backtest is unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --- Rolling diagnostics ----------------------------------------------------

def rolling_volatility(returns: pd.Series, window: int = 20) -> pd.Series:
    """Trailing annualised volatility of a return series."""
    return returns.rolling(window).std(ddof=0) * np.sqrt(TRADING_DAYS)


def rolling_sharpe(returns: pd.Series, window: int = 63) -> pd.Series:
    """Trailing annualised Sharpe ratio (risk-free rate assumed zero)."""
    mean = returns.rolling(window).mean() * TRADING_DAYS
    vol = returns.rolling(window).std(ddof=0) * np.sqrt(TRADING_DAYS)
    return (mean / vol).replace([np.inf, -np.inf], np.nan)


def rolling_drawdown(equity: pd.Series) -> pd.Series:
    """Drawdown of an equity curve relative to its running peak (≤ 0)."""
    return equity / equity.cummax() - 1.0


# --- Optional exposure controls ---------------------------------------------

def apply_volatility_cap(
    exposure: pd.Series,
    returns: pd.Series,
    vol_cap: float,
    lookback: int = 20,
) -> pd.Series:
    """Scale exposure down when trailing volatility would breach ``vol_cap``.

    The cap can only *reduce* exposure (scale ≤ 1), never lever it up. Where
    trailing volatility is undefined, exposure is left unchanged.
    """
    realized = returns.rolling(lookback).std(ddof=0) * np.sqrt(TRADING_DAYS)
    scale = (float(vol_cap) / realized).clip(upper=1.0)
    scale = scale.where(realized > 0, other=1.0).fillna(1.0)
    return exposure * scale


def apply_risk_controls(exposure: pd.Series, returns: pd.Series, config) -> pd.Series:
    """Apply optional volatility cap and a hard max-exposure clip.

    No-op unless ``config.use_risk_controls`` is true. ``config`` is duck-typed.
    """
    if not getattr(config, "use_risk_controls", False):
        return exposure
    capped = apply_volatility_cap(
        exposure,
        returns,
        float(getattr(config, "vol_cap", 0.30)),
        int(getattr(config, "vol_lookback", 20)),
    )
    max_exposure = float(getattr(config, "max_exposure", 1.0))
    allow_short = bool(getattr(config, "allow_short", False))
    lower = -abs(max_exposure) if allow_short else 0.0
    return capped.clip(lower=lower, upper=abs(max_exposure))


def apply_drawdown_guard(
    position: pd.Series,
    market_return: pd.Series,
    level: float = 0.20,
) -> pd.Series:
    """De-risk to flat after a trailing drawdown breach, re-enter at a new high.

    A simple, explainable stop-style rule applied to the (already shifted)
    exposure: once the strategy's running drawdown reaches ``-level`` the guard
    flattens exposure, re-enabling only when equity makes a fresh high. The
    decision at each bar uses drawdown information through the prior bar, so the
    rule is causal. Implemented sequentially because it is path-dependent.
    """
    pos = position.to_numpy(dtype=float, copy=True)
    mret = np.nan_to_num(market_return.to_numpy(dtype=float))
    out = pos.copy()

    equity = 1.0
    peak = 1.0
    guard_off = False
    for t in range(len(pos)):
        if guard_off:
            out[t] = 0.0
        equity *= 1.0 + out[t] * mret[t]
        peak = max(peak, equity)
        drawdown = equity / peak - 1.0
        if drawdown <= -abs(level):
            guard_off = True
        elif equity >= peak:
            guard_off = False
    return pd.Series(out, index=position.index)
