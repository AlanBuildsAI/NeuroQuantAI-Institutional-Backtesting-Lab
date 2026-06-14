"""Position-sizing methods that turn a raw signal into a target exposure.

All methods are **causal**: they use only trailing information to scale the
signal, and the backtest engine shifts the resulting exposure by one bar before
any return is applied, so no future information leaks in.

Defaults are deliberately conservative: no leverage (``max_exposure = 1.0``),
no shorting (``allow_short = False``). ``fixed_unit`` reproduces the original
0/1 long-or-flat behaviour exactly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252

SIZING_METHODS = (
    "fixed_unit",
    "fixed_fraction",
    "volatility_target",
    "capped_exposure",
)


def _clip(exposure: pd.Series, max_exposure: float, allow_short: bool) -> pd.Series:
    """Clip exposure to the allowed range (no leverage / no shorting by default)."""
    lower = -abs(max_exposure) if allow_short else 0.0
    return exposure.clip(lower=lower, upper=abs(max_exposure))


def fixed_unit(target: pd.Series) -> pd.Series:
    """Exposure equals the raw signal (the original 0/1 long-or-flat behaviour)."""
    return target


def fixed_fraction(target: pd.Series, fraction: float = 1.0) -> pd.Series:
    """Exposure equals the signal scaled by a constant fraction."""
    return target * float(fraction)


def volatility_target(
    target: pd.Series,
    returns: pd.Series,
    target_volatility: float = 0.15,
    lookback: int = 20,
) -> pd.Series:
    """Scale exposure to aim at a target annualised volatility.

    Uses **trailing** realised volatility (a rolling standard deviation of past
    returns, annualised). Where trailing volatility is undefined (warm-up) or
    zero, the scale is left undefined and the engine treats it as flat.
    """
    realized = returns.rolling(lookback).std(ddof=0) * np.sqrt(TRADING_DAYS)
    scale = (float(target_volatility) / realized).replace(
        [np.inf, -np.inf], np.nan
    )
    return target * scale


def compute_exposure(target_position: pd.Series, returns: pd.Series, config) -> pd.Series:
    """Dispatch to the configured sizing method and clip to allowed exposure.

    Returns the *unshifted* target exposure; the backtest engine applies the
    one-bar lag. ``config`` is duck-typed (attribute access) to avoid importing
    :class:`neuroquant.backtest.BacktestConfig` here.
    """
    method = getattr(config, "sizing_method", "fixed_unit")
    max_exposure = float(getattr(config, "max_exposure", 1.0))
    allow_short = bool(getattr(config, "allow_short", False))
    fraction = float(getattr(config, "fixed_fraction", 1.0))

    if method == "fixed_unit":
        raw = fixed_unit(target_position)
    elif method == "fixed_fraction":
        raw = fixed_fraction(target_position, fraction)
    elif method == "capped_exposure":
        # Fraction sizing whose only real effect is the max-exposure clip below.
        raw = fixed_fraction(target_position, fraction)
    elif method == "volatility_target":
        raw = volatility_target(
            target_position,
            returns,
            float(getattr(config, "target_volatility", 0.15)),
            int(getattr(config, "vol_lookback", 20)),
        )
    else:
        raise ValueError(
            f"Unknown sizing_method '{method}'. Choose one of: "
            f"{', '.join(SIZING_METHODS)}."
        )

    return _clip(raw, max_exposure, allow_short)
