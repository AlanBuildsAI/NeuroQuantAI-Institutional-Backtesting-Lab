"""Volatility features for option research."""

from __future__ import annotations

import numpy as np
import pandas as pd


TRADING_DAYS = 252


def realized_volatility(close: pd.Series, window: int = 20) -> pd.Series:
    """Annualized close-to-close realized volatility."""

    if window < 2:
        raise ValueError("window must be at least 2")
    prices = pd.Series(close, dtype=float)
    if (prices <= 0).any():
        raise ValueError("close prices must be positive")
    log_returns = np.log(prices).diff()
    return log_returns.rolling(window).std(ddof=0) * np.sqrt(TRADING_DAYS)


def attach_realized_volatility(
    chain: pd.DataFrame,
    underlying: pd.DataFrame,
    window: int = 20,
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """Backward as-of merge realized volatility onto option snapshots."""

    if timestamp_col not in chain.columns:
        raise ValueError("chain requires timestamp")
    if timestamp_col not in underlying.columns or "close" not in underlying.columns:
        raise ValueError("underlying requires timestamp and close")

    u = underlying[[timestamp_col, "close"]].copy()
    u[timestamp_col] = pd.to_datetime(u[timestamp_col], utc=True)
    u = u.sort_values(timestamp_col)
    u["realized_vol"] = realized_volatility(u["close"], window)

    c = chain.copy()
    c[timestamp_col] = pd.to_datetime(c[timestamp_col], utc=True)
    c = c.sort_values(timestamp_col)
    return pd.merge_asof(
        c,
        u[[timestamp_col, "realized_vol"]],
        on=timestamp_col,
        direction="backward",
    )


def iv_to_rv_ratio(iv: pd.Series, realized_vol: pd.Series) -> pd.Series:
    """IV divided by realized volatility. Values below one mean IV is lower."""

    iv = pd.Series(iv, dtype=float)
    rv = pd.Series(realized_vol, dtype=float)
    return iv / rv.replace(0.0, np.nan)


def long_volatility_candidate_mask(
    frame: pd.DataFrame,
    max_iv_to_rv: float = 1.0,
    max_iv_rank: float | None = None,
) -> pd.Series:
    """Research filter for relatively inexpensive long-volatility setups."""

    if "iv" not in frame.columns or "realized_vol" not in frame.columns:
        raise ValueError("frame requires iv and realized_vol")
    ratio = iv_to_rv_ratio(frame["iv"], frame["realized_vol"])
    mask = ratio <= max_iv_to_rv
    if max_iv_rank is not None:
        if "iv_rank" not in frame.columns:
            raise ValueError("max_iv_rank requires an iv_rank column")
        mask &= frame["iv_rank"] <= max_iv_rank
    return mask.fillna(False)
