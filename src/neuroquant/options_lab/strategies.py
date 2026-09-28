"""Causal entry-signal recipes that feed the options backtester."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .volatility import realized_volatility


@dataclass(frozen=True)
class CheapVolBullishConfig:
    """Simple explainable signal for long-volatility bullish research."""

    fast_ma: int = 20
    slow_ma: int = 60
    rv_window: int = 20
    max_iv_to_rv: float = 1.0
    cooldown_days: int = 10

    def __post_init__(self) -> None:
        if not 1 <= self.fast_ma < self.slow_ma:
            raise ValueError("require 1 <= fast_ma < slow_ma")
        if self.rv_window < 2:
            raise ValueError("rv_window must be at least 2")
        if self.max_iv_to_rv <= 0:
            raise ValueError("max_iv_to_rv must be positive")
        if self.cooldown_days < 0:
            raise ValueError("cooldown_days cannot be negative")


def build_cheap_vol_bullish_entries(
    underlying: pd.DataFrame,
    atm_iv: pd.DataFrame,
    config: CheapVolBullishConfig | None = None,
) -> pd.DataFrame:
    """Generate causal daily entry decisions.

    The rule is deliberately interpretable:
      1. prior-close fast MA > slow MA;
      2. prior observed ATM IV / prior realized volatility <= threshold;
      3. a cooldown prevents overlapping daily entries.

    Both trend and volatility inputs are shifted one bar before the decision.
    """

    config = config or CheapVolBullishConfig()
    if not {"timestamp", "close"}.issubset(underlying.columns):
        raise ValueError("underlying requires timestamp and close")
    if not {"timestamp", "iv"}.issubset(atm_iv.columns):
        raise ValueError("atm_iv requires timestamp and iv")

    u = underlying[["timestamp", "close"]].copy()
    u["timestamp"] = pd.to_datetime(u["timestamp"], utc=True)
    u = u.sort_values("timestamp").drop_duplicates("timestamp")
    u["fast_ma"] = u["close"].rolling(config.fast_ma).mean()
    u["slow_ma"] = u["close"].rolling(config.slow_ma).mean()
    u["realized_vol"] = realized_volatility(u["close"], config.rv_window)

    v = atm_iv[["timestamp", "iv"]].copy()
    v["timestamp"] = pd.to_datetime(v["timestamp"], utc=True)
    v = v.sort_values("timestamp").drop_duplicates("timestamp")

    data = pd.merge_asof(u, v, on="timestamp", direction="backward")
    for col in ["fast_ma", "slow_ma", "realized_vol", "iv"]:
        data[f"lag_{col}"] = data[col].shift(1)

    data["iv_to_rv"] = data["lag_iv"] / data["lag_realized_vol"]
    raw = (
        (data["lag_fast_ma"] > data["lag_slow_ma"])
        & (data["iv_to_rv"] <= config.max_iv_to_rv)
    ).fillna(False)

    entries = []
    last_entry = None
    for ts, signal in zip(data["timestamp"], raw):
        allowed = (
            last_entry is None
            or (ts - last_entry).days >= config.cooldown_days
        )
        take = bool(signal and allowed)
        entries.append(take)
        if take:
            last_entry = ts

    data["entry_signal"] = entries
    return data[
        [
            "timestamp",
            "close",
            "lag_fast_ma",
            "lag_slow_ma",
            "lag_iv",
            "lag_realized_vol",
            "iv_to_rv",
            "entry_signal",
        ]
    ].copy()
