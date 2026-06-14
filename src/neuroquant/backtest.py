"""Experiment engine: signal generation, costs, baseline, and sweeps.

This module turns a validated signal series into a reproducible experiment.
The core idea is a moving-average crossover rule, but the analytical point
is the *workflow*: signals are shifted to avoid look-ahead bias, simple
transaction costs are applied on position changes, and every strategy is
compared against a transparent buy-and-hold BASELINE.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .metrics import compute_kpis
from .validation import validate_price_frame, validate_window_config


@dataclass(frozen=True)
class BacktestConfig:
    """Configuration for a single backtest run.

    Attributes
    ----------
    short_window:
        Fast moving-average window (must be < long_window).
    long_window:
        Slow moving-average window.
    cost_per_trade:
        Round-trip transaction cost + slippage applied as a fraction of the
        position value each time the position changes (e.g. 0.001 = 0.1%).
    """

    short_window: int = 20
    long_window: int = 60
    cost_per_trade: float = 0.001


def generate_signals(frame: pd.DataFrame, config: BacktestConfig) -> pd.DataFrame:
    """Build positions and returns for one configuration.

    Signal logic:
      * compute fast and slow simple moving averages of ``close``
      * go "in market" (position = 1) when fast > slow, else flat (0)
      * SHIFT the position by one bar so today's decision uses only data
        available up to yesterday -- this avoids look-ahead bias
      * apply a transaction cost whenever the position changes

    Returns a DataFrame with intermediate and result columns including
    ``market_return``, ``strategy_return``, ``baseline_return``,
    ``strategy_equity`` and ``baseline_equity``.
    """
    validate_window_config(config.short_window, config.long_window)
    validate_price_frame(frame, min_rows=config.long_window + 1)

    data = frame.copy()
    data["fast_ma"] = data["close"].rolling(config.short_window).mean()
    data["slow_ma"] = data["close"].rolling(config.long_window).mean()

    # Raw signal: 1 when fast above slow, else 0.
    raw_signal = (data["fast_ma"] > data["slow_ma"]).astype(float)

    # Shift by 1 bar so we only act on information from the previous bar.
    data["position"] = raw_signal.shift(1).fillna(0.0)

    # One-bar simple returns of the underlying signal series (the baseline).
    data["market_return"] = data["close"].pct_change().fillna(0.0)

    # Cost charged whenever the position changes (entry or exit).
    position_change = data["position"].diff().abs().fillna(0.0)
    data["cost"] = position_change * config.cost_per_trade

    # Strategy return = exposure * market return, minus trading costs.
    data["strategy_return"] = (
        data["position"] * data["market_return"] - data["cost"]
    )
    data["baseline_return"] = data["market_return"]

    data["strategy_equity"] = (1.0 + data["strategy_return"]).cumprod()
    data["baseline_equity"] = (1.0 + data["baseline_return"]).cumprod()

    return data


def run_backtest(frame: pd.DataFrame, config: BacktestConfig) -> dict:
    """Run a single backtest and return signals plus KPI metrics.

    Returns
    -------
    dict
        ``{"config": BacktestConfig, "signals": DataFrame, "kpis": dict}``.
    """
    signals = generate_signals(frame, config)
    kpis = compute_kpis(signals)
    return {"config": config, "signals": signals, "kpis": kpis}


def run_parameter_sweep(
    frame: pd.DataFrame,
    short_windows: list[int],
    long_windows: list[int],
    cost_per_trade: float = 0.001,
) -> pd.DataFrame:
    """Run many configurations and return a tidy comparison table.

    Every valid (short, long) combination where short < long is evaluated.
    The result is one row per scenario with its configuration and KPIs,
    sorted by Sharpe ratio (best first) for easy scanning.

    Returns
    -------
    pandas.DataFrame
        Tidy frame with columns: short_window, long_window, cost_per_trade,
        and all KPI columns from :func:`neuroquant.metrics.compute_kpis`.
    """
    rows: list[dict] = []
    for short_window in short_windows:
        for long_window in long_windows:
            if short_window >= long_window:
                continue
            config = BacktestConfig(
                short_window=short_window,
                long_window=long_window,
                cost_per_trade=cost_per_trade,
            )
            result = run_backtest(frame, config)
            row = {
                "short_window": short_window,
                "long_window": long_window,
                "cost_per_trade": cost_per_trade,
            }
            row.update(result["kpis"])
            rows.append(row)

    if not rows:
        raise ValueError(
            "Parameter sweep produced no valid scenarios. Ensure at least "
            "one short_window is smaller than a long_window."
        )

    summary = pd.DataFrame(rows)
    summary = summary.sort_values(
        "sharpe_ratio", ascending=False
    ).reset_index(drop=True)
    return summary
