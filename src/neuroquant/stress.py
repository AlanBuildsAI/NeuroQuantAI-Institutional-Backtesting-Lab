"""Cost sensitivity and stress diagnostics.

Two complementary robustness views for a *selected* configuration:

* :func:`cost_sensitivity_analysis` re-runs the same rule under a ladder of
  transaction-cost assumptions, showing how quickly costs erode the result.
* :func:`stress_test_summary` applies simple, clearly-labelled transformations
  to the realised return stream (volatility shock, adverse drift, amplified
  downside) to see how fragile the outcome is.

These are descriptive stress diagnostics on synthetic (or user-supplied) data,
not forecasts and not statements about real markets.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from .backtest import BacktestConfig, run_backtest
from .metrics import TRADING_DAYS, compute_kpis

DEFAULT_COST_LADDER = (0.0, 0.0005, 0.001, 0.0025, 0.005)


def cost_sensitivity_analysis(
    frame: pd.DataFrame,
    config: BacktestConfig,
    costs: tuple[float, ...] = DEFAULT_COST_LADDER,
) -> pd.DataFrame:
    """Re-run one configuration across a ladder of transaction-cost levels.

    Returns one row per cost level with: ``cost``, ``total_return``,
    ``sharpe_ratio``, ``max_drawdown``, ``trade_count`` and ``turnover``. Rows
    are ordered by increasing cost so the erosion is easy to read.
    """
    rows: list[dict] = []
    for cost in costs:
        result = run_backtest(frame, replace(config, cost_per_trade=cost))
        kpis = result["kpis"]
        rows.append(
            {
                "cost": cost,
                "total_return": kpis["total_return"],
                "sharpe_ratio": kpis["sharpe_ratio"],
                "max_drawdown": kpis["max_drawdown"],
                "trade_count": kpis["trade_count"],
                "turnover": kpis["turnover"],
            }
        )
    return pd.DataFrame(rows).sort_values("cost").reset_index(drop=True)


def _summary_from_returns(returns: pd.Series) -> dict:
    """Total return, Sharpe and max drawdown from a strategy-return slice."""
    equity = (1.0 + returns).cumprod()
    total_return = float(equity.iloc[-1] - 1.0)
    std = float(returns.std(ddof=0))
    if std > 0:
        sharpe = float(returns.mean() * TRADING_DAYS / (std * np.sqrt(TRADING_DAYS)))
    else:
        sharpe = 0.0
    drawdown = float((equity / equity.cummax() - 1.0).min())
    return {
        "total_return": total_return,
        "sharpe_ratio": sharpe,
        "max_drawdown": drawdown,
    }


def stress_test_summary(
    frame: pd.DataFrame,
    config: BacktestConfig,
    vol_shock: float = 1.5,
    adverse_daily_drift: float = 0.0005,
    downside_amplifier: float = 1.5,
    higher_cost: float = 0.005,
) -> pd.DataFrame:
    """Summarise the selected configuration under several stress transforms.

    Scenarios (each clearly synthetic and labelled):

    * ``baseline`` — the realised result, unchanged,
    * ``higher_costs`` — the same rule re-run at ``higher_cost`` per trade,
    * ``volatility_shock`` — returns de-meaned and re-scaled by ``vol_shock``,
    * ``adverse_drift`` — a small constant subtracted from every return,
    * ``amplified_downside`` — negative returns multiplied by ``downside_amplifier``.

    Returns one row per scenario with ``scenario``, ``total_return``,
    ``sharpe_ratio`` and ``max_drawdown``.
    """
    base_signals = run_backtest(frame, config)["signals"]
    base_returns = base_signals["strategy_return"]
    mean = float(base_returns.mean())

    higher_cost_returns = run_backtest(
        frame, replace(config, cost_per_trade=higher_cost)
    )["signals"]["strategy_return"]

    scenarios = {
        "baseline": base_returns,
        "higher_costs": higher_cost_returns,
        "volatility_shock": mean + (base_returns - mean) * vol_shock,
        "adverse_drift": base_returns - adverse_daily_drift,
        "amplified_downside": base_returns.where(
            base_returns >= 0, base_returns * downside_amplifier
        ),
    }

    rows = [
        {"scenario": name, **_summary_from_returns(returns)}
        for name, returns in scenarios.items()
    ]
    return pd.DataFrame(rows)
