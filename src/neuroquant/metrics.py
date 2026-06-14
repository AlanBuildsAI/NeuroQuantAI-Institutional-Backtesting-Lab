"""KPI metric definitions.

Each metric is a small, documented function of the per-bar return series.
Together they form a decision-ready scorecard that compares a strategy
configuration against a transparent buy-and-hold baseline.

All metrics are descriptive summaries of synthetic data. They are NOT
forecasts and carry no claim about real-world performance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Trading days per year, used to annualise volatility and risk-adjusted ratios.
TRADING_DAYS = 252


def _max_drawdown(equity: pd.Series) -> float:
    """Largest peak-to-trough decline of an equity curve (a negative fraction)."""
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def compute_kpis(signals: pd.DataFrame) -> dict:
    """Compute the KPI scorecard for one backtest result.

    Parameters
    ----------
    signals:
        Output of :func:`neuroquant.backtest.generate_signals`, containing
        ``strategy_return``, ``baseline_return``, ``position`` and the
        cumulative equity columns.

    Returns
    -------
    dict
        Dictionary of KPIs:

        total_return
            Cumulative strategy return over the full window (fraction).
        baseline_return
            Cumulative buy-and-hold return over the same window (fraction).
        excess_return
            Strategy total return minus the baseline total return.
        annualized_return
            Geometric (compounded) strategy return scaled to one year.
        annualized_volatility
            Standard deviation of strategy returns scaled to one year.
        sharpe_ratio
            Annualised mean strategy return divided by annualised volatility
            (risk-free rate assumed zero).
        sortino_ratio
            Like Sharpe but penalising only downside (negative) volatility.
        calmar_ratio
            Annualised return divided by the absolute max drawdown.
        information_ratio
            Annualised active return (strategy minus baseline) divided by the
            tracking error of that difference series.
        max_drawdown
            Worst peak-to-trough decline of the strategy equity curve.
        active_days
            Number of bars the strategy held a position (exposure count).
        trade_count
            Number of position changes (entries + exits).
        turnover
            Average per-bar change in exposure (activity / cost proxy).
        correlation_to_baseline
            Correlation between strategy and baseline daily returns.
        win_loss_ratio
            Average winning return divided by average absolute losing return.

    All figures describe synthetic (or user-supplied) historical data. They
    are descriptive research diagnostics, not forecasts, and carry no claim
    about future or real-world performance.
    """
    strat = signals["strategy_return"]
    base = signals["baseline_return"]
    position = signals["position"]
    n_periods = len(strat)

    total_return = float(signals["strategy_equity"].iloc[-1] - 1.0)
    baseline_return = float(signals["baseline_equity"].iloc[-1] - 1.0)
    excess_return = total_return - baseline_return

    # Annualised (geometric) return: compound growth scaled to one year.
    if n_periods > 0 and (1.0 + total_return) > 0:
        annualized_return = float(
            (1.0 + total_return) ** (TRADING_DAYS / n_periods) - 1.0
        )
    else:
        annualized_return = 0.0

    annualized_volatility = float(strat.std(ddof=0) * np.sqrt(TRADING_DAYS))

    mean_daily = float(strat.mean())
    if annualized_volatility > 0:
        sharpe_ratio = mean_daily * TRADING_DAYS / annualized_volatility
    else:
        sharpe_ratio = 0.0

    downside = strat[strat < 0]
    downside_std = float(downside.std(ddof=0)) if len(downside) else 0.0
    if downside_std > 0:
        sortino_ratio = (mean_daily * TRADING_DAYS) / (
            downside_std * np.sqrt(TRADING_DAYS)
        )
    else:
        sortino_ratio = 0.0

    max_drawdown = _max_drawdown(signals["strategy_equity"])

    # Calmar ratio: annualised return per unit of worst-case drawdown.
    if max_drawdown < 0:
        calmar_ratio = annualized_return / abs(max_drawdown)
    else:
        calmar_ratio = 0.0

    active_days = int((position > 0).sum())
    trade_count = int(position.diff().abs().fillna(0.0).gt(0).sum())

    # Turnover: average per-period change in exposure (a cost/activity proxy).
    turnover = float(position.diff().abs().fillna(0.0).mean())

    if strat.std(ddof=0) > 0 and base.std(ddof=0) > 0:
        correlation_to_baseline = float(strat.corr(base))
    else:
        correlation_to_baseline = 0.0

    # Information ratio: annualised active return per unit of tracking error
    # (volatility of the strategy-minus-baseline difference series).
    active = strat - base
    tracking_error = float(active.std(ddof=0) * np.sqrt(TRADING_DAYS))
    if tracking_error > 0:
        information_ratio = float(active.mean()) * TRADING_DAYS / tracking_error
    else:
        information_ratio = 0.0

    wins = strat[strat > 0]
    losses = strat[strat < 0]
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(abs(losses.mean())) if len(losses) else 0.0
    win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0

    return {
        "total_return": total_return,
        "baseline_return": baseline_return,
        "excess_return": excess_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": float(sharpe_ratio),
        "sortino_ratio": float(sortino_ratio),
        "calmar_ratio": float(calmar_ratio),
        "information_ratio": information_ratio,
        "max_drawdown": max_drawdown,
        "active_days": active_days,
        "trade_count": trade_count,
        "turnover": turnover,
        "correlation_to_baseline": correlation_to_baseline,
        "win_loss_ratio": float(win_loss_ratio),
    }
