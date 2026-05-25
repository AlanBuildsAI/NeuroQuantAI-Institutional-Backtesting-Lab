"""
Minimal synthetic backtest example.

This script is intentionally simple and uses synthetic price data only.
It is included to make the repository reproducible as a portfolio project.

Not financial advice. Not intended for live trading.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_synthetic_prices(days: int = 252, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic daily price series."""
    rng = np.random.default_rng(seed)
    daily_returns = rng.normal(loc=0.0004, scale=0.015, size=days)
    prices = 100 * (1 + pd.Series(daily_returns)).cumprod()
    return pd.DataFrame({"close": prices})


def run_moving_average_backtest(df: pd.DataFrame, short_window: int = 10, long_window: int = 30) -> pd.DataFrame:
    """Run a simple moving-average crossover backtest."""
    data = df.copy()
    data["short_ma"] = data["close"].rolling(short_window).mean()
    data["long_ma"] = data["close"].rolling(long_window).mean()
    data["signal"] = np.where(data["short_ma"] > data["long_ma"], 1, 0)
    data["position"] = data["signal"].shift(1).fillna(0)
    data["market_return"] = data["close"].pct_change().fillna(0)
    data["strategy_return"] = data["position"] * data["market_return"]
    data["equity_curve"] = (1 + data["strategy_return"]).cumprod()
    return data


def summarize_performance(results: pd.DataFrame) -> dict:
    """Calculate basic performance metrics."""
    returns = results["strategy_return"]
    total_return = results["equity_curve"].iloc[-1] - 1
    volatility = returns.std() * np.sqrt(252)
    sharpe = (returns.mean() * 252) / volatility if volatility != 0 else 0
    running_max = results["equity_curve"].cummax()
    drawdown = (results["equity_curve"] - running_max) / running_max
    max_drawdown = drawdown.min()
    return {
        "total_return_pct": round(total_return * 100, 2),
        "annualized_volatility_pct": round(volatility * 100, 2),
        "sharpe_ratio": round(float(sharpe), 2),
        "max_drawdown_pct": round(max_drawdown * 100, 2),
    }


if __name__ == "__main__":
    prices = generate_synthetic_prices()
    results = run_moving_average_backtest(prices)
    summary = summarize_performance(results)
    print("Synthetic backtest summary")
    for metric, value in summary.items():
        print(f"{metric}: {value}")
