"""
NeuroQuantAI - synthetic backtesting analytics workflow.

Portfolio-oriented example for analytics roles. Uses synthetic data only and is not
financial advice, trading advice, or a live trading system.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

TRADING_DAYS = 252
OUTPUT_DIR = Path("sample_outputs")
ASSET_DIR = Path("docs/assets")


@dataclass(frozen=True)
class BacktestConfig:
    short_window: int
    long_window: int
    transaction_cost_bps: float = 5.0
    slippage_bps: float = 2.0


def generate_synthetic_prices(days: int = TRADING_DAYS * 2, seed: int = 42) -> pd.DataFrame:
    """Generate a reproducible synthetic daily close-price series."""
    rng = np.random.default_rng(seed)
    daily_returns = rng.normal(loc=0.00035, scale=0.014, size=days)
    close = 100 * (1 + pd.Series(daily_returns)).cumprod()
    return pd.DataFrame({"day": np.arange(1, days + 1), "close": close.round(4)})


def validate_price_data(df: pd.DataFrame) -> None:
    """Run lightweight data quality checks before analysis."""
    required_columns = {"close"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df["close"].isna().any():
        raise ValueError("Close price series contains missing values")
    if (df["close"] <= 0).any():
        raise ValueError("Close prices must be positive")


def run_moving_average_backtest(df: pd.DataFrame, config: BacktestConfig) -> pd.DataFrame:
    """Run a moving-average crossover backtest with costs and slippage."""
    if config.short_window >= config.long_window:
        raise ValueError("short_window must be smaller than long_window")

    validate_price_data(df)
    data = df.copy()
    data["short_ma"] = data["close"].rolling(config.short_window).mean()
    data["long_ma"] = data["close"].rolling(config.long_window).mean()
    data["signal"] = np.where(data["short_ma"] > data["long_ma"], 1, 0)
    data["position"] = data["signal"].shift(1).fillna(0)
    data["market_return"] = data["close"].pct_change().fillna(0)

    trade_count_series = data["position"].diff().abs().fillna(data["position"].abs())
    round_trip_cost = (config.transaction_cost_bps + config.slippage_bps) / 10_000
    data["implementation_cost"] = trade_count_series * round_trip_cost
    data["strategy_return"] = (data["position"] * data["market_return"]) - data["implementation_cost"]
    data["equity_curve"] = (1 + data["strategy_return"]).cumprod()
    data["benchmark_equity_curve"] = (1 + data["market_return"]).cumprod()
    data["drawdown"] = data["equity_curve"] / data["equity_curve"].cummax() - 1
    return data


def summarize_performance(results: pd.DataFrame, config: BacktestConfig) -> dict[str, float | int]:
    """Calculate decision-ready performance and risk metrics."""
    returns = results["strategy_return"]
    market_returns = results["market_return"]
    total_return = results["equity_curve"].iloc[-1] - 1
    benchmark_return = results["benchmark_equity_curve"].iloc[-1] - 1
    volatility = returns.std() * np.sqrt(TRADING_DAYS)
    sharpe = (returns.mean() * TRADING_DAYS) / volatility if volatility != 0 else 0.0
    max_drawdown = results["drawdown"].min()
    active_days = int((results["position"] > 0).sum())
    trade_count = int(results["position"].diff().abs().fillna(0).sum())
    downside = returns[returns < 0].std() * np.sqrt(TRADING_DAYS)
    sortino = (returns.mean() * TRADING_DAYS) / downside if downside and downside != 0 else 0.0
    correlation = returns.corr(market_returns)

    return {
        "short_window": config.short_window,
        "long_window": config.long_window,
        "total_return_pct": round(total_return * 100, 2),
        "benchmark_return_pct": round(benchmark_return * 100, 2),
        "annualized_volatility_pct": round(volatility * 100, 2),
        "sharpe_ratio": round(float(sharpe), 2),
        "sortino_ratio": round(float(sortino), 2),
        "max_drawdown_pct": round(max_drawdown * 100, 2),
        "active_days": active_days,
        "trade_count": trade_count,
        "market_correlation": round(float(correlation), 2) if not pd.isna(correlation) else 0.0,
    }


def run_parameter_sweep(price_data: pd.DataFrame, configs: Iterable[BacktestConfig]) -> pd.DataFrame:
    """Compare multiple strategy configurations in a structured summary table."""
    rows = []
    for config in configs:
        results = run_moving_average_backtest(price_data, config)
        rows.append(summarize_performance(results, config))
    return pd.DataFrame(rows).sort_values(["sharpe_ratio", "total_return_pct"], ascending=False)


def export_outputs(results: pd.DataFrame, summary_table: pd.DataFrame) -> None:
    """Export dashboard-ready CSVs and simple chart images."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)

    summary_table.to_csv(OUTPUT_DIR / "parameter_sweep_summary.csv", index=False)
    results[["day", "close", "equity_curve", "benchmark_equity_curve", "drawdown"]].to_csv(
        OUTPUT_DIR / "equity_curve_sample.csv", index=False
    )

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping chart export")
        return

    plt.figure(figsize=(9, 5))
    plt.plot(results["day"], results["equity_curve"], label="Strategy")
    plt.plot(results["day"], results["benchmark_equity_curve"], label="Benchmark")
    plt.title("Synthetic Strategy vs Benchmark Equity Curve")
    plt.xlabel("Synthetic trading day")
    plt.ylabel("Growth of $1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ASSET_DIR / "equity_curve.svg")
    plt.close()

    plt.figure(figsize=(9, 4))
    plt.plot(results["day"], results["drawdown"])
    plt.title("Strategy Drawdown Profile")
    plt.xlabel("Synthetic trading day")
    plt.ylabel("Drawdown")
    plt.tight_layout()
    plt.savefig(ASSET_DIR / "drawdown_profile.svg")
    plt.close()


def main() -> None:
    prices = generate_synthetic_prices()
    sweep_configs = [
        BacktestConfig(short_window=5, long_window=20),
        BacktestConfig(short_window=10, long_window=30),
        BacktestConfig(short_window=20, long_window=60),
    ]
    summary_table = run_parameter_sweep(prices, sweep_configs)
    selected_config = BacktestConfig(short_window=20, long_window=60)
    selected_results = run_moving_average_backtest(prices, selected_config)

    export_outputs(selected_results, summary_table)

    print("Synthetic backtest parameter sweep")
    print(summary_table.to_string(index=False))
    print("\nSaved outputs:")
    print(f"- {OUTPUT_DIR / 'parameter_sweep_summary.csv'}")
    print(f"- {OUTPUT_DIR / 'equity_curve_sample.csv'}")
    print(f"- {ASSET_DIR / 'equity_curve.svg'}")
    print(f"- {ASSET_DIR / 'drawdown_profile.svg'}")


if __name__ == "__main__":
    main()
