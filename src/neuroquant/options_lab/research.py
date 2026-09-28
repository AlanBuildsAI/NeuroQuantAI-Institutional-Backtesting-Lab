"""Validation and robustness tools for option-trade result tables."""

from __future__ import annotations

import numpy as np
import pandas as pd


def chronological_trade_split(
    trades: pd.DataFrame,
    train_fraction: float = 0.70,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological train/test split; never shuffle time-series trades."""

    if not 0.1 <= train_fraction <= 0.9:
        raise ValueError("train_fraction must be between 0.1 and 0.9")
    if trades.empty:
        return trades.copy(), trades.copy()
    data = trades.sort_values("entry_timestamp").reset_index(drop=True)
    cut = max(1, min(len(data) - 1, int(len(data) * train_fraction)))
    if len(data) == 1:
        return data.copy(), data.iloc[0:0].copy()
    return data.iloc[:cut].copy(), data.iloc[cut:].copy()


def bootstrap_trade_returns(
    trades: pd.DataFrame,
    n_simulations: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """Bootstrap trade return sequences for robustness, not forecasting."""

    if n_simulations <= 0:
        raise ValueError("n_simulations must be positive")
    if "return_on_debit" not in trades.columns or trades.empty:
        raise ValueError("trades requires non-empty return_on_debit")
    values = trades["return_on_debit"].dropna().to_numpy(dtype=float)
    if len(values) == 0:
        raise ValueError("no finite trade returns")

    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_simulations):
        sample = rng.choice(values, size=len(values), replace=True)
        equity = np.cumprod(1.0 + sample)
        peaks = np.maximum.accumulate(equity)
        dd = equity / peaks - 1.0
        rows.append(
            {
                "simulation": i,
                "compound_return": float(equity[-1] - 1.0),
                "max_drawdown": float(dd.min()),
                "losing_path": bool(equity[-1] < 1.0),
            }
        )
    return pd.DataFrame(rows)


def execution_stress_summary(
    scenario_results: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Compare already-run fill/cost scenarios in one table."""

    rows = []
    for name, trades in scenario_results.items():
        if trades.empty:
            rows.append(
                {
                    "scenario": name,
                    "trade_count": 0,
                    "total_pnl": 0.0,
                    "mean_return_on_debit": float("nan"),
                    "win_rate": float("nan"),
                }
            )
            continue
        rows.append(
            {
                "scenario": name,
                "trade_count": int(len(trades)),
                "total_pnl": float(trades["pnl"].sum()),
                "mean_return_on_debit": float(
                    trades["return_on_debit"].mean()
                ),
                "win_rate": float((trades["pnl"] > 0).mean()),
            }
        )
    return pd.DataFrame(rows)
