"""Event-driven bull-call-spread backtester using observed option quotes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import normalize_option_chain
from .execution import ExecutionConfig, execution_price
from .models import Side
from .selector import BullCallSelectionConfig, select_bull_call_spread


@dataclass(frozen=True)
class VerticalBacktestConfig:
    """Rules for a defined-risk long vertical backtest."""

    selection: BullCallSelectionConfig = BullCallSelectionConfig()
    execution: ExecutionConfig = ExecutionConfig()
    quantity: int = 1
    max_hold_days: int = 60
    exit_dte: int = 21
    profit_target_pct_max: float | None = 0.50
    stop_loss_pct_debit: float | None = 0.60
    fill_mode: str = "realistic"

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")
        if self.max_hold_days <= 0:
            raise ValueError("max_hold_days must be positive")
        if self.exit_dte < 0:
            raise ValueError("exit_dte cannot be negative")


def _same_contract_rows(
    chain: pd.DataFrame,
    symbol: str,
    expiration,
    strike: float,
    timestamp,
) -> pd.DataFrame:
    rows = chain[
        (chain["symbol"] == symbol)
        & (chain["expiration"] == expiration)
        & (chain["strike"] == strike)
        & (chain["option_type"] == "call")
        & (chain["timestamp"] <= timestamp)
    ]
    if rows.empty:
        return rows
    return rows.sort_values("timestamp").tail(1)


def _spread_close_value(
    chain: pd.DataFrame,
    symbol: str,
    expiration,
    long_strike: float,
    short_strike: float,
    timestamp,
    config: VerticalBacktestConfig,
) -> tuple[float, dict] | None:
    long_rows = _same_contract_rows(
        chain, symbol, expiration, long_strike, timestamp
    )
    short_rows = _same_contract_rows(
        chain, symbol, expiration, short_strike, timestamp
    )
    if long_rows.empty or short_rows.empty:
        return None
    long = long_rows.iloc[0].to_dict()
    short = short_rows.iloc[0].to_dict()

    # To close: sell the long and buy back the short.
    long_px = execution_price(long, Side.SELL, config.execution, config.fill_mode)
    short_px = execution_price(short, Side.BUY, config.execution, config.fill_mode)
    credit = long_px - short_px
    commission = 2 * config.execution.commission_per_contract
    dollars = (
        credit * 100.0 - commission
    ) * config.quantity
    return float(dollars), {"long": long, "short": short}


def backtest_bull_call_entries(
    chain: pd.DataFrame,
    entry_timestamps,
    symbol: str,
    config: VerticalBacktestConfig | None = None,
) -> pd.DataFrame:
    """Backtest independent bull-call entries on specified timestamps.

    The function uses only chain observations available at each entry and each
    later exit evaluation. It makes no synthetic fill at a missing quote.
    """

    config = config or VerticalBacktestConfig()
    data = normalize_option_chain(chain)
    data = data[data["symbol"] == symbol.upper()].copy()
    if data.empty:
        raise ValueError(f"no chain data for {symbol}")

    results = []
    for raw_entry in entry_timestamps:
        entry_ts = pd.Timestamp(raw_entry)
        entry_ts = (
            entry_ts.tz_localize("UTC")
            if entry_ts.tzinfo is None
            else entry_ts.tz_convert("UTC")
        )
        available = data[data["timestamp"] <= entry_ts].copy()
        if available.empty:
            continue
        key = ["expiration", "strike", "option_type"]
        available = available.loc[
            available.groupby(key)["timestamp"].idxmax()
        ].reset_index(drop=True)

        try:
            pick = select_bull_call_spread(
                available, config.selection, config.execution
            )
        except ValueError:
            continue

        long = pick["long"]
        short = pick["short"]
        long_px = execution_price(
            long, Side.BUY, config.execution, config.fill_mode
        )
        short_px = execution_price(
            short, Side.SELL, config.execution, config.fill_mode
        )
        debit_per_share = long_px - short_px
        if debit_per_share <= 0:
            continue

        commissions_open = 2 * config.execution.commission_per_contract
        initial_debit = (
            debit_per_share * 100.0 + commissions_open
        ) * config.quantity
        width = float(pick["width"])
        max_value = width * 100.0 * config.quantity
        max_profit = max_value - initial_debit
        breakeven = float(long["strike"]) + initial_debit / (100.0 * config.quantity)

        expiry = pd.Timestamp(pick["expiration"], tz="UTC")
        last_allowed = min(
            entry_ts + pd.Timedelta(days=config.max_hold_days),
            expiry,
        )
        checkpoints = sorted(
            data[
                (data["timestamp"] > entry_ts)
                & (data["timestamp"] <= last_allowed)
            ]["timestamp"].unique()
        )

        exit_cash = None
        exit_ts = None
        exit_reason = None
        for ts in checkpoints:
            close = _spread_close_value(
                data,
                symbol.upper(),
                pick["expiration"],
                float(long["strike"]),
                float(short["strike"]),
                pd.Timestamp(ts),
                config,
            )
            if close is None:
                continue
            value, quotes = close
            pnl = value - initial_debit
            current_dte = int(quotes["long"]["dte"])

            if (
                config.profit_target_pct_max is not None
                and max_profit > 0
                and pnl >= max_profit * config.profit_target_pct_max
            ):
                exit_cash, exit_ts, exit_reason = value, pd.Timestamp(ts), "profit_target"
                break
            if (
                config.stop_loss_pct_debit is not None
                and pnl <= -initial_debit * config.stop_loss_pct_debit
            ):
                exit_cash, exit_ts, exit_reason = value, pd.Timestamp(ts), "stop_loss"
                break
            if current_dte <= config.exit_dte:
                exit_cash, exit_ts, exit_reason = value, pd.Timestamp(ts), "exit_dte"
                break

        if exit_cash is None:
            # Use the latest observable quote before the configured horizon.
            close = _spread_close_value(
                data,
                symbol.upper(),
                pick["expiration"],
                float(long["strike"]),
                float(short["strike"]),
                last_allowed,
                config,
            )
            if close is None:
                continue
            exit_cash, quotes = close
            exit_ts = pd.Timestamp(quotes["long"]["timestamp"])
            exit_reason = "max_hold"

        pnl = float(exit_cash - initial_debit)
        results.append(
            {
                "symbol": symbol.upper(),
                "entry_timestamp": entry_ts,
                "exit_timestamp": exit_ts,
                "expiration": pick["expiration"],
                "long_strike": float(long["strike"]),
                "short_strike": float(short["strike"]),
                "quantity": config.quantity,
                "entry_debit": float(initial_debit),
                "breakeven": breakeven,
                "max_profit": float(max_profit),
                "exit_value": float(exit_cash),
                "pnl": pnl,
                "return_on_debit": pnl / initial_debit,
                "exit_reason": exit_reason,
            }
        )

    return pd.DataFrame(results)


def summarize_vertical_backtest(trades: pd.DataFrame) -> dict:
    """Compact strategy-level diagnostics for option-trade outcomes."""

    if trades.empty:
        return {
            "trade_count": 0,
            "win_rate": float("nan"),
            "mean_return_on_debit": float("nan"),
            "median_return_on_debit": float("nan"),
            "total_pnl": 0.0,
            "profit_factor": float("nan"),
        }
    pnl = trades["pnl"].astype(float)
    wins = pnl[pnl > 0].sum()
    losses = -pnl[pnl < 0].sum()
    return {
        "trade_count": int(len(trades)),
        "win_rate": float((pnl > 0).mean()),
        "mean_return_on_debit": float(trades["return_on_debit"].mean()),
        "median_return_on_debit": float(trades["return_on_debit"].median()),
        "total_pnl": float(pnl.sum()),
        "profit_factor": float(wins / losses) if losses > 0 else float("inf"),
    }
