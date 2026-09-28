"""Tests for NeuroQuant Options Lab."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from neuroquant.options_lab import (
    BullCallSelectionConfig,
    BullCallSpread,
    ExecutionConfig,
    OptionContract,
    OptionType,
    Side,
    VerticalBacktestConfig,
    backtest_bull_call_entries,
    black_scholes_greeks,
    black_scholes_price,
    execution_price,
    implied_volatility,
    normalize_option_chain,
    realized_volatility,
    select_bull_call_spread,
    summarize_vertical_backtest,
)


def test_bsm_call_known_value_and_iv_roundtrip():
    price = black_scholes_price(
        spot=100,
        strike=100,
        t_years=1,
        rate=0.05,
        volatility=0.20,
        option_type="call",
    )
    assert price == pytest.approx(10.4506, rel=1e-4)
    iv = implied_volatility(price, 100, 100, 1, 0.05, "call")
    assert iv == pytest.approx(0.20, rel=1e-5)


def test_greeks_are_sensible():
    g = black_scholes_greeks(100, 100, 0.5, 0.04, 0.30, "call")
    assert 0 < g.delta < 1
    assert g.gamma > 0
    assert g.vega_per_vol_point > 0
    assert g.theta_per_day < 0


def test_bull_call_expiry_value_is_capped():
    long = OptionContract("XYZ", date(2027, 1, 15), 100, OptionType.CALL)
    short = OptionContract("XYZ", date(2027, 1, 15), 120, OptionType.CALL)
    spread = BullCallSpread(long, short)
    assert spread.expiry_value_per_share(90) == 0
    assert spread.expiry_value_per_share(110) == 10
    assert spread.expiry_value_per_share(150) == 20


def test_execution_realistic_is_between_mid_and_touch():
    row = {"bid": 4.0, "ask": 5.0, "open_interest": 100, "volume": 10}
    cfg = ExecutionConfig(slippage_half_spread_fraction=0.4)
    buy = execution_price(row, Side.BUY, cfg)
    sell = execution_price(row, Side.SELL, cfg)
    assert 4.5 < buy < 5.0
    assert 4.0 < sell < 4.5


def test_chain_normalization_and_selector():
    ts = "2026-09-28T19:00:00Z"
    expiry = "2027-01-15"
    rows = []
    for strike, delta, bid, ask in [
        (90, 0.75, 15.0, 15.4),
        (100, 0.61, 10.0, 10.4),
        (110, 0.46, 6.0, 6.3),
        (120, 0.31, 3.2, 3.4),
        (130, 0.20, 1.7, 1.9),
    ]:
        rows.append(
            {
                "timestamp": ts,
                "symbol": "XYZ",
                "expiration": expiry,
                "strike": strike,
                "option_type": "C",
                "underlying_price": 105.0,
                "bid": bid,
                "ask": ask,
                "open_interest": 500,
                "volume": 50,
                "delta": delta,
                "iv": 0.35,
            }
        )
    chain = normalize_option_chain(pd.DataFrame(rows))
    pick = select_bull_call_spread(
        chain,
        BullCallSelectionConfig(
            min_dte=60,
            max_dte=150,
            target_long_delta=0.60,
            target_short_delta=0.30,
            min_width_pct_spot=0.10,
            max_width_pct_spot=0.30,
        ),
        ExecutionConfig(max_spread_pct=0.20),
    )
    assert pick["long"]["strike"] == 100
    assert pick["short"]["strike"] == 120


def test_realized_volatility_is_positive_after_warmup():
    close = pd.Series([100 + i + (i % 3) for i in range(40)], dtype=float)
    rv = realized_volatility(close, 10)
    assert rv.iloc[:9].isna().all()
    assert rv.dropna().gt(0).all()


def test_vertical_backtest_uses_observed_quotes_and_closes_profitably():
    entry = pd.Timestamp("2026-09-01T20:00:00Z")
    later = pd.Timestamp("2026-09-21T20:00:00Z")
    expiry = "2026-12-18"

    chain = pd.DataFrame(
        [
            {
                "timestamp": entry,
                "symbol": "XYZ",
                "expiration": expiry,
                "strike": 100,
                "option_type": "call",
                "underlying_price": 105,
                "bid": 9.8,
                "ask": 10.2,
                "open_interest": 1000,
                "volume": 100,
                "delta": 0.60,
                "iv": 0.35,
            },
            {
                "timestamp": entry,
                "symbol": "XYZ",
                "expiration": expiry,
                "strike": 120,
                "option_type": "call",
                "underlying_price": 105,
                "bid": 3.0,
                "ask": 3.2,
                "open_interest": 1000,
                "volume": 100,
                "delta": 0.30,
                "iv": 0.35,
            },
            {
                "timestamp": later,
                "symbol": "XYZ",
                "expiration": expiry,
                "strike": 100,
                "option_type": "call",
                "underlying_price": 122,
                "bid": 23.0,
                "ask": 23.4,
                "open_interest": 1000,
                "volume": 100,
                "delta": 0.82,
                "iv": 0.37,
            },
            {
                "timestamp": later,
                "symbol": "XYZ",
                "expiration": expiry,
                "strike": 120,
                "option_type": "call",
                "underlying_price": 122,
                "bid": 8.0,
                "ask": 8.3,
                "open_interest": 1000,
                "volume": 100,
                "delta": 0.55,
                "iv": 0.37,
            },
        ]
    )

    config = VerticalBacktestConfig(
        selection=BullCallSelectionConfig(
            min_dte=60,
            max_dte=150,
            min_width_pct_spot=0.10,
            max_width_pct_spot=0.30,
        ),
        execution=ExecutionConfig(
            commission_per_contract=0.65,
            slippage_half_spread_fraction=0.0,
            max_spread_pct=0.20,
        ),
        max_hold_days=40,
        profit_target_pct_max=0.40,
        stop_loss_pct_debit=None,
        fill_mode="mid",
    )
    trades = backtest_bull_call_entries(chain, [entry], "XYZ", config)
    assert len(trades) == 1
    assert trades.iloc[0]["pnl"] > 0
    assert trades.iloc[0]["exit_reason"] == "profit_target"
    summary = summarize_vertical_backtest(trades)
    assert summary["trade_count"] == 1
    assert summary["win_rate"] == 1.0
