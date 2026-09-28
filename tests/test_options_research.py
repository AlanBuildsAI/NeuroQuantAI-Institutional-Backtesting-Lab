"""Research-protocol tests for Options Lab."""

import pandas as pd

from neuroquant.options_lab import (
    CheapVolBullishConfig,
    bootstrap_trade_returns,
    build_cheap_vol_bullish_entries,
    chronological_trade_split,
)


def test_cheap_vol_signal_is_lagged_and_has_cooldown():
    dates = pd.date_range("2026-01-01", periods=100, freq="B", tz="UTC")
    underlying = pd.DataFrame(
        {"timestamp": dates, "close": [100 + i * 0.5 for i in range(100)]}
    )
    atm_iv = pd.DataFrame(
        {"timestamp": dates, "iv": [0.05] * 100}
    )
    result = build_cheap_vol_bullish_entries(
        underlying,
        atm_iv,
        CheapVolBullishConfig(
            fast_ma=5,
            slow_ma=10,
            rv_window=5,
            max_iv_to_rv=1_000_000.0,
            cooldown_days=10,
        ),
    )
    entries = result[result["entry_signal"]]
    assert len(entries) > 1
    gaps = entries["timestamp"].diff().dropna().dt.days
    assert (gaps >= 10).all()


def test_chronological_split_keeps_order():
    trades = pd.DataFrame(
        {
            "entry_timestamp": pd.to_datetime(
                ["2026-03-01", "2026-01-01", "2026-02-01"], utc=True
            ),
            "return_on_debit": [0.3, 0.1, -0.2],
        }
    )
    train, test = chronological_trade_split(trades, 0.67)
    assert train["entry_timestamp"].max() < test["entry_timestamp"].min()


def test_bootstrap_trade_returns_is_reproducible():
    trades = pd.DataFrame({"return_on_debit": [0.1, -0.05, 0.2, -0.1]})
    a = bootstrap_trade_returns(trades, n_simulations=20, seed=7)
    b = bootstrap_trade_returns(trades, n_simulations=20, seed=7)
    pd.testing.assert_frame_equal(a, b)
