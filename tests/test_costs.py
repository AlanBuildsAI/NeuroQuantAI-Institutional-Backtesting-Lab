"""Tests for the structured cost model."""

import pandas as pd
import pytest

from neuroquant.backtest import BacktestConfig
from neuroquant.costs import CostRates, decompose_costs, resolve_cost_rates


def test_legacy_scalar_path():
    rates = resolve_cost_rates(BacktestConfig(cost_per_trade=0.0012))
    assert rates.fee == 0.0012
    assert rates.spread == 0.0
    assert rates.slippage == 0.0
    assert rates.total == 0.0012


def test_structured_components_sum():
    cfg = BacktestConfig(fee=0.0003, spread=0.0002, slippage=0.0001)
    rates = resolve_cost_rates(cfg)
    assert rates.total == pytest.approx(0.0006)


def test_decompose_costs_matches_total():
    turnover = pd.Series([0.0, 1.0, 0.5, 1.0])
    rates = CostRates(fee=0.001, spread=0.0005, slippage=0.0005)
    table = decompose_costs(turnover, rates)
    assert {"fee_cost", "spread_cost", "slippage_cost", "total_cost"}.issubset(
        table.columns
    )
    recombined = (
        table["fee_cost"] + table["spread_cost"] + table["slippage_cost"]
    )
    assert (abs(recombined - table["total_cost"]) < 1e-12).all()
