"""Tests for cost sensitivity and stress diagnostics."""

from neuroquant.backtest import BacktestConfig
from neuroquant.stress import (
    DEFAULT_COST_LADDER,
    cost_sensitivity_analysis,
    stress_test_summary,
)


def test_cost_sensitivity_returns_expected_rows(sample_data):
    config = BacktestConfig(20, 60)
    table = cost_sensitivity_analysis(sample_data, config)
    expected = {
        "cost",
        "total_return",
        "sharpe_ratio",
        "max_drawdown",
        "trade_count",
        "turnover",
    }
    assert expected.issubset(table.columns)
    assert len(table) == len(DEFAULT_COST_LADDER)
    # Costs are reported in increasing order.
    assert table["cost"].tolist() == sorted(table["cost"].tolist())


def test_higher_costs_do_not_improve_net_return(sample_data):
    """A deterministic check: zero cost must beat the highest cost level."""
    config = BacktestConfig(20, 60)
    table = cost_sensitivity_analysis(sample_data, config).set_index("cost")
    cheapest = table.loc[table.index.min(), "total_return"]
    priciest = table.loc[table.index.max(), "total_return"]
    assert cheapest >= priciest


def test_stress_summary_has_scenarios(sample_data):
    config = BacktestConfig(20, 60)
    table = stress_test_summary(sample_data, config)
    assert {"scenario", "total_return", "sharpe_ratio", "max_drawdown"}.issubset(
        table.columns
    )
    scenarios = set(table["scenario"])
    assert {"baseline", "higher_costs", "volatility_shock", "adverse_drift",
            "amplified_downside"}.issubset(scenarios)


def test_adverse_drift_hurts_relative_to_baseline(sample_data):
    config = BacktestConfig(20, 60)
    table = stress_test_summary(sample_data, config).set_index("scenario")
    assert table.loc["adverse_drift", "total_return"] <= table.loc[
        "baseline", "total_return"
    ]
