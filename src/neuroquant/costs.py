"""Structured transaction-cost model: fee + spread + slippage.

The backtest historically used a single ``cost_per_trade`` scalar. This module
generalises that into explicit fee, spread and slippage components while
preserving the old behaviour: when no structured components are supplied, the
legacy scalar is treated as an all-in fee so results are unchanged.

Costs are charged on *turnover* — the absolute change in exposure from one bar
to the next — so partial sizing and continuous exposures are handled naturally.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class CostRates:
    """Per-turnover cost rates (fractions of traded notional)."""

    fee: float = 0.0
    spread: float = 0.0
    slippage: float = 0.0

    @property
    def total(self) -> float:
        """Combined round-trip cost rate per unit of turnover."""
        return self.fee + self.spread + self.slippage


def resolve_cost_rates(config) -> CostRates:
    """Resolve cost rates from a config, preserving legacy scalar behaviour.

    If ``config.fee`` is ``None`` (the default), the legacy ``cost_per_trade``
    scalar is used as an all-in fee (spread/slippage zero) so existing results
    are reproduced exactly. Otherwise the explicit fee/spread/slippage fields
    are used.
    """
    fee = getattr(config, "fee", None)
    if fee is None:
        return CostRates(fee=float(getattr(config, "cost_per_trade", 0.0)))
    return CostRates(
        fee=float(fee),
        spread=float(getattr(config, "spread", 0.0)),
        slippage=float(getattr(config, "slippage", 0.0)),
    )


def decompose_costs(turnover: pd.Series, rates: CostRates) -> pd.DataFrame:
    """Break per-bar turnover into fee / spread / slippage / total cost series."""
    return pd.DataFrame(
        {
            "fee_cost": turnover * rates.fee,
            "spread_cost": turnover * rates.spread,
            "slippage_cost": turnover * rates.slippage,
            "total_cost": turnover * rates.total,
        },
        index=turnover.index,
    )
