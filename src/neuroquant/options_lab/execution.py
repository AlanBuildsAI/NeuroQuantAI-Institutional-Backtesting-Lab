"""NBBO-aware execution model for realistic paper and historical fills."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import math

from .models import Side


@dataclass(frozen=True)
class ExecutionConfig:
    """Deterministic execution assumptions.

    slippage_half_spread_fraction controls how far from midpoint a realistic
    fill moves toward the adverse side. Zero is midpoint and one crosses the
    full half-spread to ask for buys or bid for sells.
    """

    commission_per_contract: float = 0.65
    slippage_half_spread_fraction: float = 0.35
    min_open_interest: int = 25
    min_volume: int = 0
    max_spread_pct: float = 0.35
    reject_zero_bid: bool = True

    def __post_init__(self) -> None:
        if self.commission_per_contract < 0:
            raise ValueError("commission_per_contract cannot be negative")
        if not 0.0 <= self.slippage_half_spread_fraction <= 1.0:
            raise ValueError("slippage_half_spread_fraction must be in [0, 1]")
        if self.max_spread_pct <= 0:
            raise ValueError("max_spread_pct must be positive")


def _number(
    row: Mapping[str, Any], key: str, default: float | None = None
) -> float | None:
    value = row.get(key, default)
    if value is None:
        return default
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(value):
        return default
    return value


def quote_mid(row: Mapping[str, Any]) -> float:
    bid = _number(row, "bid")
    ask = _number(row, "ask")
    if bid is None or ask is None:
        raise ValueError("quote requires bid and ask")
    if bid < 0 or ask < 0 or ask < bid:
        raise ValueError("invalid NBBO quote")
    return (bid + ask) / 2.0


def quote_spread_pct(row: Mapping[str, Any]) -> float:
    bid = float(row["bid"])
    ask = float(row["ask"])
    mid = (bid + ask) / 2.0
    return (ask - bid) / mid if mid > 0 else float("inf")


def is_liquid(row: Mapping[str, Any], config: ExecutionConfig) -> bool:
    bid = _number(row, "bid", 0.0) or 0.0
    ask = _number(row, "ask", 0.0) or 0.0
    if ask < bid or ask <= 0:
        return False
    if config.reject_zero_bid and bid <= 0:
        return False
    if quote_spread_pct(row) > config.max_spread_pct:
        return False

    oi = _number(row, "open_interest")
    if oi is not None and oi < config.min_open_interest:
        return False
    volume = _number(row, "volume")
    if volume is not None and volume < config.min_volume:
        return False
    return True


def execution_price(
    row: Mapping[str, Any],
    side: Side | str,
    config: ExecutionConfig,
    mode: str = "realistic",
) -> float:
    """Estimate fill from observed NBBO.

    mid is optimistic benchmarking; realistic applies configured adverse
    slippage; conservative crosses to ask for buys and bid for sells.
    """

    side = Side(side)
    bid = float(row["bid"])
    ask = float(row["ask"])
    if ask < bid or bid < 0:
        raise ValueError("invalid NBBO quote")
    mid = (bid + ask) / 2.0
    half = (ask - bid) / 2.0

    if mode == "mid":
        return float(mid)
    if mode == "conservative":
        return float(ask if side is Side.BUY else bid)
    if mode != "realistic":
        raise ValueError("mode must be mid, realistic, or conservative")

    slip = config.slippage_half_spread_fraction * half
    value = mid + slip if side is Side.BUY else mid - slip
    return float(min(value, ask) if side is Side.BUY else max(value, bid))


def leg_cash_flow(
    row: Mapping[str, Any],
    side: Side | str,
    quantity: int,
    config: ExecutionConfig,
    mode: str = "realistic",
    multiplier: int = 100,
) -> float:
    """Signed dollar cash flow net of commission."""

    if quantity <= 0:
        raise ValueError("quantity must be positive")
    side = Side(side)
    px = execution_price(row, side, config, mode)
    gross = px * multiplier * quantity
    signed = -gross if side is Side.BUY else gross
    commission = config.commission_per_contract * quantity
    return float(signed - commission)


def combo_cash_flow(
    legs: list[tuple[Mapping[str, Any], Side | str, int]],
    config: ExecutionConfig,
    mode: str = "realistic",
    multiplier: int = 100,
) -> float:
    return float(
        sum(
            leg_cash_flow(row, side, qty, config, mode, multiplier)
            for row, side, qty in legs
        )
    )
