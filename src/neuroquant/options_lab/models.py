"""Core option-domain models used by NeuroQuant Options Lab."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum


class OptionType(str, Enum):
    CALL = "call"
    PUT = "put"


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass(frozen=True)
class OptionContract:
    """Canonical option identity without vendor-specific IDs."""

    symbol: str
    expiration: date
    strike: float
    option_type: OptionType
    multiplier: int = 100

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("symbol must be non-empty")
        if self.strike <= 0:
            raise ValueError("strike must be positive")
        if self.multiplier <= 0:
            raise ValueError("multiplier must be positive")


@dataclass(frozen=True)
class OptionQuote:
    """One NBBO-style option quote snapshot."""

    contract: OptionContract
    timestamp: datetime
    underlying_price: float
    bid: float
    ask: float
    bid_size: int | None = None
    ask_size: int | None = None
    volume: int | None = None
    open_interest: int | None = None
    iv: float | None = None
    delta: float | None = None
    gamma: float | None = None
    theta: float | None = None
    vega: float | None = None

    def __post_init__(self) -> None:
        if self.underlying_price <= 0:
            raise ValueError("underlying_price must be positive")
        if self.bid < 0 or self.ask < 0:
            raise ValueError("bid/ask cannot be negative")
        if self.ask < self.bid:
            raise ValueError("crossed quote: ask is below bid")

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @property
    def spread_pct(self) -> float:
        return self.spread / self.mid if self.mid > 0 else float("inf")


@dataclass(frozen=True)
class OptionLeg:
    contract: OptionContract
    side: Side
    quantity: int = 1

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")


@dataclass(frozen=True)
class BullCallSpread:
    """Long lower-strike call plus short higher-strike call."""

    long_call: OptionContract
    short_call: OptionContract
    quantity: int = 1

    def __post_init__(self) -> None:
        if self.long_call.symbol != self.short_call.symbol:
            raise ValueError("spread legs must share the same underlying")
        if self.long_call.expiration != self.short_call.expiration:
            raise ValueError("spread legs must share the same expiration")
        if self.long_call.option_type is not OptionType.CALL:
            raise ValueError("long leg must be a call")
        if self.short_call.option_type is not OptionType.CALL:
            raise ValueError("short leg must be a call")
        if self.long_call.strike >= self.short_call.strike:
            raise ValueError("bull call long strike must be below short strike")
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")

    @property
    def width(self) -> float:
        return self.short_call.strike - self.long_call.strike

    def expiry_value_per_share(self, spot: float) -> float:
        long_intrinsic = max(spot - self.long_call.strike, 0.0)
        short_intrinsic = max(spot - self.short_call.strike, 0.0)
        return long_intrinsic - short_intrinsic
