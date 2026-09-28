"""Provider-neutral interfaces for future market-data and paper-broker adapters.

Concrete network adapters are intentionally not bundled in v1. Keeping provider
I/O behind protocols prevents research logic from depending on ThetaData,
Cboe, IBKR, or any single vendor.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

import pandas as pd


class OptionDataProvider(Protocol):
    def chain_snapshot(self, symbol: str, timestamp: datetime) -> pd.DataFrame:
        """Return observed option quotes available at timestamp."""


class PaperBroker(Protocol):
    def submit_vertical(
        self,
        symbol: str,
        expiration: str,
        long_strike: float,
        short_strike: float,
        quantity: int,
        limit_debit: float | None = None,
    ) -> str:
        """Submit a defined-risk paper vertical and return broker order ID."""


@dataclass(frozen=True)
class ProviderCapabilities:
    historical_nbbo: bool
    greeks: bool
    open_interest: bool
    intraday: bool
    paper_orders: bool


RECOMMENDED_CAPABILITY_PROFILES = {
    "historical_options_vendor": ProviderCapabilities(
        historical_nbbo=True,
        greeks=True,
        open_interest=True,
        intraday=True,
        paper_orders=False,
    ),
    "paper_broker": ProviderCapabilities(
        historical_nbbo=False,
        greeks=True,
        open_interest=True,
        intraday=True,
        paper_orders=True,
    ),
}
