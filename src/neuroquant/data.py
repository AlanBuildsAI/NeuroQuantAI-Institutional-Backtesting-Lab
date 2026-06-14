"""Reproducible synthetic time-series generation.

We deliberately avoid any live or downloaded market data. Instead we
generate a *synthetic signal series* with a geometric random walk so that
every result in this project is fully reproducible from a single seed.
Framing it as a neutral "signal series" keeps the project an analytics
demonstration rather than a market-prediction tool.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_synthetic_series(
    n_days: int = 750,
    seed: int = 42,
    start_value: float = 100.0,
    drift: float = 0.0002,
    volatility: float = 0.012,
    start_date: str = "2021-01-01",
) -> pd.DataFrame:
    """Generate a reproducible synthetic signal series.

    The series is produced with a geometric random walk:
        value_t = value_{t-1} * exp(drift + volatility * z_t)
    where z_t are standard normal draws from a seeded generator.

    Parameters
    ----------
    n_days:
        Number of observations (business days) to generate.
    seed:
        Random seed. The same seed always returns the same series.
    start_value:
        Initial level of the series.
    drift:
        Per-step deterministic drift term.
    volatility:
        Per-step volatility (standard deviation of log returns).
    start_date:
        Calendar start date for the date index.

    Returns
    -------
    pandas.DataFrame
        A frame indexed by a business-day ``DatetimeIndex`` with a single
        ``close`` column holding the synthetic signal level.
    """
    if n_days <= 0:
        raise ValueError("n_days must be a positive integer.")

    rng = np.random.default_rng(seed)
    shocks = rng.standard_normal(n_days)
    log_returns = drift + volatility * shocks
    levels = start_value * np.exp(np.cumsum(log_returns))

    index = pd.bdate_range(start=start_date, periods=n_days, name="date")
    return pd.DataFrame({"close": levels}, index=index)
