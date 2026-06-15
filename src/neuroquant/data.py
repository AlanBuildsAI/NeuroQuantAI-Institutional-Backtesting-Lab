"""Reproducible data inputs for the research lab.

By default we avoid any live or downloaded market data: the lab generates a
*synthetic signal series* with a geometric random walk so that every result
is fully reproducible from a single seed. Framing it as a neutral "signal
series" keeps the project a research demonstration rather than a
market-prediction tool.

An optional CSV loader (:func:`load_csv_series`) lets a user point the same
pipeline at their own local historical research data. It performs the same
data-quality gates and never calls the network or any API.
"""

from __future__ import annotations

from pathlib import Path

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


# Columns the CSV loader will keep if present, in canonical order. A
# ``benchmark_close`` column (if supplied) enables benchmark comparison.
_OPTIONAL_CSV_COLUMNS = ("open", "high", "low", "volume", "benchmark_close")


def load_csv_series(
    path: str | Path,
    date_column: str | None = None,
    close_column: str = "close",
) -> pd.DataFrame:
    """Load an *optional*, user-supplied research series from a local CSV.

    This is a convenience loader for analysts who want to run the same
    reproducible pipeline over their own historical data instead of the
    default synthetic series. It is intentionally minimal and offline: it
    reads a local file only and never contacts the network or any API.

    The CSV must contain a timestamp column (named ``date`` or ``timestamp``
    by default, or given explicitly via ``date_column``) and a ``close``
    column. ``open``, ``high``, ``low`` and ``volume`` are kept if present.

    Quality gates enforced here (raising :class:`ValueError` on failure):

      * the timestamp and ``close`` columns exist and parse,
      * timestamps are unique (no duplicates) and sorted ascending,
      * ``close`` has no missing values,
      * ``close`` is strictly positive.

    Parameters
    ----------
    path:
        Local path to a CSV file.
    date_column:
        Name of the timestamp column. If ``None``, the loader looks for a
        column named ``date`` then ``timestamp`` (case-insensitive).
    close_column:
        Name of the level/price column to analyse (default ``close``).

    Returns
    -------
    pandas.DataFrame
        Frame indexed by a sorted ``DatetimeIndex`` named ``date`` with a
        ``close`` column (plus any available OHLCV columns).
    """
    path = Path(path)
    if not path.exists():
        raise ValueError(f"CSV file not found: {path}")

    raw = pd.read_csv(path)
    return load_csv_frame(raw, date_column=date_column, close_column=close_column)


def load_csv_frame(
    raw: pd.DataFrame,
    date_column: str | None = None,
    close_column: str = "close",
) -> pd.DataFrame:
    """Validate an already-loaded CSV DataFrame and return a clean series.

    This holds the shared parsing and data-quality logic used by
    :func:`load_csv_series` (file path) and by in-memory callers such as the
    Streamlit demo's uploaded-file handler. It never touches disk or the
    network. See :func:`load_csv_series` for the quality gates enforced.
    """
    if not isinstance(raw, pd.DataFrame):
        raise ValueError("Expected a pandas DataFrame of CSV contents.")
    if raw.empty:
        raise ValueError("The CSV contained no rows.")

    lowered = {c.lower(): c for c in raw.columns}

    if date_column is None:
        for candidate in ("date", "timestamp"):
            if candidate in lowered:
                date_column = lowered[candidate]
                break
        else:
            raise ValueError(
                "No timestamp column found. Provide one via date_column, or "
                "name a column 'date' or 'timestamp'. "
                f"Available columns: {list(raw.columns)}."
            )
    elif date_column not in raw.columns:
        raise ValueError(
            f"Timestamp column '{date_column}' not found. "
            f"Available columns: {list(raw.columns)}."
        )

    if close_column not in raw.columns:
        raise ValueError(
            f"Required '{close_column}' column not found. "
            f"Available columns: {list(raw.columns)}."
        )

    timestamps = pd.to_datetime(raw[date_column], errors="coerce")
    if timestamps.isna().any():
        n_bad = int(timestamps.isna().sum())
        raise ValueError(
            f"Column '{date_column}' has {n_bad} unparseable timestamp(s)."
        )
    if timestamps.duplicated().any():
        n_dupes = int(timestamps.duplicated().sum())
        raise ValueError(
            f"Column '{date_column}' contains {n_dupes} duplicate timestamp(s)."
        )

    keep = [close_column] + [
        lowered[c] for c in _OPTIONAL_CSV_COLUMNS if c in lowered
    ]
    frame = raw[keep].copy()
    frame.columns = ["close"] + [
        c for c in _OPTIONAL_CSV_COLUMNS if c in lowered
    ]
    frame.index = pd.DatetimeIndex(timestamps, name="date")
    frame = frame.sort_index()

    if frame["close"].isna().any():
        n_missing = int(frame["close"].isna().sum())
        raise ValueError(f"Column 'close' has {n_missing} missing value(s).")
    if (frame["close"] <= 0).any():
        n_bad = int((frame["close"] <= 0).sum())
        raise ValueError(f"Column 'close' has {n_bad} non-positive value(s).")

    return frame


def data_quality_report(
    frame: pd.DataFrame,
    short_rows_warning: int = 300,
    zero_return_fraction_warning: float = 0.20,
) -> list[str]:
    """Return soft, non-fatal data-quality warnings for a price frame.

    Unlike :func:`validate_price_frame` (which raises on hard failures), this
    surfaces advisory notes — too few rows for stable walk-forward/robustness,
    a high share of exactly-zero returns (stale or low-resolution data), and
    whether a benchmark column is available. Returns an empty list when the
    series looks clean.
    """
    warnings: list[str] = []
    n = len(frame)
    if n < short_rows_warning:
        warnings.append(
            f"Short series: only {n} rows. Walk-forward and robustness "
            "diagnostics need a longer history to be meaningful."
        )

    if "close" in frame.columns and n > 1:
        returns = frame["close"].pct_change().dropna()
        if len(returns):
            zero_fraction = float((returns == 0).mean())
            if zero_fraction > zero_return_fraction_warning:
                warnings.append(
                    f"{zero_fraction:.0%} of returns are exactly zero — the "
                    "series may be stale, padded, or low-resolution."
                )

    if "benchmark_close" not in frame.columns:
        warnings.append(
            "No 'benchmark_close' column — benchmark comparison is unavailable "
            "for this series."
        )

    return warnings
