"""Offline option-chain ingestion and normalization.

The lab never invents historical contracts. Research data must come from an
observed chain snapshot (CSV/export/API materialized by the user/provider).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "timestamp",
    "symbol",
    "expiration",
    "strike",
    "option_type",
    "underlying_price",
    "bid",
    "ask",
}

OPTIONAL_COLUMNS = (
    "bid_size",
    "ask_size",
    "volume",
    "open_interest",
    "iv",
    "delta",
    "gamma",
    "theta",
    "vega",
    "iv_rank",
)


def normalize_option_chain(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a validated, canonical option-chain dataframe."""

    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("option chain must be a non-empty DataFrame")

    lowered = {str(c).lower().strip(): c for c in frame.columns}
    missing = [c for c in REQUIRED_COLUMNS if c not in lowered]
    if missing:
        raise ValueError(f"option chain missing required columns: {sorted(missing)}")

    keep = list(REQUIRED_COLUMNS) + [c for c in OPTIONAL_COLUMNS if c in lowered]
    out = frame[[lowered[c] for c in keep]].copy()
    out.columns = keep

    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out["expiration"] = pd.to_datetime(out["expiration"], errors="coerce").dt.date
    out["symbol"] = out["symbol"].astype(str).str.upper().str.strip()
    out["option_type"] = out["option_type"].astype(str).str.lower().str.strip()
    out["option_type"] = out["option_type"].replace({"c": "call", "p": "put"})

    numeric = [
        "strike",
        "underlying_price",
        "bid",
        "ask",
        *[c for c in OPTIONAL_COLUMNS if c in out.columns],
    ]
    for col in numeric:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    if out["timestamp"].isna().any() or out["expiration"].isna().any():
        raise ValueError("timestamp/expiration contain invalid values")
    if (~out["option_type"].isin(["call", "put"])).any():
        raise ValueError("option_type must be call/put (or C/P)")
    if (out["strike"] <= 0).any() or (out["underlying_price"] <= 0).any():
        raise ValueError("strike and underlying_price must be positive")
    if (out["bid"] < 0).any() or (out["ask"] < 0).any():
        raise ValueError("bid/ask cannot be negative")
    if (out["ask"] < out["bid"]).any():
        raise ValueError("crossed quotes found: ask below bid")

    out["dte"] = (
        pd.to_datetime(out["expiration"])
        - out["timestamp"].dt.tz_convert(None).dt.normalize()
    ).dt.days
    out = out[out["dte"] >= 0].copy()

    key = ["timestamp", "symbol", "expiration", "strike", "option_type"]
    if out.duplicated(key).any():
        raise ValueError("duplicate contract snapshots found")

    return out.sort_values(key).reset_index(drop=True)


def load_option_chain_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise ValueError(f"option-chain CSV not found: {path}")
    return normalize_option_chain(pd.read_csv(path))


def chain_at_or_before(
    frame: pd.DataFrame,
    timestamp,
    symbol: str | None = None,
) -> pd.DataFrame:
    """Latest complete-ish snapshot at or before a timestamp.

    For intraday vendor data, contracts can update at slightly different
    timestamps. We therefore take the latest row per contract at/before the
    requested instant rather than requiring a single exact timestamp.
    """

    data = normalize_option_chain(frame)
    ts = pd.Timestamp(timestamp)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    data = data[data["timestamp"] <= ts]
    if symbol:
        data = data[data["symbol"] == symbol.upper()]
    if data.empty:
        return data
    key = ["symbol", "expiration", "strike", "option_type"]
    idx = data.groupby(key)["timestamp"].idxmax()
    return data.loc[idx].sort_values(["expiration", "option_type", "strike"]).reset_index(drop=True)
