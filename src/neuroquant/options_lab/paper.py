"""Shadow-ledger utilities for reconciling broker paper fills with realistic fills."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class ShadowTrade:
    trade_id: str
    timestamp: str
    symbol: str
    strategy: str
    action: str
    broker_fill: float
    realistic_fill: float
    conservative_fill: float
    quantity: int
    notes: str = ""


COLUMNS = list(ShadowTrade.__annotations__.keys())


def append_shadow_trade(path: str | Path, trade: ShadowTrade) -> pd.DataFrame:
    """Append one immutable paper-vs-realistic execution record."""

    path = Path(path)
    row = pd.DataFrame([asdict(trade)])
    if path.exists():
        existing = pd.read_csv(path)
        out = pd.concat([existing, row], ignore_index=True)
    else:
        out = row
    if out["trade_id"].duplicated().any():
        raise ValueError("trade_id must be unique")
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return out


def reconciliation_report(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {"fills": 0, "mean_broker_minus_realistic": 0.0, "mean_realistic_minus_conservative": 0.0}
    return {
        "fills": int(len(frame)),
        "mean_broker_minus_realistic": float(
            (frame["broker_fill"] - frame["realistic_fill"]).mean()
        ),
        "mean_realistic_minus_conservative": float(
            (frame["realistic_fill"] - frame["conservative_fill"]).mean()
        ),
    }
