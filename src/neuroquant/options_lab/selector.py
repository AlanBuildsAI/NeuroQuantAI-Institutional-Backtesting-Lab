"""Contract selection for reproducible vertical-spread research."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .execution import ExecutionConfig, is_liquid


@dataclass(frozen=True)
class BullCallSelectionConfig:
    min_dte: int = 60
    max_dte: int = 150
    target_long_delta: float = 0.60
    target_short_delta: float = 0.30
    delta_tolerance: float = 0.20
    min_width_pct_spot: float = 0.08
    max_width_pct_spot: float = 0.40
    require_iv_below_rv: bool = False

    def __post_init__(self) -> None:
        if self.min_dte < 1 or self.max_dte < self.min_dte:
            raise ValueError("invalid DTE range")
        if not 0 < self.target_short_delta < self.target_long_delta < 1:
            raise ValueError("target deltas must satisfy 0 < short < long < 1")
        if self.delta_tolerance <= 0:
            raise ValueError("delta_tolerance must be positive")


def _candidate_calls(
    chain: pd.DataFrame,
    selection: BullCallSelectionConfig,
    execution: ExecutionConfig,
) -> pd.DataFrame:
    calls = chain[
        (chain["option_type"] == "call")
        & chain["dte"].between(selection.min_dte, selection.max_dte)
    ].copy()
    if calls.empty:
        return calls
    calls = calls[
        calls.apply(lambda row: is_liquid(row, execution), axis=1)
    ].copy()
    if selection.require_iv_below_rv:
        if not {"iv", "realized_vol"}.issubset(calls.columns):
            raise ValueError("IV<RV selection requires iv and realized_vol columns")
        calls = calls[calls["iv"] < calls["realized_vol"]].copy()
    return calls


def select_bull_call_spread(
    chain: pd.DataFrame,
    selection: BullCallSelectionConfig | None = None,
    execution: ExecutionConfig | None = None,
) -> dict:
    """Select a liquid bull call spread without peeking forward.

    Expiration is chosen by closeness to the midpoint of the DTE window.
    Within that expiration, long and short legs are chosen by target delta.
    """

    selection = selection or BullCallSelectionConfig()
    execution = execution or ExecutionConfig()
    calls = _candidate_calls(chain, selection, execution)
    if calls.empty:
        raise ValueError("no liquid call candidates satisfy the filters")
    if "delta" not in calls.columns or calls["delta"].isna().all():
        raise ValueError("delta is required for deterministic contract selection")

    target_dte = (selection.min_dte + selection.max_dte) / 2.0
    expiries = (
        calls.groupby("expiration", as_index=False)["dte"]
        .median()
        .assign(distance=lambda x: (x["dte"] - target_dte).abs())
        .sort_values(["distance", "dte"])
    )
    expiry = expiries.iloc[0]["expiration"]
    sub = calls[calls["expiration"] == expiry].copy()

    sub["long_error"] = (sub["delta"] - selection.target_long_delta).abs()
    long = sub.sort_values(["long_error", "strike"]).iloc[0]
    if long["long_error"] > selection.delta_tolerance:
        raise ValueError("no long call close enough to target delta")

    spot = float(long["underlying_price"])
    shorts = sub[sub["strike"] > float(long["strike"])].copy()
    if shorts.empty:
        raise ValueError("no higher-strike call available for short leg")
    shorts["width_pct"] = (shorts["strike"] - float(long["strike"])) / spot
    shorts = shorts[
        shorts["width_pct"].between(
            selection.min_width_pct_spot, selection.max_width_pct_spot
        )
    ].copy()
    if shorts.empty:
        raise ValueError("no short call satisfies spread-width constraints")

    shorts["short_error"] = (shorts["delta"] - selection.target_short_delta).abs()
    short = shorts.sort_values(["short_error", "strike"]).iloc[0]
    if short["short_error"] > selection.delta_tolerance:
        raise ValueError("no short call close enough to target delta")

    return {
        "long": long.to_dict(),
        "short": short.to_dict(),
        "expiration": expiry,
        "spot": spot,
        "width": float(short["strike"] - long["strike"]),
    }
