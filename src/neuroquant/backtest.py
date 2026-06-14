"""Experiment engine: signal families, costs, baseline, and sweeps.

This module turns a validated signal series into a reproducible experiment.
It supports several candidate signal families (trend, momentum, mean
reversion, and a composite of all three) defined in :mod:`neuroquant.signals`.
The analytical point is the *workflow*: a target position is built from past
data, shifted one bar to avoid look-ahead bias, charged a simple transaction
cost on every change, and compared against a transparent buy-and-hold
BASELINE.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .metrics import compute_kpis
from .signals import (
    SIGNAL_FAMILIES,
    build_signal_frame,
    required_warmup,
)
from .validation import ValidationError, validate_price_frame, validate_window_config


@dataclass(frozen=True)
class BacktestConfig:
    """Configuration for a single backtest run.

    Attributes
    ----------
    signal_family:
        One of ``trend``, ``momentum``, ``mean_reversion`` or ``composite``.
    short_window / long_window:
        Fast / slow moving-average windows for the trend (and composite) family.
    momentum_window:
        Look-back for the momentum family.
    zscore_window / zscore_entry:
        Window and (positive) entry threshold for the mean-reversion family;
        the rule goes long when the z-score falls below ``-zscore_entry``.
    cost_per_trade:
        Round-trip transaction cost + slippage applied as a fraction of the
        position value each time the position changes (e.g. 0.001 = 0.1%).
    use_volatility_filter / vol_filter_window / vol_filter_quantile:
        Optional trailing volatility filter that damps exposure when rolling
        volatility exceeds the given expanding quantile of past volatility.
    """

    # short_window / long_window come first so positional construction —
    # BacktestConfig(20, 60) — keeps mapping to the trend windows as before.
    short_window: int = 20
    long_window: int = 60
    signal_family: str = "trend"
    momentum_window: int = 60
    zscore_window: int = 20
    zscore_entry: float = 1.0
    cost_per_trade: float = 0.001
    use_volatility_filter: bool = False
    vol_filter_window: int = 60
    vol_filter_quantile: float = 0.8


def config_label(config: BacktestConfig) -> str:
    """Short, human-readable label for a configuration."""
    if config.signal_family == "trend":
        label = f"trend {config.short_window}/{config.long_window}"
    elif config.signal_family == "momentum":
        label = f"momentum {config.momentum_window}"
    elif config.signal_family == "mean_reversion":
        label = f"meanrev {config.zscore_window}/{config.zscore_entry:g}"
    else:
        label = f"composite {config.short_window}/{config.long_window}"
    if config.use_volatility_filter:
        label += " +volfilter"
    return label


def _validate_config(config: BacktestConfig) -> None:
    """Light parameter validation per family (data length is checked later)."""
    if config.signal_family not in SIGNAL_FAMILIES:
        raise ValidationError(
            f"Unknown signal_family '{config.signal_family}'. "
            f"Choose one of: {', '.join(SIGNAL_FAMILIES)}."
        )
    if config.signal_family in ("trend", "composite"):
        validate_window_config(config.short_window, config.long_window)
    if config.signal_family in ("momentum", "composite"):
        if config.momentum_window <= 0:
            raise ValidationError("momentum_window must be positive.")
    if config.signal_family in ("mean_reversion", "composite"):
        if config.zscore_window <= 0:
            raise ValidationError("zscore_window must be positive.")
        if config.zscore_entry <= 0:
            raise ValidationError("zscore_entry must be positive.")


def generate_signals(frame: pd.DataFrame, config: BacktestConfig) -> pd.DataFrame:
    """Build positions and returns for one configuration.

    Steps:
      * build the family's unshifted target position (see
        :func:`neuroquant.signals.build_signal_frame`),
      * SHIFT the target by one bar so today's decision uses only data
        available up to yesterday — this avoids look-ahead bias,
      * apply a transaction cost whenever the position changes,
      * compute strategy and baseline returns plus cumulative equity.

    Returns a DataFrame with the family's component columns plus
    ``position``, ``market_return``, ``strategy_return``, ``baseline_return``,
    ``cost``, ``strategy_equity`` and ``baseline_equity``.
    """
    _validate_config(config)
    validate_price_frame(frame, min_rows=required_warmup(config) + 2)

    signal_frame = build_signal_frame(frame, config)

    data = frame.copy()
    for col in signal_frame.columns:
        if col != "target_position":
            data[col] = signal_frame[col]

    # Shift by one bar: act only on information from the previous bar.
    data["position"] = signal_frame["target_position"].shift(1).fillna(0.0)

    data["market_return"] = data["close"].pct_change().fillna(0.0)

    position_change = data["position"].diff().abs().fillna(0.0)
    data["cost"] = position_change * config.cost_per_trade

    data["strategy_return"] = (
        data["position"] * data["market_return"] - data["cost"]
    )
    data["baseline_return"] = data["market_return"]

    data["strategy_equity"] = (1.0 + data["strategy_return"]).cumprod()
    data["baseline_equity"] = (1.0 + data["baseline_return"]).cumprod()

    return data


def run_backtest(frame: pd.DataFrame, config: BacktestConfig) -> dict:
    """Run a single backtest and return signals plus KPI metrics.

    Returns
    -------
    dict
        ``{"config": BacktestConfig, "signals": DataFrame, "kpis": dict}``.
    """
    signals = generate_signals(frame, config)
    kpis = compute_kpis(signals)
    return {"config": config, "signals": signals, "kpis": kpis}


def run_parameter_sweep(
    frame: pd.DataFrame,
    short_windows: list[int],
    long_windows: list[int],
    cost_per_trade: float = 0.001,
) -> pd.DataFrame:
    """Sweep the **trend** family across short/long window pairs.

    Every valid (short, long) combination where short < long is evaluated. The
    result is one row per scenario with its configuration and KPIs, sorted by
    Sharpe ratio (best first). This trend-only grid powers the Sharpe heatmap.

    Returns
    -------
    pandas.DataFrame
        Tidy frame with columns: short_window, long_window, cost_per_trade,
        and all KPI columns from :func:`neuroquant.metrics.compute_kpis`.
    """
    rows: list[dict] = []
    for short_window in short_windows:
        for long_window in long_windows:
            if short_window >= long_window:
                continue
            config = BacktestConfig(
                signal_family="trend",
                short_window=short_window,
                long_window=long_window,
                cost_per_trade=cost_per_trade,
            )
            result = run_backtest(frame, config)
            row = {
                "short_window": short_window,
                "long_window": long_window,
                "cost_per_trade": cost_per_trade,
            }
            row.update(result["kpis"])
            rows.append(row)

    if not rows:
        raise ValueError(
            "Parameter sweep produced no valid scenarios. Ensure at least "
            "one short_window is smaller than a long_window."
        )

    summary = pd.DataFrame(rows)
    return summary.sort_values("sharpe_ratio", ascending=False).reset_index(
        drop=True
    )


def build_candidate_configs(cost_per_trade: float = 0.001) -> list[BacktestConfig]:
    """A modest, fast candidate set spanning every signal family.

    Kept deliberately small so the in-sample sweep and walk-forward stay quick
    while still comparing genuinely different signal families.
    """
    configs: list[BacktestConfig] = []

    # Trend family — a few window pairs.
    for short_window, long_window in [(10, 40), (20, 60), (20, 90), (30, 90)]:
        configs.append(
            BacktestConfig(
                signal_family="trend",
                short_window=short_window,
                long_window=long_window,
                cost_per_trade=cost_per_trade,
            )
        )

    # Momentum family — a few look-backs.
    for momentum_window in [20, 60, 120]:
        configs.append(
            BacktestConfig(
                signal_family="momentum",
                momentum_window=momentum_window,
                cost_per_trade=cost_per_trade,
            )
        )

    # Mean-reversion family — window / entry-threshold combinations.
    for zscore_window in [20, 40]:
        for zscore_entry in [1.0, 1.5]:
            configs.append(
                BacktestConfig(
                    signal_family="mean_reversion",
                    zscore_window=zscore_window,
                    zscore_entry=zscore_entry,
                    cost_per_trade=cost_per_trade,
                )
            )

    # Composite family — with and without the volatility filter.
    for use_filter in (False, True):
        configs.append(
            BacktestConfig(
                signal_family="composite",
                short_window=20,
                long_window=60,
                momentum_window=60,
                zscore_window=20,
                zscore_entry=1.0,
                cost_per_trade=cost_per_trade,
                use_volatility_filter=use_filter,
            )
        )

    return configs


# Identity columns describing each config in a multi-family sweep table.
CONFIG_COLUMNS = (
    "label",
    "signal_family",
    "short_window",
    "long_window",
    "momentum_window",
    "zscore_window",
    "zscore_entry",
    "use_volatility_filter",
    "cost_per_trade",
)


def run_config_sweep(
    frame: pd.DataFrame, configs: list[BacktestConfig]
) -> pd.DataFrame:
    """Evaluate an explicit list of configurations across signal families.

    Returns one row per configuration with its identity columns and KPIs,
    sorted by Sharpe ratio (best first). This is the multi-family selection
    table used to pick a candidate configuration.
    """
    if not configs:
        raise ValueError("run_config_sweep requires at least one configuration.")

    rows: list[dict] = []
    for config in configs:
        result = run_backtest(frame, config)
        row = {
            "label": config_label(config),
            "signal_family": config.signal_family,
            "short_window": config.short_window,
            "long_window": config.long_window,
            "momentum_window": config.momentum_window,
            "zscore_window": config.zscore_window,
            "zscore_entry": config.zscore_entry,
            "use_volatility_filter": config.use_volatility_filter,
            "cost_per_trade": config.cost_per_trade,
        }
        row.update(result["kpis"])
        rows.append(row)

    summary = pd.DataFrame(rows)
    return summary.sort_values("sharpe_ratio", ascending=False).reset_index(
        drop=True
    )


def config_from_row(row) -> BacktestConfig:
    """Rebuild a :class:`BacktestConfig` from a ``run_config_sweep`` row."""
    return BacktestConfig(
        signal_family=str(row.signal_family),
        short_window=int(row.short_window),
        long_window=int(row.long_window),
        momentum_window=int(row.momentum_window),
        zscore_window=int(row.zscore_window),
        zscore_entry=float(row.zscore_entry),
        cost_per_trade=float(row.cost_per_trade),
        use_volatility_filter=bool(row.use_volatility_filter),
    )
