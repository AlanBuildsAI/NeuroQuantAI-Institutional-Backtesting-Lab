"""Reusable, UI-agnostic logic for the interactive Streamlit research demo.

Everything here is plain Python that wraps the existing library modules — no
Streamlit imports — so it can be unit-tested directly and the Streamlit app
stays a thin presentation layer. Nothing here touches the network, writes to
disk, or downloads market data.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .backtest import (
    BacktestConfig,
    build_candidate_configs,
    config_label,
    run_backtest,
)
from .data import generate_synthetic_series, load_csv_frame
from .metrics import compute_kpis
from .regime import summarize_by_regime
from .research import monte_carlo_bootstrap, split_train_test, walk_forward_validation
from .signals import required_warmup
from .stress import cost_sensitivity_analysis, stress_test_summary
from .validation import validate_price_frame

# Metrics shown in the KPI scorecard, in display order, with friendly labels and
# a format hint: "pct" (percentage), "x" (ratio), or "int".
SCORECARD_METRICS: list[tuple[str, str, str]] = [
    ("total_return", "Total return", "pct"),
    ("baseline_return", "Baseline return", "pct"),
    ("excess_return", "Excess vs baseline", "pct"),
    ("annualized_return", "Annualised return", "pct"),
    ("annualized_volatility", "Annualised volatility", "pct"),
    ("sharpe_ratio", "Sharpe", "x"),
    ("sortino_ratio", "Sortino", "x"),
    ("calmar_ratio", "Calmar", "x"),
    ("information_ratio", "Information ratio", "x"),
    ("max_drawdown", "Max drawdown", "pct"),
    ("turnover", "Avg turnover", "x"),
    ("trade_count", "Trade count", "int"),
]


@dataclass(frozen=True)
class ResearchParams:
    """User-tunable research settings collected from the UI."""

    signal_family: str = "trend"
    short_window: int = 20
    long_window: int = 60
    momentum_window: int = 40
    zscore_window: int = 40
    zscore_entry: float = 1.0
    cost_per_trade: float = 0.001
    train_fraction: float = 0.70
    use_volatility_filter: bool = False
    mc_simulations: int = 500

    def to_config(self) -> BacktestConfig:
        """Build the :class:`BacktestConfig` described by these settings."""
        return BacktestConfig(
            short_window=self.short_window,
            long_window=self.long_window,
            signal_family=self.signal_family,
            momentum_window=self.momentum_window,
            zscore_window=self.zscore_window,
            zscore_entry=self.zscore_entry,
            cost_per_trade=self.cost_per_trade,
            use_volatility_filter=self.use_volatility_filter,
        )


def make_synthetic(
    n_days: int = 750,
    seed: int = 42,
    start_value: float = 100.0,
    drift: float = 0.0002,
    volatility: float = 0.012,
) -> pd.DataFrame:
    """Thin wrapper around the seeded synthetic generator (offline, reproducible)."""
    return generate_synthetic_series(
        n_days=n_days,
        seed=seed,
        start_value=start_value,
        drift=drift,
        volatility=volatility,
    )


def load_uploaded_csv(file_like) -> pd.DataFrame:
    """Validate an uploaded CSV (in-memory) using the shared data-quality gates.

    ``file_like`` is anything ``pandas.read_csv`` accepts (e.g. a Streamlit
    ``UploadedFile`` buffer). The file is never written to disk. Raises
    ``ValueError`` with a clear message if the contents are invalid.
    """
    try:
        raw = pd.read_csv(file_like)
    except Exception as exc:  # pragma: no cover - pandas parse errors vary
        raise ValueError(f"Could not read the uploaded file as CSV: {exc}") from exc
    return load_csv_frame(raw)


def _oos_kpis(full_signals: pd.DataFrame, split_at: int) -> dict:
    """KPIs for the out-of-sample slice, with equity rebased to the slice."""
    oos = full_signals.iloc[split_at:].copy()
    oos["strategy_equity"] = (1.0 + oos["strategy_return"]).cumprod()
    oos["baseline_equity"] = (1.0 + oos["baseline_return"]).cumprod()
    return compute_kpis(oos)


def _walk_forward_sizes(n: int) -> tuple[int, int]:
    """Adaptive train/test window sizes for walk-forward on ``n`` rows."""
    train_size = min(250, int(n * 0.5))
    test_size = min(125, int(n * 0.25))
    return train_size, test_size


def run_research(frame: pd.DataFrame, params: ResearchParams) -> dict:
    """Run the full research workflow for one configuration and return results.

    The return dict contains: ``config``, ``label``, ``n_obs``, ``train_size``,
    ``test_size``, ``full_signals``, ``full_kpis``, ``in_sample_kpis``,
    ``out_of_sample_kpis``, ``walk_forward`` (DataFrame or ``None``),
    ``walk_forward_message`` (str or ``None``), ``monte_carlo``,
    ``regime_summary``, ``cost_sensitivity``, ``stress_summary`` and
    ``takeaway``.

    Raises ``ValueError`` (via validation) if the data is too short for the
    chosen settings, so the caller can surface a clean message.
    """
    config = params.to_config()
    warmup = required_warmup(config)

    # Enough rows for the rule, and a training slice that clears warm-up.
    validate_price_frame(frame, min_rows=warmup + 5)
    n = len(frame)
    train_n = int(n * params.train_fraction)
    if train_n <= warmup + 2 or (n - train_n) < 5:
        raise ValueError(
            f"Not enough data for these settings: this configuration needs a "
            f"warm-up of ~{warmup} rows, but the training split only has "
            f"{train_n} of {n} rows. Use more observations, a larger train "
            f"fraction, or smaller windows."
        )

    train, _test = split_train_test(frame, train_fraction=params.train_fraction)
    split_at = len(train)

    best_result = run_backtest(frame, config)
    full_signals = best_result["signals"]
    full_kpis = best_result["kpis"]
    in_sample_kpis = run_backtest(train, config)["kpis"]
    out_of_sample_kpis = _oos_kpis(full_signals, split_at)

    # Walk-forward across a small candidate set; degrade gracefully if short.
    walk_forward = None
    walk_forward_message = None
    train_size, test_size = _walk_forward_sizes(n)
    if train_size + test_size > n:
        walk_forward_message = (
            "Not enough observations for walk-forward validation. Increase the "
            "number of synthetic days (or upload a longer series)."
        )
    else:
        try:
            walk_forward = walk_forward_validation(
                frame,
                configs=build_candidate_configs(cost_per_trade=params.cost_per_trade),
                train_size=train_size,
                test_size=test_size,
                cost_per_trade=params.cost_per_trade,
            )
        except ValueError as exc:
            walk_forward_message = str(exc)

    monte_carlo = monte_carlo_bootstrap(
        full_signals["strategy_return"], n_simulations=params.mc_simulations
    )
    regime_summary = summarize_by_regime(full_signals)
    cost_sensitivity = cost_sensitivity_analysis(frame, config)
    stress_summary = stress_test_summary(frame, config)

    return {
        "config": config,
        "label": config_label(config),
        "n_obs": n,
        "train_size": split_at,
        "test_size": n - split_at,
        "full_signals": full_signals,
        "full_kpis": full_kpis,
        "in_sample_kpis": in_sample_kpis,
        "out_of_sample_kpis": out_of_sample_kpis,
        "walk_forward": walk_forward,
        "walk_forward_message": walk_forward_message,
        "monte_carlo": monte_carlo,
        "regime_summary": regime_summary,
        "cost_sensitivity": cost_sensitivity,
        "stress_summary": stress_summary,
        "takeaway": build_takeaway(
            config_label(config), out_of_sample_kpis, monte_carlo, walk_forward
        ),
    }


def build_takeaway(
    label: str,
    out_of_sample_kpis: dict,
    monte_carlo: dict,
    walk_forward: pd.DataFrame | None,
) -> str:
    """A short, honest one-line analyst summary of a research run."""
    oos = out_of_sample_kpis["sharpe_ratio"]
    p_loss = monte_carlo["probability_of_loss"]
    if walk_forward is not None and not walk_forward.empty:
        held = float((walk_forward["test_sharpe"] > 0).mean())
        wf = f"{held:.0%} of walk-forward folds kept a positive out-of-sample Sharpe; "
    else:
        wf = ""
    return (
        f"The '{label}' candidate had an out-of-sample Sharpe of {oos:.2f}. {wf}"
        f"Monte Carlo resampling implies a ~{p_loss:.0%} chance of a losing path. "
        f"These are research diagnostics on synthetic data, not a forecast."
    )


def _format_value(value: float, kind: str) -> str:
    """Format a KPI value for display given its kind."""
    if kind == "pct":
        return f"{value * 100:+.1f}%" if value < 0 or value > 0 else "0.0%"
    if kind == "int":
        return f"{int(value)}"
    return f"{value:.2f}"


def scorecard_frame(in_sample_kpis: dict, out_of_sample_kpis: dict) -> pd.DataFrame:
    """Tidy in-sample vs out-of-sample KPI table for display."""
    rows = []
    for key, label, kind in SCORECARD_METRICS:
        if key not in in_sample_kpis:
            continue
        rows.append(
            {
                "Metric": label,
                "In-sample": _format_value(in_sample_kpis[key], kind),
                "Out-of-sample": _format_value(out_of_sample_kpis[key], kind),
            }
        )
    return pd.DataFrame(rows)


def equity_frame(signals: pd.DataFrame) -> pd.DataFrame:
    """Strategy vs baseline cumulative return (%) indexed by date, for charts."""
    return pd.DataFrame(
        {
            "Strategy": (signals["strategy_equity"] - 1.0) * 100,
            "Baseline": (signals["baseline_equity"] - 1.0) * 100,
        },
        index=signals.index,
    )


def drawdown_frame(signals: pd.DataFrame) -> pd.DataFrame:
    """Strategy drawdown (%) over time, indexed by date, for charts."""
    equity = signals["strategy_equity"]
    drawdown = (equity / equity.cummax() - 1.0) * 100
    return pd.DataFrame({"Drawdown": drawdown}, index=signals.index)
