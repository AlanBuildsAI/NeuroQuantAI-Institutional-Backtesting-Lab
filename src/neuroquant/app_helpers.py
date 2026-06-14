"""Reusable, UI-agnostic logic for the interactive Streamlit research demo.

Everything here is plain Python that wraps the existing library modules — no
Streamlit imports — so it can be unit-tested directly and the Streamlit app
stays a thin presentation layer. Nothing here touches the network, writes to
disk, or downloads market data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .backtest import (
    BacktestConfig,
    build_candidate_configs,
    config_label,
    run_backtest,
)
from .data import data_quality_report, generate_synthetic_series, load_csv_frame
from .metrics import compute_extended_kpis, compute_kpis
from .regime import summarize_by_regime
from .research import (
    monte_carlo_bootstrap,
    overfit_gap,
    robustness_score,
    split_train_test,
    walk_forward_validation,
)
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


# --- Asset & execution assumption profiles ----------------------------------
#
# These are *research assumptions only*. No profile connects to a live broker,
# exchange, API, or market-data feed. Asset profiles nudge synthetic-data
# defaults and add explanatory context; execution profiles set simulated cost
# assumptions that genuinely flow into the backtest via the per-trade cost.

# Execution profiles → simulated round-trip cost components (fractions). The
# effective per-trade cost used by the backtest is fee + spread + slippage.
# "Custom assumptions" (None) hands control back to the user.
EXECUTION_PROFILES: dict[str, dict | None] = {
    "Generic low-cost broker": {
        "fee": 0.0003,
        "spread": 0.0001,
        "slippage": 0.0001,
        "turnover_warn": 0.50,
        "note": "Low fees and tight spreads, typical of liquid equity / ETF venues.",
    },
    "Generic crypto exchange": {
        "fee": 0.0010,
        "spread": 0.0008,
        "slippage": 0.0006,
        "turnover_warn": 0.40,
        "note": "Higher taker fees and wider spreads than a low-cost broker.",
    },
    "Generic high-spread venue": {
        "fee": 0.0015,
        "spread": 0.0030,
        "slippage": 0.0015,
        "turnover_warn": 0.30,
        "note": "Wide spreads / thin liquidity; costs dominate frequent trading.",
    },
    "Custom assumptions": None,
}

# Asset profiles → synthetic-data defaults and a suggested execution profile.
# They never imply live data; they only set starting assumptions and context.
ASSET_PROFILES: dict[str, dict] = {
    "Generic equity / ETF": {
        "volatility": 0.012,
        "drift": 0.0002,
        "execution": "Generic low-cost broker",
        "note": "Broad, relatively calm series; low trading costs are typical.",
    },
    "Crypto spot": {
        "volatility": 0.035,
        "drift": 0.0003,
        "execution": "Generic crypto exchange",
        "note": "High volatility; exchange fees and spreads are material.",
    },
    "FX / macro series": {
        "volatility": 0.006,
        "drift": 0.0,
        "execution": "Generic low-cost broker",
        "note": "Lower volatility and near-zero drift; tight spreads on majors.",
    },
    "Futures-like series": {
        "volatility": 0.018,
        "drift": 0.0001,
        "execution": "Generic low-cost broker",
        "note": "Moderate volatility; real costs vary by contract and venue.",
    },
    "Custom uploaded series": {
        "volatility": 0.012,
        "drift": 0.0002,
        "execution": "Custom assumptions",
        "note": "Use Upload CSV; execution assumptions are user-defined.",
    },
}


def execution_cost(profile_name: str) -> float:
    """Effective per-trade cost (fee + spread + slippage) for a named profile.

    Raises ``KeyError`` for an unknown profile and ``ValueError`` for the
    custom profile (which has no preset cost).
    """
    profile = EXECUTION_PROFILES[profile_name]
    if profile is None:
        raise ValueError(
            "The custom profile has no preset cost; supply one explicitly."
        )
    return round(profile["fee"] + profile["spread"] + profile["slippage"], 6)


def resolve_execution(profile_name: str, custom_cost: float = 0.001) -> dict:
    """Resolve a named execution profile into concrete simulated assumptions.

    Returns a dict with ``cost`` (effective per-trade cost the backtest will
    use), the ``fee`` / ``spread`` / ``slippage`` breakdown (``None`` for
    custom), a ``turnover_warn`` threshold, and an ``is_custom`` flag.
    """
    profile = EXECUTION_PROFILES.get(profile_name)
    if profile is None:  # Custom assumptions
        return {
            "cost": float(custom_cost),
            "fee": None,
            "spread": None,
            "slippage": None,
            "turnover_warn": 0.50,
            "is_custom": True,
        }
    return {
        "cost": execution_cost(profile_name),
        "fee": profile["fee"],
        "spread": profile["spread"],
        "slippage": profile["slippage"],
        "turnover_warn": profile["turnover_warn"],
        "is_custom": False,
    }


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

    # v2: position sizing & risk (conservative defaults reproduce v1 behaviour).
    sizing_method: str = "fixed_unit"
    fixed_fraction: float = 1.0
    target_volatility: float = 0.15
    max_exposure: float = 1.0
    allow_short: bool = False
    use_risk_controls: bool = False
    vol_cap: float = 0.30
    use_drawdown_guard: bool = False
    drawdown_guard_level: float = 0.20

    # v2: structured cost components (None keeps the legacy scalar path).
    fee: float | None = None
    spread: float = 0.0
    slippage: float = 0.0

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
            sizing_method=self.sizing_method,
            fixed_fraction=self.fixed_fraction,
            target_volatility=self.target_volatility,
            max_exposure=self.max_exposure,
            allow_short=self.allow_short,
            use_risk_controls=self.use_risk_controls,
            vol_cap=self.vol_cap,
            use_drawdown_guard=self.use_drawdown_guard,
            drawdown_guard_level=self.drawdown_guard_level,
            fee=self.fee,
            spread=self.spread,
            slippage=self.slippage,
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


SAMPLE_CSV_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "samples" / "sample_prices.csv"
)


def load_sample_csv() -> pd.DataFrame:
    """Load the bundled, fully-synthetic sample CSV (offline; no network).

    The sample is generated by this project and ships with a ``benchmark_close``
    column so benchmark comparison can be demonstrated out of the box.
    """
    from .data import load_csv_series

    return load_csv_series(SAMPLE_CSV_PATH)


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

    extended_kpis = compute_extended_kpis(full_signals)
    overfit = overfit_gap(in_sample_kpis, out_of_sample_kpis)
    robustness = robustness_score(out_of_sample_kpis, walk_forward, cost_sensitivity)
    has_benchmark = "benchmark_equity" in full_signals.columns

    return {
        "config": config,
        "label": config_label(config),
        "n_obs": n,
        "train_size": split_at,
        "test_size": n - split_at,
        "full_signals": full_signals,
        "full_kpis": full_kpis,
        "extended_kpis": extended_kpis,
        "in_sample_kpis": in_sample_kpis,
        "out_of_sample_kpis": out_of_sample_kpis,
        "walk_forward": walk_forward,
        "walk_forward_message": walk_forward_message,
        "monte_carlo": monte_carlo,
        "regime_summary": regime_summary,
        "cost_sensitivity": cost_sensitivity,
        "stress_summary": stress_summary,
        "overfit": overfit,
        "robustness": robustness,
        "has_benchmark": has_benchmark,
        "data_warnings": data_quality_report(frame),
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
    """A short (≤2 sentence), analytical, honest summary of a research run."""
    oos = out_of_sample_kpis["sharpe_ratio"]
    p_loss = monte_carlo["probability_of_loss"]
    verdict = (
        "positive out-of-sample performance"
        if oos > 0
        else "weak out-of-sample evidence"
    )
    return (
        f"The '{label}' setup shows {verdict} under the current assumptions "
        f"(out-of-sample Sharpe {oos:.2f}). Robustness depends on the "
        f"~{p_loss:.0%} Monte Carlo loss probability and cost sensitivity — "
        f"these are research diagnostics, not a forecast."
    )


def analyst_warnings(result: dict, turnover_warn: float = 0.50) -> list[dict]:
    """Analyst-style, non-judgemental flags about a research run.

    Returns a list of ``{"level": ..., "text": ...}`` dicts where ``level`` is
    one of ``"warning"``, ``"info"`` or ``"success"``. The intent is to explain
    fragilities like an analyst would — never to shame the result.
    """
    notes: list[dict] = []

    oos = result["out_of_sample_kpis"]["sharpe_ratio"]
    if oos <= 0:
        notes.append({
            "level": "warning",
            "text": "Out-of-sample Sharpe is non-positive — weak evidence on "
                    "unseen data under these assumptions.",
        })
    elif oos < 0.5:
        notes.append({
            "level": "info",
            "text": "Out-of-sample Sharpe is modest; treat any edge as tentative.",
        })

    p_loss = result["monte_carlo"]["probability_of_loss"]
    if p_loss >= 0.60:
        notes.append({
            "level": "warning",
            "text": f"Monte Carlo shows a high ~{p_loss:.0%} chance of a losing "
                    "path when returns are resampled.",
        })

    cost = result["cost_sensitivity"]
    if not cost.empty:
        cheapest = float(cost.iloc[0]["total_return"])
        priciest = float(cost.iloc[-1]["total_return"])
        if cheapest > 0 and priciest <= 0:
            notes.append({
                "level": "warning",
                "text": "Result is fragile to costs: positive at zero cost but "
                        "negative at the highest cost level.",
            })

    turnover = float(result["full_kpis"].get("turnover", 0.0))
    if turnover > turnover_warn:
        notes.append({
            "level": "info",
            "text": f"Turnover ({turnover:.2f}) exceeds this profile's comfort "
                    f"threshold ({turnover_warn:.2f}); costs will bite harder.",
        })

    wf = result["walk_forward"]
    if wf is not None and not wf.empty:
        held = float((wf["test_sharpe"] > 0).mean())
        if held < 0.5:
            notes.append({
                "level": "warning",
                "text": f"Walk-forward stability is poor: only {held:.0%} of folds "
                        "kept a positive out-of-sample Sharpe.",
            })
    elif result.get("walk_forward_message"):
        notes.append({"level": "info", "text": result["walk_forward_message"]})

    overfit = result.get("overfit")
    if overfit and overfit.get("overfit_flag"):
        notes.append({
            "level": "warning",
            "text": "Possible overfitting: in-sample Sharpe is strong "
                    f"({overfit['in_sample_sharpe']:.2f}) but out-of-sample is "
                    f"weak ({overfit['out_of_sample_sharpe']:.2f}).",
        })

    for message in result.get("data_warnings", []):
        notes.append({"level": "info", "text": message})

    if not notes:
        notes.append({
            "level": "success",
            "text": "No major red flags in these diagnostics under the current "
                    "assumptions.",
        })
    return notes


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


def gross_net_frame(signals: pd.DataFrame) -> pd.DataFrame:
    """Gross vs net cumulative return (%) — the gap is the cost drag."""
    net = (signals["strategy_equity"] - 1.0) * 100
    if "gross_equity" in signals.columns:
        gross = (signals["gross_equity"] - 1.0) * 100
    else:
        gross = net
    return pd.DataFrame({"Gross": gross, "Net": net}, index=signals.index)


def exposure_frame(signals: pd.DataFrame) -> pd.DataFrame:
    """Exposure (position weight) over time, for charts."""
    return pd.DataFrame({"Exposure": signals["position"]}, index=signals.index)


def cost_breakdown(signals: pd.DataFrame) -> pd.DataFrame:
    """Total fee / spread / slippage drag over the run (as % of notional)."""
    parts = {}
    for col, label in (
        ("fee_cost", "Fee"),
        ("spread_cost", "Spread"),
        ("slippage_cost", "Slippage"),
    ):
        if col in signals.columns:
            parts[label] = float(signals[col].sum() * 100)
    if not parts:
        parts = {"Total": float(signals.get("cost", pd.Series(dtype=float)).sum() * 100)}
    return pd.DataFrame({"Cost drag (%)": parts})


def rolling_risk_frame(signals: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    """Rolling annualised volatility (%) and rolling Sharpe of net returns."""
    from .risk import rolling_sharpe, rolling_volatility

    net = signals["strategy_return"]
    return pd.DataFrame(
        {
            "Rolling vol (%)": rolling_volatility(net, 20) * 100,
            "Rolling Sharpe": rolling_sharpe(net, window),
        },
        index=signals.index,
    )


def benchmark_frame(signals: pd.DataFrame) -> pd.DataFrame | None:
    """Strategy vs benchmark cumulative return (%), or ``None`` if no benchmark."""
    if "benchmark_equity" not in signals.columns:
        return None
    return pd.DataFrame(
        {
            "Strategy": (signals["strategy_equity"] - 1.0) * 100,
            "Benchmark": (signals["benchmark_equity"] - 1.0) * 100,
        },
        index=signals.index,
    )
