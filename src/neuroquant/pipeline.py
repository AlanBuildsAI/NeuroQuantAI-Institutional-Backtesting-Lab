"""End-to-end orchestration of the quant research workflow.

The pipeline ties every stage together in one reproducible call:

    data -> validation -> feature engineering -> in-sample selection across
    signal families -> train/test split -> out-of-sample evaluation ->
    walk-forward validation -> Monte Carlo robustness -> regime analysis ->
    cost sensitivity & stress tests -> charts -> CSV exports + HTML dashboard.

Parameters are selected on training data only and judged on a held-out
out-of-sample period, so the headline numbers are not the product of fitting
and scoring on the same data.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .backtest import (
    build_candidate_configs,
    config_from_row,
    run_backtest,
    run_config_sweep,
    run_parameter_sweep,
)
from .data import generate_synthetic_series
from .features import build_feature_frame
from .metrics import compute_kpis
from .regime import summarize_by_regime
from .reporting import build_html_report, export_csvs
from .research import (
    monte_carlo_bootstrap,
    split_train_test,
    walk_forward_validation,
)
from .stress import cost_sensitivity_analysis, stress_test_summary
from .validation import validate_price_frame
from .visualization import build_all_charts

# Repository root, derived from this file's location (src/neuroquant/pipeline.py).
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ASSETS_DIR = ROOT / "docs" / "assets"
DEFAULT_OUTPUT_DIR = ROOT / "sample_outputs"

DEFAULT_SHORT_WINDOWS = [5, 10, 20, 30]
DEFAULT_LONG_WINDOWS = [40, 60, 90, 120]

TOTAL_STAGES = 11


def _oos_kpis(full_signals: pd.DataFrame, split_at: int) -> dict:
    """KPIs for the out-of-sample slice, with equity rebased to the slice."""
    oos = full_signals.iloc[split_at:].copy()
    oos["strategy_equity"] = (1.0 + oos["strategy_return"]).cumprod()
    oos["baseline_equity"] = (1.0 + oos["baseline_return"]).cumprod()
    return compute_kpis(oos)


def run_pipeline(
    seed: int = 42,
    n_days: int = 750,
    cost_per_trade: float = 0.001,
    train_fraction: float = 0.7,
    assets_dir: Path = DEFAULT_ASSETS_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    verbose: bool = True,
) -> dict:
    """Run the full research pipeline and write all deliverables.

    Returns a dictionary describing the selected configuration, its in-sample
    and out-of-sample KPIs, and every robustness/diagnostic result and artefact
    path produced along the way.
    """
    assets_dir = Path(assets_dir)
    output_dir = Path(output_dir)

    def log(message: str) -> None:
        if verbose:
            print(message)

    log(f"[1/{TOTAL_STAGES}] Generating reproducible synthetic signal series...")
    data = generate_synthetic_series(n_days=n_days, seed=seed)

    log(f"[2/{TOTAL_STAGES}] Validating input data...")
    validate_price_frame(data, min_rows=max(DEFAULT_LONG_WINDOWS) + 1)

    log(f"[3/{TOTAL_STAGES}] Engineering features...")
    feature_frame = build_feature_frame(data)

    log(f"[4/{TOTAL_STAGES}] Splitting into in-sample / out-of-sample periods...")
    train, _test = split_train_test(data, train_fraction=train_fraction)
    split_at = len(train)

    log(
        f"[5/{TOTAL_STAGES}] Selecting a candidate signal across families "
        "(in-sample)..."
    )
    candidates = build_candidate_configs(cost_per_trade=cost_per_trade)
    train_summary = run_config_sweep(train, candidates)
    best_config = config_from_row(train_summary.iloc[0])

    best_result = run_backtest(data, best_config)
    full_signals = best_result["signals"]
    in_sample_kpis = run_backtest(train, best_config)["kpis"]
    out_of_sample_kpis = _oos_kpis(full_signals, split_at)
    log(
        f"        Selected '{train_summary.iloc[0].label}' "
        f"(in-sample Sharpe {in_sample_kpis['sharpe_ratio']:.2f}; "
        f"out-of-sample Sharpe {out_of_sample_kpis['sharpe_ratio']:.2f})."
    )

    # Multi-family scenario landscape (full series) + trend heatmap grid.
    scenario_summary = run_config_sweep(data, candidates)
    trend_sweep = run_parameter_sweep(
        data,
        short_windows=DEFAULT_SHORT_WINDOWS,
        long_windows=DEFAULT_LONG_WINDOWS,
        cost_per_trade=cost_per_trade,
    )

    log(f"[6/{TOTAL_STAGES}] Running walk-forward validation across families...")
    walk_forward = walk_forward_validation(
        data, configs=candidates, cost_per_trade=cost_per_trade
    )

    log(f"[7/{TOTAL_STAGES}] Running Monte Carlo robustness analysis...")
    monte_carlo = monte_carlo_bootstrap(full_signals["strategy_return"])

    log(f"[8/{TOTAL_STAGES}] Attributing performance by volatility regime...")
    regime_summary = summarize_by_regime(full_signals)

    log(f"[9/{TOTAL_STAGES}] Running cost sensitivity and stress tests...")
    cost_sensitivity = cost_sensitivity_analysis(data, best_config)
    stress_summary = stress_test_summary(data, best_config)

    log(f"[10/{TOTAL_STAGES}] Building charts in docs/assets ...")
    chart_paths = build_all_charts(
        signals=full_signals,
        summary=scenario_summary,
        best_kpis=best_result["kpis"],
        assets_dir=assets_dir,
        best_config=best_config,
        trend_sweep=trend_sweep,
        scenario_summary=scenario_summary,
        walk_forward=walk_forward,
        monte_carlo=monte_carlo,
        regime_summary=regime_summary,
        cost_sensitivity=cost_sensitivity,
    )

    log(
        f"[11/{TOTAL_STAGES}] Exporting CSVs and HTML dashboard in "
        "sample_outputs ..."
    )
    csv_paths = export_csvs(
        scenario_summary=scenario_summary,
        signals=full_signals,
        walk_forward=walk_forward,
        regime_summary=regime_summary,
        cost_sensitivity=cost_sensitivity,
        stress_summary=stress_summary,
        feature_frame=feature_frame,
        output_dir=output_dir,
    )
    report_path = build_html_report(
        scenario_summary=scenario_summary,
        best_kpis=best_result["kpis"],
        chart_paths=chart_paths,
        output_dir=output_dir,
        in_sample_kpis=in_sample_kpis,
        out_of_sample_kpis=out_of_sample_kpis,
        walk_forward=walk_forward,
        monte_carlo=monte_carlo,
        regime_summary=regime_summary,
        cost_sensitivity=cost_sensitivity,
        stress_summary=stress_summary,
        best_config=best_config,
    )

    log("Pipeline complete.")
    return {
        "best_config": best_config,
        "best_kpis": best_result["kpis"],
        "in_sample_kpis": in_sample_kpis,
        "out_of_sample_kpis": out_of_sample_kpis,
        "scenario_summary": scenario_summary,
        "trend_sweep": trend_sweep,
        "walk_forward": walk_forward,
        "monte_carlo": monte_carlo,
        "regime_summary": regime_summary,
        "cost_sensitivity": cost_sensitivity,
        "stress_summary": stress_summary,
        "feature_frame": feature_frame,
        "chart_paths": chart_paths,
        "csv_paths": csv_paths,
        "report_path": report_path,
    }


if __name__ == "__main__":
    run_pipeline()
