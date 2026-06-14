"""End-to-end orchestration of the analytics workflow.

The pipeline ties every stage together in one reproducible call:

    synthetic data -> validation -> parameter sweep -> best config backtest
    -> KPIs -> charts (docs/assets) -> CSV exports + HTML dashboard.
"""

from __future__ import annotations

from pathlib import Path

from .backtest import BacktestConfig, run_backtest, run_parameter_sweep
from .data import generate_synthetic_series
from .reporting import build_html_report, export_csvs
from .validation import validate_price_frame
from .visualization import build_all_charts

# Repository root, derived from this file's location (src/neuroquant/pipeline.py).
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ASSETS_DIR = ROOT / "docs" / "assets"
DEFAULT_OUTPUT_DIR = ROOT / "sample_outputs"

DEFAULT_SHORT_WINDOWS = [5, 10, 20, 30]
DEFAULT_LONG_WINDOWS = [40, 60, 90, 120]


def run_pipeline(
    seed: int = 42,
    n_days: int = 750,
    cost_per_trade: float = 0.001,
    assets_dir: Path = DEFAULT_ASSETS_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    verbose: bool = True,
) -> dict:
    """Run the full analytics pipeline and write all deliverables.

    Returns a dictionary describing the best configuration, its KPIs, and the
    paths of every generated artefact.
    """
    assets_dir = Path(assets_dir)
    output_dir = Path(output_dir)

    def log(message: str) -> None:
        if verbose:
            print(message)

    log("[1/6] Generating reproducible synthetic signal series...")
    data = generate_synthetic_series(n_days=n_days, seed=seed)

    log("[2/6] Validating input data...")
    validate_price_frame(data, min_rows=max(DEFAULT_LONG_WINDOWS) + 1)

    log("[3/6] Running parameter sweep across window combinations...")
    summary = run_parameter_sweep(
        data,
        short_windows=DEFAULT_SHORT_WINDOWS,
        long_windows=DEFAULT_LONG_WINDOWS,
        cost_per_trade=cost_per_trade,
    )

    best_row = summary.iloc[0]
    best_config = BacktestConfig(
        short_window=int(best_row.short_window),
        long_window=int(best_row.long_window),
        cost_per_trade=cost_per_trade,
    )
    log(
        f"[4/6] Best config: {best_config.short_window}/"
        f"{best_config.long_window} (Sharpe {best_row.sharpe_ratio:.2f}). "
        "Running detailed backtest..."
    )
    best_result = run_backtest(data, best_config)

    log("[5/6] Building charts in docs/assets ...")
    chart_paths = build_all_charts(
        signals=best_result["signals"],
        summary=summary,
        best_kpis=best_result["kpis"],
        assets_dir=assets_dir,
    )

    log("[6/6] Exporting CSVs and HTML dashboard in sample_outputs ...")
    csv_paths = export_csvs(summary, best_result["signals"], output_dir)
    report_path = build_html_report(
        summary, best_result["kpis"], chart_paths, output_dir
    )

    log("Pipeline complete.")
    return {
        "best_config": best_config,
        "best_kpis": best_result["kpis"],
        "summary": summary,
        "chart_paths": chart_paths,
        "csv_paths": csv_paths,
        "report_path": report_path,
    }


if __name__ == "__main__":
    run_pipeline()
