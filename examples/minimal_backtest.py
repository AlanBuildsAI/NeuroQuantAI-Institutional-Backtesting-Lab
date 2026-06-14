"""Minimal end-to-end example.

Run from the repository root:

    .venv/bin/python examples/minimal_backtest.py

It generates synthetic data, validates it, runs a single backtest and a
small parameter sweep, then prints a compact KPI summary. No files are
written -- use ``python -m neuroquant.pipeline`` for the full deliverables.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running directly without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neuroquant import (  # noqa: E402
    BacktestConfig,
    generate_synthetic_series,
    run_backtest,
    run_parameter_sweep,
    validate_price_frame,
)


def main() -> None:
    data = generate_synthetic_series(n_days=500, seed=42)
    validate_price_frame(data)

    result = run_backtest(data, BacktestConfig(short_window=20, long_window=60))
    print("Single backtest KPIs (20/60 windows):")
    for key, value in result["kpis"].items():
        print(f"  {key:24s}: {value:.4f}")

    print("\nTop 3 configurations from a small sweep:")
    summary = run_parameter_sweep(
        data, short_windows=[10, 20, 30], long_windows=[60, 90]
    )
    print(
        summary[["short_window", "long_window", "sharpe_ratio", "total_return"]]
        .head(3)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
