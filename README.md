# NeuroQuantAI Institutional Backtesting Lab

A portfolio-oriented quantitative analytics project focused on building a disciplined, reproducible backtesting workflow for strategy research.

> Synthetic data only. Educational / portfolio project. Not financial advice, not trading signals, and not intended for live deployment.

## Why this project exists

Many simple trading backtests are misleading because they ignore data quality, costs, slippage, drawdowns, overfitting, and benchmark comparison. This repository demonstrates an analyst-style research workflow: generate controlled synthetic data, validate inputs, test strategy logic, calculate risk-adjusted metrics, and communicate results clearly.

The goal is not to prove a profitable strategy. The goal is to show structured analytical thinking and decision-ready reporting.

## Current working example

Run the reproducible synthetic example:

```bash
pip install -r requirements.txt
python examples/minimal_backtest.py
```

The example currently:

- generates reproducible synthetic daily price data
- validates required fields and basic data quality
- runs moving-average crossover strategy variants
- applies transaction cost and slippage assumptions
- compares strategy performance against a buy-and-hold benchmark
- calculates total return, annualized volatility, Sharpe ratio, Sortino ratio, max drawdown, trade count, active days, and market correlation
- prints a structured parameter-sweep summary table

## Example output

```text
Synthetic backtest parameter sweep
 short_window  long_window  total_return_pct  benchmark_return_pct  annualized_volatility_pct  sharpe_ratio  sortino_ratio  max_drawdown_pct  active_days  trade_count  market_correlation
           20           60            -13.31                  3.55                      14.38         -0.42          -0.44            -20.42          235           11                0.67
            5           20            -15.64                  3.55                      14.71         -0.50          -0.59            -24.00          244           32                0.69
           10           30            -18.65                  3.55                      14.30         -0.65          -0.70            -24.67          233           21                0.67
```

Negative results are intentionally acceptable in this portfolio project: the point is to show measurement discipline, not to cherry-pick a profitable synthetic outcome.

## What this demonstrates for analytics roles

- Python analytical workflows with pandas and NumPy
- data validation before analysis
- metric design and KPI-style summary tables
- parameter comparison / experiment tracking
- benchmark comparison
- risk and performance analytics
- time-series reasoning
- clean documentation and reproducible scripts
- translating raw outputs into decision-ready summaries

## Repository structure

```text
examples/
  minimal_backtest.py          # reproducible synthetic backtest + parameter sweep
docs/
  methodology.md               # workflow explanation and limitations
sample_outputs/
  parameter_sweep_summary.csv  # sample expected output table
requirements.txt
README.md
```

## Methodology summary

The current workflow follows a simple research sequence:

1. Generate synthetic close-price data.
2. Validate required fields and basic data quality.
3. Create moving-average crossover signals.
4. Shift positions by one period to avoid look-ahead bias.
5. Apply transaction cost and slippage assumptions.
6. Calculate equity curve, benchmark curve, drawdown, and risk metrics.
7. Compare multiple parameter configurations in a structured table.

## Limitations

This repository intentionally avoids live data, brokerage APIs, execution systems, and strategy claims. It uses synthetic data only and should be evaluated as a portfolio example of analytical workflow design rather than investment performance.

## Planned improvements

- Add visual charts for equity curve and drawdown
- Add CSV export from the script
- Add walk-forward validation example
- Add Monte Carlo robustness checks
- Add a small dashboard-ready output file
