# Methodology

This repository is a synthetic-data portfolio example designed to show a disciplined analytics workflow.

## Research question

Can a simple time-series strategy be evaluated in a reproducible way while accounting for input validation, benchmark comparison, implementation costs, drawdown, and risk-adjusted metrics?

## Workflow

1. Synthetic data generation
   - Creates a reproducible close-price series using a fixed random seed.
   - Uses synthetic data only so the project stays portable and dependency-light.

2. Data validation
   - Confirms required columns exist.
   - Checks for missing close prices.
   - Checks that close prices are positive.

3. Signal generation
   - Calculates short and long moving averages.
   - Creates a long-only signal when the short moving average is above the long moving average.

4. Bias control
   - Shifts the signal by one period before calculating returns to avoid using same-day information in the simulated position.

5. Implementation cost assumptions
   - Applies transaction cost and slippage assumptions in basis points when the position changes.

6. Performance measurement
   - Calculates strategy equity curve and benchmark equity curve.
   - Calculates total return, benchmark return, annualized volatility, Sharpe ratio, Sortino ratio, max drawdown, active days, trade count, and market correlation.

7. Parameter comparison
   - Runs a small parameter sweep across multiple moving-average configurations.
   - Sorts results by Sharpe ratio and total return to create a decision-ready summary table.

## Why negative results are acceptable

The current synthetic output does not try to show a winning strategy. That is intentional. A strong analytics project should measure performance honestly instead of cherry-picking a flattering scenario. The value of this repository is the research structure: data checks, assumptions, reproducibility, cost modeling, and clear reporting.

## Limitations

- Synthetic data only
- No live data integration
- No execution system
- No production risk engine
- No recommendation or prediction claim

## Portfolio relevance

This project demonstrates skills relevant to data analyst, business analyst, operations analyst, and analytics-adjacent roles:

- Python analytics
- pandas and NumPy
- data validation
- structured metrics
- KPI reporting
- experiment comparison
- risk and performance analytics
- clear documentation
