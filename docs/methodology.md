# Methodology

## Research question

> Given a controlled synthetic signal series, can a simple, transparent rule
> (a moving-average crossover) be shown to add value over a buy-and-hold
> **baseline** — and, just as importantly, can the whole investigation be made
> **reproducible, validated, and decision-ready**?

The emphasis is deliberately on the *workflow*, not on "winning". The output is
evidence and a clear interpretation, whatever the sign of the result.

## Synthetic data generation

Data is generated, never downloaded. We use a **geometric random walk**:

```
value_t = value_{t-1} * exp(drift + volatility * z_t)
```

with `z_t` drawn from a **seeded** NumPy generator. A fixed seed means the
series — and therefore every chart, CSV, and KPI — is byte-stable across runs.
We call it a *synthetic signal series* (not an asset price) to keep the project
firmly in analytics territory and away from any market-prediction framing.

## Validation checks

Before any analysis runs, inputs must pass `validate_price_frame`:

- input is a non-empty `DataFrame`,
- the required `close` column is present,
- no missing / `NaN` values,
- all values strictly positive (a level series cannot be ≤ 0),
- enough rows to support the requested rolling windows.

Configuration is guarded by `validate_window_config`: windows must be positive
integers and the short window must be **strictly less** than the long window,
otherwise a crossover is meaningless. Every failure raises a `ValidationError`
with an actionable message.

## Experiment design

For each configuration we:

1. compute fast and slow simple moving averages of `close`,
2. set a raw signal: in-market when fast > slow, else flat,
3. **shift the signal by one bar** so a decision uses only prior information —
   this removes **look-ahead bias**, the most common silent error in this kind
   of analysis,
4. apply a simple transaction cost + slippage on every position change,
5. compute strategy returns and a cumulative equity curve.

## Baseline comparison

Every strategy is measured against a **buy-and-hold baseline** over the same
window. Without a benchmark, a return number is uninterpretable; with one, we
can state plainly whether the rule added value.

## Scenario analysis (parameter sweep)

We evaluate a grid of (short, long) window pairs and collect the KPIs into a
tidy table sorted by Sharpe. This shows whether any apparent edge is **robust
across settings** or just a lucky single cell — visualised as a Sharpe heatmap.

## Metrics selected and why

| Metric | Why it is included |
| --- | --- |
| `total_return` / `baseline_return` | Headline outcome vs the benchmark |
| `annualized_volatility` | Standardised risk scale |
| `sharpe_ratio` | Reward per unit of total risk |
| `sortino_ratio` | Reward per unit of *downside* risk |
| `max_drawdown` | Worst-case peak-to-trough decline |
| `active_days` | How often the strategy is exposed |
| `trade_count` | Activity / turnover (cost driver) |
| `correlation_to_baseline` | How differentiated the strategy really is |
| `win_loss_ratio` | Shape of the return distribution |

## Limitations

- Synthetic data has no real-world structure; results do not generalise to any
  market and are not meant to.
- A single random seed is one realisation; broad conclusions would need many
  seeds (a natural next step).
- The cost model is intentionally simple (a flat per-change fraction).
- The strategy family is deliberately minimal to keep focus on the workflow.

## Why it avoids live data and prediction claims

Using only seeded synthetic data keeps the project **reproducible**, removes any
dependence on third-party APIs or keys, and makes clear that the goal is a
demonstration of analytics craft — not forecasting markets or giving advice.

## What a hiring manager should evaluate

- Are inputs validated before they are trusted?
- Is the experiment fair (benchmark, no look-ahead)?
- Are the metrics sensible, documented, and honestly reported?
- Is everything reproducible from a seed?
- Is the result communicated clearly to a non-technical audience?

## Methodology references / concepts

This project applies widely taught analytics best practices rather than any
proprietary method:

- **Input validation / data-quality gates** before analysis.
- **Reproducibility** via fixed random seeds and committed outputs.
- **Benchmark / baseline comparison** so results are interpretable.
- **Avoiding look-ahead bias** by lagging signals relative to information.
- **Scenario / sensitivity analysis** across a parameter grid.
- **Clear visual and written reporting** for decision-makers.

(No external sources were browsed and no citations are fabricated; the above are
standard, well-established analytics concepts.)
