# Methodology — NeuroQuantAI Synthetic Quant Research & Analytics Lab

## Research question

> Given a controlled synthetic signal series, can a simple, transparent
> candidate signal (a moving-average crossover) be shown to add value over a
> buy-and-hold **baseline** — and, just as importantly, can the whole
> investigation be made **reproducible, validated, bias-aware, robustness-tested,
> and decision-ready**?

The emphasis is deliberately on the *research workflow*, not on "winning". The
output is evidence and a clear interpretation, whatever the sign of the result.
"Backtesting" here means a **controlled research experiment**, not trading
advice or a deployment-ready system.

## Synthetic data generation

Data is generated, never downloaded. We use a **geometric random walk**:

```
value_t = value_{t-1} * exp(drift + volatility * z_t)
```

with `z_t` drawn from a **seeded** NumPy generator. A fixed seed means the
series — and therefore every chart, CSV, and KPI — is byte-stable across runs.
We call it a *synthetic signal series* (not an asset price) to keep the project
firmly in research territory and away from any market-prediction framing.

Synthetic data is the **default**. An optional CSV loader
(`neuroquant.data.load_csv_series`) lets an analyst point the same pipeline at
their own local historical data (a timestamp column plus `close`). It reads a
local file only — no network, no API — and applies the same data-quality gates
(sorted, de-duplicated timestamps; no missing or non-positive `close`).

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

## Feature engineering

`features.py` builds small, **causal** features (no forward shifts): returns,
moving averages and price-to-MA distance, multi-horizon momentum, rolling
z-scores, rolling volatility, drawdown-from-rolling-high, and volatility regime
labels. Because no feature looks ahead, building the feature frame introduces no
leakage; positions derived from features are additionally lagged one bar.

## Signal families

The lab compares four long-or-flat candidate families (`signals.py`):

- **trend** — fast/slow moving-average crossover,
- **momentum** — positive trailing momentum,
- **mean reversion** — long when the z-score is oversold,
- **composite** — a transparent average of the three signals above, thresholded
  (a literal composite of simple signals, not a machine-learning model).

An optional **volatility filter** damps exposure when rolling volatility exceeds
a *trailing* (expanding-quantile) threshold, so the filter itself never uses
future data.

## Experiment design

For each configuration we:

1. build the family's target position from past-only features,
2. **shift the position by one bar** so a decision uses only prior information —
   this removes **look-ahead bias**, the most common silent error in this kind
   of analysis,
3. apply a simple transaction cost + slippage on every position change,
4. compute strategy returns and a cumulative equity curve.

Candidate selection runs across all families on the in-sample period; the
winner is then judged out-of-sample and in walk-forward folds.

## Baseline comparison

Every strategy is measured against a **buy-and-hold baseline** over the same
window. Without a benchmark, a return number is uninterpretable; with one, we
can state plainly whether the rule added value.

## Scenario analysis (parameter sweep)

We evaluate a grid of (short, long) window pairs and collect the KPIs into a
tidy table sorted by Sharpe. This shows whether any apparent edge is **robust
across settings** or just a lucky single cell — visualised as a Sharpe heatmap.

## In-sample / out-of-sample split

Choosing parameters and judging them on the same data is the most common way to
fool yourself. We split the series chronologically into an **in-sample**
(training) period and a later, non-overlapping **out-of-sample** (test) period.
The configuration is selected on the in-sample sweep and then evaluated on the
out-of-sample period it never saw. Both reads are reported side by side.

## Walk-forward validation

A single split is one trial. Walk-forward validation repeats the
select-then-evaluate cycle over rolling windows: pick the best configuration on
a training window, evaluate it on the *next* window, then step forward. The
result is a per-fold table comparing in-sample and out-of-sample Sharpe — a
direct read on whether an in-sample choice tends to survive on unseen data.

## Monte Carlo robustness

To gauge how fragile an outcome is, we bootstrap the realised per-bar return
stream (resampling with replacement) into many alternative equity paths and
summarise the spread: median, 5th and 95th percentile total return, the
**probability of a losing path**, and the drawdown distribution. This is a
descriptive robustness diagnostic, not a forecast.

## Regime, cost, and stress diagnostics

Beyond a single headline number we attribute and stress the selected result:

- **Regime attribution** (`regime.py`) breaks performance down by volatility
  regime (low / normal / high / stress) so we can see whether the result came
  from calm or turbulent periods.
- **Cost sensitivity** (`stress.py`) re-runs the selected rule across a ladder
  of transaction-cost assumptions; by construction higher costs never improve a
  net result.
- **Stress diagnostics** (`stress.py`) apply clearly-labelled synthetic
  transforms — higher costs, a volatility shock, an adverse drift, and amplified
  downside — to probe fragility. These are diagnostics, not predictions.

## Metrics selected and why

| Metric | Why it is included |
| --- | --- |
| `total_return` / `baseline_return` | Headline outcome vs the benchmark |
| `excess_return` | Strategy return net of the baseline |
| `annualized_return` | Compounded growth on a one-year scale |
| `annualized_volatility` | Standardised risk scale |
| `sharpe_ratio` | Reward per unit of total risk |
| `sortino_ratio` | Reward per unit of *downside* risk |
| `calmar_ratio` | Return per unit of worst-case drawdown |
| `information_ratio` | Active return per unit of tracking error vs baseline |
| `max_drawdown` | Worst-case peak-to-trough decline |
| `active_days` | How often the signal is exposed |
| `trade_count` / `turnover` | Activity (cost drivers) |
| `correlation_to_baseline` | How differentiated the signal really is |
| `win_loss_ratio` | Shape of the return distribution |

## Limitations

- Synthetic data has no real-world structure; results do not generalise to any
  market and are not meant to.
- A single random seed is one realisation; the walk-forward and Monte Carlo
  steps probe stability, but broad conclusions would still need many seeds.
- The cost / slippage model is intentionally simple (a flat per-change fraction).
- The signal family is deliberately minimal to keep focus on the workflow.

## Why it avoids live data and prediction claims

Using only seeded synthetic data keeps the project **reproducible**, removes any
dependence on third-party APIs or keys, and makes clear that the goal is a
demonstration of analytics craft — not forecasting markets or giving advice.

## What a reviewer should evaluate

- Are inputs validated before they are trusted?
- Is the experiment fair (benchmark, no look-ahead, selection separated from
  evaluation)?
- Are the metrics sensible, documented, and honestly reported?
- Is the result tested out-of-sample and for robustness?
- Is everything reproducible from a seed?
- Is the conclusion communicated clearly to a non-technical audience?

## Methodology references / concepts

This project applies widely taught analytics best practices rather than any
proprietary method:

- **Input validation / data-quality gates** before analysis.
- **Reproducibility** via fixed random seeds and committed outputs.
- **Benchmark / baseline comparison** so results are interpretable.
- **Avoiding look-ahead bias** by lagging signals relative to information.
- **Out-of-sample evaluation and walk-forward validation** to separate
  parameter selection from judgement.
- **Bootstrap / Monte Carlo robustness analysis** to gauge outcome stability.
- **Scenario / sensitivity analysis** across a parameter grid.
- **Clear visual and written reporting** for decision-makers.

(No external sources were browsed and no citations are fabricated; the above are
standard, well-established analytics concepts.)
