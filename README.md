# NeuroQuantAI — Synthetic Quant Research & Analytics Lab

[![Tests](https://github.com/AlanBuildsAI/NeuroQuantAI-Institutional-Backtesting-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/AlanBuildsAI/NeuroQuantAI-Institutional-Backtesting-Lab/actions/workflows/tests.yml)

NeuroQuantAI is a quant-inspired Python research and analytics lab for
evaluating candidate signals using reproducible data pipelines, validation
gates, benchmark comparison, transaction-cost assumptions, in/out-of-sample
testing, walk-forward validation, Monte Carlo robustness checks, KPI
scorecards, and self-contained visual reporting.

> **Disclaimer.** This is a research and analytics **demonstration**. It runs
> on **synthetic data by default**; optional local CSV research data can be
> supplied by the user. It is **not** financial advice, **not** an investment
> recommendation, **not** a live-trading system, and makes **no performance
> guarantees**. There are **no live data feeds, no APIs, and no brokerage
> connections** anywhere in this repository.

---

## What this is

A serious, end-to-end demonstration of how credible quant research is done —
applied to a neutral *synthetic signal series* so the focus stays on
**methodology, not market calls**. The candidate signal (a moving-average
crossover) is deliberately simple; the value is the disciplined workflow around
it: validated inputs, a fair baseline, parameters chosen **in-sample** and
judged **out-of-sample**, walk-forward folds, Monte Carlo robustness, a
documented KPI scorecard, and a one-page dashboard a reviewer can read in
minutes.

"Backtesting" here is used as a **controlled research experiment** for measuring
and comparing signals under honest assumptions — not as trading advice or a
deployment-ready system.

![Executive research dashboard](docs/assets/dashboard_snapshot.png)

---

## Research workflow

```
data generation / optional CSV load
        ↓
   validation gates
        ↓
in-sample parameter sweep  →  select candidate configuration
        ↓
train / test split  →  out-of-sample evaluation
        ↓
   walk-forward validation
        ↓
  Monte Carlo robustness
        ↓
KPI scorecard  →  visuals  →  self-contained HTML report
```

Parameters are selected on the in-sample period only and judged on a held-out
out-of-sample period, so headline numbers are not the product of fitting and
scoring on the same data.

---

## What this project demonstrates

- **Quant research methodology** — hypothesis → evidence, not curve-fitting.
- **Reproducible Python pipeline** — seeded synthetic data; outputs regenerate
  byte-stable from one command.
- **Validation gates** — schema, missing/`NaN`, non-positive, and sufficiency
  checks that fail fast with clear messages.
- **Bias-aware signal evaluation** — signals are shifted by one bar to avoid
  **look-ahead bias**; parameter selection is isolated from evaluation.
- **Benchmark comparison** — every configuration is measured against a
  transparent buy-and-hold **baseline**.
- **Cost / slippage assumptions** — a simple transaction cost is charged on
  every position change.
- **Scenario analysis** — a parameter sweep across window combinations, as a
  tidy table and a Sharpe heatmap.
- **Walk-forward testing** — rolling train/test folds to probe stability on
  unseen data.
- **Monte Carlo robustness** — bootstrapped return paths describing the spread
  of outcomes and the probability of a losing path.
- **KPI reporting** — a documented risk/return scorecard (Sharpe, Sortino,
  Calmar, information ratio, drawdown, turnover, and more).
- **Dashboarding** — a self-contained, offline HTML report.
- **Testing & CI** — a fast `pytest` suite run on every push via GitHub Actions.

---

## Visual overview

| Visual | What it answers |
| --- | --- |
| ![Executive dashboard](docs/assets/dashboard_snapshot.png) | **Executive dashboard** — selected configuration, KPI scorecard, and analyst takeaway at a glance. |
| ![Scenario comparison](docs/assets/scenario_comparison.png) | **Scenario comparison** — top configurations across return, risk-adjusted score, and downside. |
| ![Baseline comparison](docs/assets/equity_curve.png) | **Baseline comparison** — does the candidate signal add information versus simply holding the series? |
| ![Parameter sweep heatmap](docs/assets/sweep_heatmap.png) | **Parameter sweep** — how the outcome changes across the experiment grid (robust, or a lucky cell?). |
| ![Walk-forward validation](docs/assets/walk_forward.png) | **Walk-forward validation** — does the in-sample choice survive out-of-sample? |
| ![Monte Carlo robustness](docs/assets/monte_carlo.png) | **Monte Carlo robustness** — how stable is the outcome when returns are resampled? |

---

## Why results may be weak or negative

The goal is **not** to cherry-pick a winning backtest. It is to test a
hypothesis honestly. On synthetic data a simple rule will frequently fail to
beat its baseline — and that is a perfectly good result. A weak or negative
finding is valuable when it is **reproducible, fairly measured, and clearly
communicated**, because in real research "the evidence does not support the
hypothesis" is often the most useful conclusion you can deliver.

---

## How to run

Using the bundled virtualenv (`.venv`) and the Makefile:

```bash
make install   # install pandas, numpy, matplotlib, pytest
make run        # run the full pipeline -> charts, CSVs, HTML dashboard
make report    # alias for run (regenerates all deliverables)
make test       # run the pytest suite
make clean      # remove generated artefacts
```

Equivalent raw commands (no Make required):

```bash
.venv/bin/pip install -r requirements.txt
PYTHONPATH=src .venv/bin/python -m neuroquant.pipeline   # full run
.venv/bin/python examples/minimal_backtest.py            # tiny demo, no files written
.venv/bin/python -m pytest tests/ -q                     # tests
```

Optional editable install (then drop the `PYTHONPATH=src` prefix):

```bash
.venv/bin/pip install -e .
```

Using your own research data (optional, offline only):

```python
from neuroquant.data import load_csv_series
frame = load_csv_series("my_series.csv")  # needs date/timestamp + close columns
```

---

## Outputs

Charts in `docs/assets/` (PNG + SVG):

| File | What it is |
| --- | --- |
| `dashboard_snapshot` | Executive KPI-card snapshot |
| `equity_curve` | Strategy vs baseline cumulative return |
| `drawdown` | Drawdown (risk) profile |
| `scenario_comparison` | Top configs across return / Sharpe / drawdown |
| `sweep_heatmap` | Sharpe across every window combination |
| `return_distribution` | Daily return histogram |
| `walk_forward` | In-sample vs out-of-sample Sharpe by fold |
| `monte_carlo` | Bootstrapped total-return distribution |

Reports in `sample_outputs/`:

| File | What it is |
| --- | --- |
| `dashboard.html` | Self-contained one-page report (opens offline) |
| `parameter_sweep_summary.csv` | Tidy scenario comparison table |
| `equity_curve_sample.csv` | Per-day equity curve of the selected config |
| `walk_forward_summary.csv` | Per-fold walk-forward results |

---

## Limitations

- Synthetic data by default; it has no real-world structure and results do not
  generalise to any market.
- A single, deliberately simple signal family (one moving-average crossover).
- No live trading, no order routing, and no execution modelling beyond a
  simplified flat cost / slippage assumption.
- Not investment advice and not production trading infrastructure.

---

## Future improvements

- Additional signal families and feature engineering.
- More realistic transaction-cost and slippage modelling.
- Portfolio-level (multi-series) testing.
- First-class local CSV research datasets and loaders.
- Richer report export options.

---

## Why this matters across analytics roles

The same workflow — validate inputs, design a fair experiment, separate
selection from evaluation, test robustness, measure risk as well as return, and
communicate the evidence — transfers directly to **data analytics**, **business
analytics**, **operations analytics**, **product analytics**, **healthcare
operations analytics**, and **quant analytics**. The differentiator is not a
clever model; it is a trustworthy, reproducible, well-communicated process.

Further reading: [`docs/methodology.md`](docs/methodology.md),
[`docs/analytics_explanation.md`](docs/analytics_explanation.md),
[`docs/portfolio_relevance.md`](docs/portfolio_relevance.md).
