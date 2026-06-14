# NeuroQuant — Python Analytics Case Study

A small, polished, end-to-end **Python data-analytics case study**. It takes a
reproducible synthetic time series and walks it through a complete analyst
workflow: **data validation → experiment design → KPI metric design → scenario
comparison → visual storytelling → decision-ready reporting.**

> **Disclaimer.** This is an **analytics demonstration built on synthetic data
> only**. It is **not** a trading system, **not** financial advice, and makes
> **no** market-prediction or performance claims. There are no API keys, no live
> data, and no brokerage connections anywhere in this repo.

---

## What it is

The project frames a classic "moving-average crossover" rule as an **analytics
experiment** over a neutral *synthetic signal series*. The interesting part is
not the rule — it is the **repeatable, auditable workflow** around it: validated
inputs, a transparent baseline, a parameter sweep across scenarios, a documented
KPI scorecard, and a self-contained one-page dashboard a non-technical reviewer
can read in under two minutes.

![Executive dashboard](docs/assets/dashboard_snapshot.png)

![Strategy vs baseline](docs/assets/equity_curve.png)

![Parameter sweep heatmap](docs/assets/sweep_heatmap.png)

---

## Analytics skills demonstrated

- **Data validation / data quality** — schema, missing values, non-positive
  values, sufficiency checks, and configuration guards that fail fast with clear
  messages (`src/neuroquant/validation.py`).
- **Reproducibility** — every result is seeded; the committed outputs regenerate
  byte-stable (`src/neuroquant/data.py`).
- **Experiment design & benchmark comparison** — each scenario is compared to a
  transparent buy-and-hold **baseline**; signals are shifted by one step to
  avoid **look-ahead bias** (`src/neuroquant/backtest.py`).
- **KPI / metric design** — a documented scorecard (return, volatility, Sharpe,
  Sortino, drawdown, correlation, win/loss, etc.) (`src/neuroquant/metrics.py`).
- **Scenario analysis** — a parameter sweep across window combinations returns a
  tidy comparison table.
- **Visual storytelling** — senior-level matplotlib charts, each stating its
  analytical question and takeaway (`src/neuroquant/visualization.py`).
- **Decision-ready reporting** — clean CSV exports plus a self-contained offline
  HTML dashboard (`src/neuroquant/reporting.py`).
- **Documentation** — methodology, plain-English walkthrough, portfolio notes.

---

## How to run

Using the bundled virtualenv (`.venv`) and the Makefile:

```bash
make install   # install pandas, numpy, matplotlib, pytest
make run       # run the full pipeline -> charts, CSVs, HTML dashboard
make report    # alias for run (regenerates all deliverables)
make test      # run the pytest suite
make clean     # remove generated artefacts
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

---

## Outputs created

| Path | What it is |
| --- | --- |
| `docs/assets/dashboard_snapshot.{png,svg}` | Executive KPI-card snapshot |
| `docs/assets/equity_curve.{png,svg}` | Strategy vs baseline cumulative return |
| `docs/assets/drawdown.{png,svg}` | Drawdown (risk) profile |
| `docs/assets/scenario_comparison.{png,svg}` | Top configs across return / Sharpe / drawdown |
| `docs/assets/sweep_heatmap.{png,svg}` | Sharpe across every window combination |
| `docs/assets/return_distribution.{png,svg}` | Daily return histogram |
| `sample_outputs/parameter_sweep_summary.csv` | Tidy scenario comparison table |
| `sample_outputs/equity_curve_sample.csv` | Per-day equity curve of the best config |
| `sample_outputs/dashboard.html` | Self-contained one-page report (opens offline) |

---

## What the dashboard means

The dashboard answers an analyst's questions, not a trader's:

- **Best config / Top Sharpe** — which scenario looked strongest on a
  risk-adjusted basis in this controlled experiment.
- **Strategy vs baseline return** — did the rule add anything over the simple
  benchmark? (Often it does not — that is fine; see below.)
- **Max drawdown** — the worst peak-to-trough decline, a plain risk read.
- **Trades / active days** — how much the strategy actually acted.
- **Analyst takeaway** — a one-line, decision-ready interpretation.

---

## Why weak or negative results are acceptable here

This is an **analytics** portfolio piece. The deliverable is a **trustworthy,
reproducible workflow** — clean data, a fair benchmark, honest metrics, and
clear reporting — **not** a profitable model. A negative Sharpe on synthetic
data is a perfectly good result: it shows the pipeline measures and reports
reality faithfully instead of cherry-picking a flattering number. In real
business analytics, "the test was inconclusive, and here is the rigorous
evidence" is frequently the most valuable answer you can deliver.

---

## Why it is relevant to analytics roles

The exact same workflow transfers directly to **Data Analyst**, **Business
Analyst**, **Operations Analyst**, and **Product / Analytics** roles:

- validating messy inputs before trusting them,
- designing experiments with a fair benchmark,
- defining and documenting KPIs,
- comparing scenarios in a tidy table,
- turning numbers into clear visuals and a one-page decision summary,
- and documenting the whole thing so others can reproduce it.

Further reading: [`docs/methodology.md`](docs/methodology.md),
[`docs/analytics_explanation.md`](docs/analytics_explanation.md),
[`docs/portfolio_relevance.md`](docs/portfolio_relevance.md).
