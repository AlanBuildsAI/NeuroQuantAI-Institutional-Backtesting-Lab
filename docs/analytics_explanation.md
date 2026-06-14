# Research Explanation (Plain English)

This is the same story the code tells, without the jargon. It follows the data
from raw input all the way to a decision-ready research dashboard.

## 1. Start with controlled, reproducible data

Instead of pulling unpredictable data from the internet, we **generate** a
synthetic series with a fixed random seed. Think of it as a lab sample: the same
recipe always produces the same sample, so anyone can rerun the project and get
identical results. The output is a simple table with a date and a `close` value.

## 2. Validate before trusting

Real analysts never trust raw data blindly. Before anything else we check:

- Is the `close` column actually there?
- Are there blanks or missing values?
- Are any values zero or negative (which would be impossible for a level)?
- Do we have enough rows to compute the rolling averages we need?

If any check fails, the program **stops immediately with a clear message**
instead of quietly producing wrong numbers later.

## 3. Turn data into experiments

We test a simple rule: compare a fast moving average to a slow one. When the
fast line is above the slow line, the rule is "in"; otherwise it sits out. One
subtle but important detail: we make each decision using **yesterday's**
information, never today's. This prevents the rule from "cheating" by peeking at
the future — a classic mistake called look-ahead bias.

We also charge a small cost every time the rule switches in or out, so the
numbers reflect that acting is not free.

## 4. Compare against a fair benchmark

A return number on its own means little. So every experiment is compared to a
plain **buy-and-hold baseline** — what you would have gotten by doing nothing
clever. This is the analyst's equivalent of an A/B test control group.

## 5. Measure with documented KPIs

We summarise each experiment with a **scorecard**: total return vs the baseline,
how bumpy the ride was (volatility, drawdown), reward-per-risk (Sharpe,
Sortino), how active the rule was, and how different it really was from the
baseline. Every KPI is defined in plain terms in the code and methodology.

## 6. Run many scenarios

Rather than trusting one setting, we sweep across many combinations of fast and
slow windows and put the results in a tidy table. A heatmap then shows at a
glance whether good results are **consistent** across settings or just a lucky
one-off cell.

## 7. Don't judge a setting on the data that picked it

Picking the "best" setting and then grading it on the same data is how analysts
fool themselves. So we split the timeline into an earlier **training** stretch
and a later **test** stretch the selection never sees, choose the setting on
training, and report how it does on the unseen test. We then repeat this
**walk-forward** — train, test, step forward — to see whether the choice holds
up fold after fold.

## 8. Stress-test the result

Finally we ask "how lucky was the ordering of returns?" by reshuffling the
realised returns thousands of times (a **Monte Carlo bootstrap**) to see the
range of outcomes and how often a path ends in a loss. This describes how
*stable* the result is — it does not predict the future.

## 9. Visualise and report

Finally we turn the numbers into a small set of clean charts and a one-page HTML
dashboard with KPI cards, the best configuration, and a one-line analyst
takeaway. The dashboard opens offline in any browser — no internet, no setup —
so a reviewer can understand the whole study in a couple of minutes.

## 10. Interpret honestly

If the candidate signal beats the baseline, we say so. If it does not — which on
synthetic data it often will not — we say that too. The value of the project is
the **trustworthy process**: controlled data, validation, a fair benchmark,
out-of-sample and robustness testing, honest metrics, and clear communication.
