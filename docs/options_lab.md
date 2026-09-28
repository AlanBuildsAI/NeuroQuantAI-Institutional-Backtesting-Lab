# NeuroQuant Options Lab

NeuroQuant Options Lab extends the existing equity/ETF research engine with an
offline-first framework for **observed option-chain research** and **paper-trade
reconciliation**.

The design goal is not to make simulated options look perfect. It is to make
the assumptions visible, conservative, reproducible, and easy to stress.

## What v1 supports

- Canonical calls/puts and bull-call spreads.
- Black-Scholes-Merton pricing, Greeks, and implied-volatility inversion for
  sanity checks and scenario work.
- Historical chain CSV normalization using actual contract identity:
  timestamp, expiration, strike, type, underlying price, bid and ask.
- Optional bid/ask sizes, volume, open interest, IV and Greeks.
- NBBO-aware fill modes:
  - midpoint: optimistic benchmark only;
  - realistic: configurable adverse movement from midpoint;
  - conservative: buy at ask / sell at bid.
- Per-contract commissions.
- Liquidity gates for bid/ask width, zero bids, volume and open interest.
- Realized-volatility features and IV/RV filters.
- Delta/DTE/width-based bull-call-spread selection.
- Event-driven historical bull-call-spread backtests.
- Profit target, stop loss, exit-DTE and max-hold exits.
- Shadow paper ledger for comparing broker paper fills with NeuroQuant's
  realistic and conservative fills.

## Data contract

Historical option backtests require observed chain snapshots. NeuroQuant does
not fabricate missing contracts or historical quotes.

Minimum CSV columns:

```text
timestamp
symbol
expiration
strike
option_type
underlying_price
bid
ask
```

Optional fields:

```text
bid_size ask_size volume open_interest
iv delta gamma theta vega iv_rank
```

Example:

```python
from neuroquant.options_lab import (
    load_option_chain_csv,
    VerticalBacktestConfig,
    backtest_bull_call_entries,
)

chain = load_option_chain_csv("data/options/tsla_chain_history.csv")

trades = backtest_bull_call_entries(
    chain,
    entry_timestamps=[
        "2026-01-15 20:00:00+00:00",
        "2026-03-02 20:00:00+00:00",
    ],
    symbol="TSLA",
    config=VerticalBacktestConfig(),
)
```

## Why the execution model is intentionally conservative

Paper brokers can fill at the midpoint even when a real order might not have
received that price. The lab therefore separates three values:

1. broker paper fill;
2. deterministic realistic fill;
3. conservative touch fill.

A strategy is not considered robust because it works on midpoint fills. It
should be re-run under wider spreads, worse slippage, larger commissions and
stricter liquidity filters.

## Long-volatility research hypothesis

One initial study supported by v1 is:

> When implied volatility is inexpensive relative to recent realized
> volatility, does a defined-risk bull call spread have positive out-of-sample
> expectancy after realistic transaction costs?

A rigorous experiment should predefine:

- signal used to choose entry dates;
- IV/RV threshold;
- target long and short deltas;
- DTE range;
- minimum liquidity;
- spread-width range;
- exits;
- fill model;
- commissions.

Then evaluate chronologically with train/test or walk-forward splits. Do not
optimize and score on the same period.

## Important v1 limitations

- No broker connection and no live order routing.
- No live data API in the repository.
- No American early-exercise/assignment engine yet.
- No dividend-event assignment model yet.
- No portfolio margin model.
- Backtests need historical snapshots dense enough to observe both legs at
  entries and exits; missing quotes produce no fabricated fill.
- BSM is a scenario/sanity-check tool, not a replacement for observed NBBO.

These limitations are deliberate. The next production-style milestones are a
provider adapter, early-assignment model, portfolio risk aggregation, and a
paper-broker adapter behind an interface that keeps research code broker-neutral.
