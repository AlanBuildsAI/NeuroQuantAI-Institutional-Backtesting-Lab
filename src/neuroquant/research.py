"""Research-validity tools: train/test split, walk-forward, and robustness.

These functions add the discipline that separates a casual backtest from a
credible research study:

* :func:`split_train_test` — hold out an out-of-sample period so parameters
  are never chosen and judged on the same data,
* :func:`walk_forward_validation` — repeatedly select a configuration on a
  training window and evaluate it on the *next, unseen* window,
* :func:`monte_carlo_bootstrap` — resample the realised return stream to
  describe how fragile or stable an outcome is.

Everything here is descriptive robustness analysis on synthetic (or
user-supplied) data. None of it predicts markets or implies real-money
readiness.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import BacktestConfig, generate_signals, run_parameter_sweep
from .metrics import compute_kpis


def split_train_test(
    frame: pd.DataFrame, train_fraction: float = 0.7
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a time series into in-sample (train) and out-of-sample (test).

    The split is chronological — never shuffled — so the test period always
    comes strictly *after* the training period. The two slices do not overlap
    and together reconstruct the original frame.

    Parameters
    ----------
    frame:
        The full series, indexed in time order.
    train_fraction:
        Fraction of rows assigned to the training period (0 < f < 1).

    Returns
    -------
    tuple(DataFrame, DataFrame)
        ``(train, test)`` with no overlap.
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be strictly between 0 and 1.")

    split_at = int(len(frame) * train_fraction)
    if split_at <= 0 or split_at >= len(frame):
        raise ValueError(
            "train_fraction produces an empty train or test split; "
            "use more data or a more central fraction."
        )
    return frame.iloc[:split_at].copy(), frame.iloc[split_at:].copy()


def _kpis_on_slice(signals: pd.DataFrame, start: int) -> dict:
    """Recompute equity and KPIs for the rows from ``start`` onward.

    The moving averages in ``signals`` were computed on a longer frame, so the
    positions already have proper warm-up. We only need to rebase the equity
    curves to 1.0 at the start of the evaluation slice before scoring it.
    """
    sliced = signals.iloc[start:].copy()
    sliced["strategy_equity"] = (1.0 + sliced["strategy_return"]).cumprod()
    sliced["baseline_equity"] = (1.0 + sliced["baseline_return"]).cumprod()
    return compute_kpis(sliced)


def walk_forward_validation(
    frame: pd.DataFrame,
    short_windows: list[int],
    long_windows: list[int],
    train_size: int = 250,
    test_size: int = 125,
    cost_per_trade: float = 0.001,
) -> pd.DataFrame:
    """Run rolling walk-forward validation and return a tidy fold summary.

    For each fold we:

      1. take a training window of ``train_size`` rows,
      2. select the best configuration on that window by Sharpe ratio
         (parameters are chosen on training data only),
      3. evaluate that fixed configuration on the *next* ``test_size`` rows —
         data the selection never saw,
      4. record both the in-sample and out-of-sample headline KPIs.

    The window then steps forward by ``test_size`` and the process repeats.
    This is the standard guard against overfitting a single in-sample sweep.

    Returns
    -------
    pandas.DataFrame
        One row per fold with columns: ``fold``, ``train_start``,
        ``train_end``, ``test_end``, ``short_window``, ``long_window``,
        ``train_sharpe``, ``test_sharpe``, ``test_total_return``,
        ``test_baseline_return``, ``test_max_drawdown``.
    """
    n = len(frame)
    window = train_size + test_size
    if n < window:
        raise ValueError(
            f"Need at least train_size + test_size = {window} rows for "
            f"walk-forward validation, got {n}."
        )

    rows: list[dict] = []
    fold = 0
    start = 0
    while start + window <= n:
        train = frame.iloc[start : start + train_size]
        window_frame = frame.iloc[start : start + window]

        sweep = run_parameter_sweep(
            train, short_windows, long_windows, cost_per_trade=cost_per_trade
        )
        best = sweep.iloc[0]
        config = BacktestConfig(
            short_window=int(best.short_window),
            long_window=int(best.long_window),
            cost_per_trade=cost_per_trade,
        )

        # Evaluate the chosen config out-of-sample. Signals are built on the
        # full window so the test positions have proper moving-average warm-up,
        # then scored only on the test slice.
        signals = generate_signals(window_frame, config)
        test_kpis = _kpis_on_slice(signals, train_size)

        rows.append(
            {
                "fold": fold,
                "train_start": train.index[0].date().isoformat(),
                "train_end": train.index[-1].date().isoformat(),
                "test_end": window_frame.index[-1].date().isoformat(),
                "short_window": config.short_window,
                "long_window": config.long_window,
                "train_sharpe": float(best.sharpe_ratio),
                "test_sharpe": test_kpis["sharpe_ratio"],
                "test_total_return": test_kpis["total_return"],
                "test_baseline_return": test_kpis["baseline_return"],
                "test_max_drawdown": test_kpis["max_drawdown"],
            }
        )
        fold += 1
        start += test_size

    return pd.DataFrame(rows)


def monte_carlo_bootstrap(
    strategy_returns: pd.Series,
    n_simulations: int = 1000,
    seed: int = 123,
) -> dict:
    """Bootstrap the realised return stream to describe outcome stability.

    We resample the per-bar strategy returns *with replacement* to build many
    alternative equity paths of the same length, then summarise the spread of
    total return and drawdown across them. This is a robustness diagnostic —
    "how much did the ordering and selection of returns matter?" — and is not
    a forecast of future results.

    Parameters
    ----------
    strategy_returns:
        Per-bar strategy returns (e.g. ``signals['strategy_return']``).
    n_simulations:
        Number of bootstrap resamples to draw.
    seed:
        Seed for the resampling generator (keeps the analysis reproducible).

    Returns
    -------
    dict
        Summary statistics plus the raw simulated arrays for plotting:
        ``median_return``, ``p5_return``, ``p95_return``,
        ``probability_of_loss``, ``median_max_drawdown``, ``p5_max_drawdown``,
        ``n_simulations``, ``total_returns`` (np.ndarray),
        ``max_drawdowns`` (np.ndarray).
    """
    returns = np.asarray(strategy_returns, dtype=float)
    returns = returns[~np.isnan(returns)]
    n = returns.size
    if n == 0:
        raise ValueError("strategy_returns is empty; nothing to bootstrap.")

    rng = np.random.default_rng(seed)
    total_returns = np.empty(n_simulations)
    max_drawdowns = np.empty(n_simulations)

    for i in range(n_simulations):
        sample = rng.choice(returns, size=n, replace=True)
        equity = np.cumprod(1.0 + sample)
        total_returns[i] = equity[-1] - 1.0
        running_max = np.maximum.accumulate(equity)
        drawdown = equity / running_max - 1.0
        max_drawdowns[i] = drawdown.min()

    return {
        "median_return": float(np.median(total_returns)),
        "p5_return": float(np.percentile(total_returns, 5)),
        "p95_return": float(np.percentile(total_returns, 95)),
        "probability_of_loss": float(np.mean(total_returns < 0)),
        "median_max_drawdown": float(np.median(max_drawdowns)),
        "p5_max_drawdown": float(np.percentile(max_drawdowns, 5)),
        "n_simulations": int(n_simulations),
        "total_returns": total_returns,
        "max_drawdowns": max_drawdowns,
    }
