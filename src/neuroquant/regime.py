"""Regime-aware performance attribution.

We label every bar with a volatility regime (low / normal / high / stress) from
the realised series, then summarise how the selected configuration behaved
inside each regime. This answers a question a single headline number hides:
*was the result driven by calm periods, or did it survive turbulence?*

Regime labels here are descriptive (computed on the realised series for
attribution), not a trading signal — see :func:`neuroquant.features.add_regime_labels`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import REGIME_ORDER, add_regime_labels
from .metrics import TRADING_DAYS

# A regime needs at least this many bars before a Sharpe is meaningful.
MIN_REGIME_DAYS_FOR_SHARPE = 20


def _max_drawdown_from_returns(returns: pd.Series) -> float:
    """Max drawdown of an equity curve rebuilt from a return slice."""
    if returns.empty:
        return 0.0
    equity = (1.0 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return float(drawdown.min())


def summarize_by_regime(
    signals: pd.DataFrame, regime_window: int = 60
) -> pd.DataFrame:
    """Summarise strategy performance within each volatility regime.

    Parameters
    ----------
    signals:
        Output of :func:`neuroquant.backtest.generate_signals` (needs ``close``,
        ``position``, ``strategy_return`` and ``baseline_return``).
    regime_window:
        Rolling window used to derive volatility regimes.

    Returns
    -------
    pandas.DataFrame
        One row per regime (ordered low → stress) with columns: ``regime``,
        ``n_days``, ``active_days``, ``active_share``, ``total_return``,
        ``baseline_return``, ``sharpe_ratio``, ``max_drawdown``. Regimes with no
        observations are omitted.
    """
    labelled = add_regime_labels(signals[["close"]], regime_window)
    regime = labelled["vol_regime"]

    rows: list[dict] = []
    for name in REGIME_ORDER:
        mask = (regime == name).reindex(signals.index, fill_value=False)
        if not mask.any():
            continue

        strat = signals.loc[mask, "strategy_return"]
        base = signals.loc[mask, "baseline_return"]
        position = signals.loc[mask, "position"]
        n_days = int(mask.sum())

        if n_days >= MIN_REGIME_DAYS_FOR_SHARPE and strat.std(ddof=0) > 0:
            sharpe = float(
                strat.mean() * TRADING_DAYS / (strat.std(ddof=0) * np.sqrt(TRADING_DAYS))
            )
        else:
            sharpe = float("nan")

        active_days = int((position > 0).sum())
        rows.append(
            {
                "regime": name,
                "n_days": n_days,
                "active_days": active_days,
                "active_share": active_days / n_days if n_days else 0.0,
                "total_return": float((1.0 + strat).prod() - 1.0),
                "baseline_return": float((1.0 + base).prod() - 1.0),
                "sharpe_ratio": sharpe,
                "max_drawdown": _max_drawdown_from_returns(strat),
            }
        )

    return pd.DataFrame(rows)
