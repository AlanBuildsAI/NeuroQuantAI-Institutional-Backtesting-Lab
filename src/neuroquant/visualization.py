"""Matplotlib charts for the analytics dashboard.

Every chart is built to be read by a non-technical reviewer: a clear title,
a subtitle stating the analytical question, a short takeaway annotation, and
restrained styling. Charts are saved to ``docs/assets`` as both SVG (crisp,
version-control friendly) and PNG (easy to embed).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Headless backend so charts render without a display.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Consistent, calm visual style across all figures.
COLOR_STRATEGY = "#1f6feb"
COLOR_BASELINE = "#8b949e"
COLOR_NEGATIVE = "#d1242f"
COLOR_POSITIVE = "#1a7f37"
COLOR_CARD = "#0d1f3c"
COLOR_ACCENT = "#1f6feb"

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 140,
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def _save(fig: plt.Figure, assets_dir: Path, name: str) -> dict:
    """Save a figure as both SVG and PNG, returning their paths."""
    assets_dir.mkdir(parents=True, exist_ok=True)
    svg_path = assets_dir / f"{name}.svg"
    png_path = assets_dir / f"{name}.png"
    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    return {"svg": svg_path, "png": png_path}


def _titled(ax, title: str, question: str) -> None:
    """Set a bold title with extra padding plus an italic question subtitle.

    The subtitle is centered just below the title so the two never overlap,
    regardless of axes width.
    """
    ax.set_title(title, pad=26)
    ax.annotate(
        question,
        xy=(0.5, 1.015),
        xycoords="axes fraction",
        ha="center",
        fontsize=9,
        style="italic",
        color="#57606a",
        va="bottom",
    )


def plot_equity_curve(signals: pd.DataFrame, assets_dir: Path) -> dict:
    """Strategy vs baseline cumulative-return line chart."""
    fig, ax = plt.subplots(figsize=(8, 4.2))
    strat = (signals["strategy_equity"] - 1.0) * 100
    base = (signals["baseline_equity"] - 1.0) * 100

    ax.plot(signals.index, strat, label="Strategy", color=COLOR_STRATEGY, lw=2)
    ax.plot(
        signals.index,
        base,
        label="Baseline (buy & hold)",
        color=COLOR_BASELINE,
        lw=1.8,
        ls="--",
    )
    _titled(
        ax,
        "Strategy vs Baseline: Cumulative Return",
        "Does the rule add value over simply holding the series?",
    )
    ax.set_ylabel("Cumulative return (%)")
    ax.axhline(0, color="#d0d7de", lw=1)
    ax.legend(loc="upper left", frameon=False)

    final_strat = strat.iloc[-1]
    final_base = base.iloc[-1]
    verdict = "outperformed" if final_strat > final_base else "underperformed"
    ax.annotate(
        f"Strategy {verdict} the baseline\n"
        f"({final_strat:+.1f}% vs {final_base:+.1f}%)",
        xy=(0.99, 0.04),
        xycoords="axes fraction",
        ha="right",
        fontsize=9,
        color="#57606a",
        bbox=dict(boxstyle="round", fc="#f6f8fa", ec="#d0d7de"),
    )
    return _save(fig, assets_dir, "equity_curve")


def plot_drawdown(signals: pd.DataFrame, assets_dir: Path) -> dict:
    """Drawdown profile of the strategy equity curve."""
    equity = signals["strategy_equity"]
    drawdown = (equity / equity.cummax() - 1.0) * 100

    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.fill_between(
        signals.index, drawdown, 0, color=COLOR_NEGATIVE, alpha=0.30
    )
    ax.plot(signals.index, drawdown, color=COLOR_NEGATIVE, lw=1.2)
    _titled(
        ax,
        "Drawdown Profile",
        "How deep and how long are the strategy's losing stretches?",
    )
    ax.set_ylabel("Drawdown (%)")

    trough = drawdown.min()
    trough_date = drawdown.idxmin()
    ax.annotate(
        f"Worst drawdown: {trough:.1f}%",
        xy=(trough_date, trough),
        xytext=(0.35, 0.18),
        textcoords="axes fraction",
        fontsize=9,
        color="#57606a",
        arrowprops=dict(arrowstyle="->", color="#57606a"),
    )
    return _save(fig, assets_dir, "drawdown")


def plot_scenario_comparison(summary: pd.DataFrame, assets_dir: Path) -> dict:
    """Compare the top configurations across three KPIs side by side."""
    top = summary.head(6).copy()
    labels = [
        f"{int(r.short_window)}/{int(r.long_window)}"
        for r in top.itertuples()
    ]
    x = np.arange(len(top))

    fig, axes = plt.subplots(1, 3, figsize=(11, 4.4))
    panels = [
        ("sharpe_ratio", "Sharpe ratio", COLOR_STRATEGY),
        ("total_return", "Total return", COLOR_POSITIVE),
        ("max_drawdown", "Max drawdown", COLOR_NEGATIVE),
    ]
    for ax, (col, title, color) in zip(axes, panels):
        values = top[col].values
        if col in ("total_return",):
            values = values * 100
        if col == "max_drawdown":
            values = values * 100
        ax.bar(x, values, color=color, alpha=0.85)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.axhline(0, color="#d0d7de", lw=1)

    fig.suptitle(
        "Scenario Comparison: Top Configurations (short/long windows)",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )
    fig.text(
        0.5,
        0.90,
        "Which window pair balances reward (return, Sharpe) against risk "
        "(drawdown)?",
        ha="center",
        fontsize=9,
        style="italic",
        color="#57606a",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    return _save(fig, assets_dir, "scenario_comparison")


def plot_sweep_heatmap(summary: pd.DataFrame, assets_dir: Path) -> dict:
    """Heatmap of Sharpe ratio across short/long window combinations."""
    pivot = summary.pivot(
        index="short_window", columns="long_window", values="sharpe_ratio"
    ).sort_index()

    fig, ax = plt.subplots(figsize=(7.5, 5))
    data = pivot.values.astype(float)
    im = ax.imshow(data, cmap="RdYlBu", aspect="auto", origin="lower")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Long window")
    ax.set_ylabel("Short window")
    _titled(
        ax,
        "Parameter Sweep: Sharpe Ratio Heatmap",
        "Is performance robust across settings, or a lucky spike?",
    )
    ax.grid(False)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if not np.isnan(data[i, j]):
                ax.text(
                    j,
                    i,
                    f"{data[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="#24292f",
                )
    fig.colorbar(im, ax=ax, label="Sharpe ratio", shrink=0.85)
    return _save(fig, assets_dir, "sweep_heatmap")


def plot_return_distribution(signals: pd.DataFrame, assets_dir: Path) -> dict:
    """Histogram of daily strategy returns."""
    returns = signals["strategy_return"] * 100
    active = returns[signals["position"] > 0]

    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.hist(returns, bins=40, color=COLOR_STRATEGY, alpha=0.55, label="All days")
    ax.hist(active, bins=40, color=COLOR_POSITIVE, alpha=0.55, label="In-market days")
    ax.axvline(0, color="#57606a", lw=1)
    _titled(
        ax,
        "Daily Return Distribution",
        "Are returns symmetric, or skewed by a few large moves?",
    )
    ax.set_xlabel("Daily return (%)")
    ax.set_ylabel("Frequency")
    ax.legend(frameon=False)
    return _save(fig, assets_dir, "return_distribution")


def plot_kpi_dashboard(
    summary: pd.DataFrame,
    best_kpis: dict,
    assets_dir: Path,
    best_config,
) -> dict:
    """Executive KPI-card snapshot image (dashboard header).

    ``best_config`` is the configuration selected in-sample; the cards report
    that configuration's full-series KPIs so the label and the numbers always
    describe the same scenario.
    """
    config_label = f"{best_config.short_window} / {best_config.long_window}"
    cards = [
        ("Selected config", config_label, "in-sample pick"),
        ("Sharpe ratio", f"{best_kpis['sharpe_ratio']:.2f}", "full series"),
        ("Strategy return", f"{best_kpis['total_return'] * 100:+.1f}%", "cumulative"),
        ("Baseline return", f"{best_kpis['baseline_return'] * 100:+.1f}%", "buy & hold"),
        ("Max drawdown", f"{best_kpis['max_drawdown'] * 100:.1f}%", "worst decline"),
        ("Trades", f"{best_kpis['trade_count']}", "position changes"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(11, 4.4))
    fig.patch.set_facecolor("white")
    for ax, (label, value, sub) in zip(axes.flat, cards):
        ax.axis("off")
        ax.add_patch(
            plt.Rectangle(
                (0.02, 0.05),
                0.96,
                0.9,
                transform=ax.transAxes,
                facecolor=COLOR_CARD,
                edgecolor="none",
                zorder=0,
            )
        )
        ax.text(0.5, 0.74, label.upper(), ha="center", va="center",
                fontsize=9, color="#9bb4d8", transform=ax.transAxes)
        ax.text(0.5, 0.45, value, ha="center", va="center",
                fontsize=22, fontweight="bold", color="white",
                transform=ax.transAxes)
        ax.text(0.5, 0.18, sub, ha="center", va="center",
                fontsize=8, color="#7e9cc7", transform=ax.transAxes)

    fig.suptitle(
        "Executive Research Dashboard — Synthetic Quant Analytics",
        fontsize=14,
        fontweight="bold",
    )
    takeaway = (
        "Analyst takeaway: a reproducible research workflow — the configuration "
        "is selected in-sample and judged against a baseline out-of-sample."
    )
    fig.text(0.5, 0.005, takeaway, ha="center", fontsize=9,
             style="italic", color="#57606a")
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    return _save(fig, assets_dir, "dashboard_snapshot")


def plot_walk_forward(walk_forward: pd.DataFrame, assets_dir: Path) -> dict:
    """Out-of-sample vs in-sample Sharpe across walk-forward folds."""
    folds = walk_forward["fold"].astype(int).tolist()
    x = np.arange(len(folds))
    width = 0.38

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(
        x - width / 2,
        walk_forward["train_sharpe"],
        width,
        label="In-sample (train)",
        color=COLOR_BASELINE,
        alpha=0.85,
    )
    ax.bar(
        x + width / 2,
        walk_forward["test_sharpe"],
        width,
        label="Out-of-sample (test)",
        color=COLOR_STRATEGY,
        alpha=0.9,
    )
    ax.axhline(0, color="#d0d7de", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {f}" for f in folds])
    ax.set_ylabel("Sharpe ratio")
    _titled(
        ax,
        "Walk-Forward Validation: In-Sample vs Out-of-Sample",
        "Does the configuration chosen on training data hold up on unseen data?",
    )
    ax.legend(frameon=False, loc="best")

    held = float((walk_forward["test_sharpe"] > 0).mean()) * 100
    ax.annotate(
        f"{held:.0f}% of folds kept a positive\nout-of-sample Sharpe",
        xy=(0.99, 0.04),
        xycoords="axes fraction",
        ha="right",
        fontsize=9,
        color="#57606a",
        bbox=dict(boxstyle="round", fc="#f6f8fa", ec="#d0d7de"),
    )
    return _save(fig, assets_dir, "walk_forward")


def plot_monte_carlo(mc: dict, assets_dir: Path) -> dict:
    """Distribution of bootstrapped total returns (robustness analysis)."""
    total_returns = np.asarray(mc["total_returns"]) * 100
    p5 = mc["p5_return"] * 100
    median = mc["median_return"] * 100
    p95 = mc["p95_return"] * 100

    fig, ax = plt.subplots(figsize=(8, 4.2))
    counts, _, _ = ax.hist(
        total_returns, bins=40, color=COLOR_STRATEGY, alpha=0.55
    )
    ax.axvline(0, color="#57606a", lw=1)

    # Headroom above the tallest bar so the percentile labels never touch it.
    top = counts.max() * 1.22
    ax.set_ylim(0, top)
    for value, label, color in (
        (p5, "5th pct", COLOR_NEGATIVE),
        (median, "median", COLOR_POSITIVE),
        (p95, "95th pct", COLOR_POSITIVE),
    ):
        ax.axvline(value, color=color, ls="--", lw=1.4)
        ax.annotate(
            f"{label}\n{value:+.1f}%",
            xy=(value, top * 0.99),
            ha="center",
            va="top",
            fontsize=8,
            color=color,
        )
    _titled(
        ax,
        "Monte Carlo Robustness: Bootstrapped Total Return",
        "How stable is the outcome when the return stream is resampled?",
    )
    ax.set_xlabel("Total return (%)")
    ax.set_ylabel("Frequency")

    # Probability-of-loss callout in the lower-left, away from the labels.
    ax.annotate(
        f"P(loss) ≈ {mc['probability_of_loss'] * 100:.0f}% over "
        f"{mc['n_simulations']} resamples",
        xy=(0.01, 0.05),
        xycoords="axes fraction",
        ha="left",
        va="bottom",
        fontsize=9,
        color="#57606a",
        bbox=dict(boxstyle="round", fc="#f6f8fa", ec="#d0d7de"),
    )
    return _save(fig, assets_dir, "monte_carlo")


def build_all_charts(
    signals: pd.DataFrame,
    summary: pd.DataFrame,
    best_kpis: dict,
    assets_dir: Path,
    best_config,
    walk_forward: pd.DataFrame | None = None,
    monte_carlo: dict | None = None,
) -> dict:
    """Generate every chart and return a name -> {svg, png} mapping.

    ``walk_forward`` and ``monte_carlo`` are optional; when supplied, the
    walk-forward and Monte Carlo robustness charts are added to the output.
    """
    assets_dir = Path(assets_dir)
    charts = {
        "dashboard_snapshot": plot_kpi_dashboard(
            summary, best_kpis, assets_dir, best_config
        ),
        "equity_curve": plot_equity_curve(signals, assets_dir),
        "drawdown": plot_drawdown(signals, assets_dir),
        "scenario_comparison": plot_scenario_comparison(summary, assets_dir),
        "sweep_heatmap": plot_sweep_heatmap(summary, assets_dir),
        "return_distribution": plot_return_distribution(signals, assets_dir),
    }
    if walk_forward is not None and not walk_forward.empty:
        charts["walk_forward"] = plot_walk_forward(walk_forward, assets_dir)
    if monte_carlo is not None:
        charts["monte_carlo"] = plot_monte_carlo(monte_carlo, assets_dir)
    return charts
