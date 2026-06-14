"""CSV exports and a self-contained one-page HTML research report.

The HTML dashboard inlines all styling and embeds charts as base64 PNGs, so
it opens offline with no external JavaScript, CDN, or network access.
"""

from __future__ import annotations

import base64
from pathlib import Path

import pandas as pd

KPI_LABELS = {
    "total_return": "Strategy total return",
    "baseline_return": "Baseline total return",
    "excess_return": "Excess return vs baseline",
    "annualized_return": "Annualised return",
    "annualized_volatility": "Annualised volatility",
    "sharpe_ratio": "Sharpe ratio",
    "sortino_ratio": "Sortino ratio",
    "calmar_ratio": "Calmar ratio",
    "information_ratio": "Information ratio",
    "max_drawdown": "Max drawdown",
    "active_days": "Active (in-market) days",
    "trade_count": "Trade count",
    "turnover": "Avg turnover",
    "correlation_to_baseline": "Correlation to baseline",
    "win_loss_ratio": "Win / loss ratio",
}

PERCENT_KEYS = {
    "total_return",
    "baseline_return",
    "excess_return",
    "annualized_return",
    "annualized_volatility",
    "max_drawdown",
}

SIGNED_PERCENT_KEYS = {
    "total_return",
    "baseline_return",
    "excess_return",
    "annualized_return",
}


def export_csvs(
    scenario_summary: pd.DataFrame,
    signals: pd.DataFrame,
    walk_forward: pd.DataFrame,
    regime_summary: pd.DataFrame,
    cost_sensitivity: pd.DataFrame,
    stress_summary: pd.DataFrame,
    feature_frame: pd.DataFrame,
    output_dir: Path,
) -> dict:
    """Write every tidy research table to ``output_dir`` as CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}

    paths["summary_csv"] = output_dir / "parameter_sweep_summary.csv"
    scenario_summary.round(6).to_csv(paths["summary_csv"], index=False)

    paths["equity_csv"] = output_dir / "equity_curve_sample.csv"
    equity_cols = [
        "close",
        "position",
        "strategy_return",
        "baseline_return",
        "strategy_equity",
        "baseline_equity",
    ]
    signals[equity_cols].round(6).to_csv(
        paths["equity_csv"], index_label="date"
    )

    paths["walk_forward_csv"] = output_dir / "walk_forward_summary.csv"
    walk_forward.round(6).to_csv(paths["walk_forward_csv"], index=False)

    paths["regime_csv"] = output_dir / "regime_summary.csv"
    regime_summary.round(6).to_csv(paths["regime_csv"], index=False)

    paths["cost_csv"] = output_dir / "cost_sensitivity.csv"
    cost_sensitivity.round(6).to_csv(paths["cost_csv"], index=False)

    paths["stress_csv"] = output_dir / "stress_test_summary.csv"
    stress_summary.round(6).to_csv(paths["stress_csv"], index=False)

    paths["feature_csv"] = output_dir / "feature_sample.csv"
    feature_frame.round(6).to_csv(paths["feature_csv"], index_label="date")

    return paths


def _format_kpi(key: str, value) -> str:
    """Human-friendly KPI formatting for the HTML cards."""
    if key in PERCENT_KEYS:
        sign = "+" if key in SIGNED_PERCENT_KEYS else ""
        return f"{value * 100:{sign}.1f}%"
    if key in ("active_days", "trade_count"):
        return f"{int(value)}"
    return f"{value:.2f}"


def _img_tag(png_path: Path, alt: str) -> str:
    """Return an <img> tag with the PNG embedded as base64 (offline-safe)."""
    encoded = base64.b64encode(Path(png_path).read_bytes()).decode("ascii")
    return f'<img alt="{alt}" src="data:image/png;base64,{encoded}" />'


def _table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a simple HTML table from string cells."""
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows
    )
    return f"<table class='wf'><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _split_summary_html(in_sample_kpis: dict, out_of_sample_kpis: dict) -> str:
    rows = [
        ["Sharpe ratio",
         f"{in_sample_kpis['sharpe_ratio']:.2f}",
         f"{out_of_sample_kpis['sharpe_ratio']:.2f}"],
        ["Total return",
         f"{in_sample_kpis['total_return']:+.1%}",
         f"{out_of_sample_kpis['total_return']:+.1%}"],
        ["Max drawdown",
         f"{in_sample_kpis['max_drawdown']:.1%}",
         f"{out_of_sample_kpis['max_drawdown']:.1%}"],
    ]
    return _table(["Metric", "In-sample", "Out-of-sample"], rows)


def _walk_forward_html(walk_forward: pd.DataFrame) -> str:
    rows = [
        [str(int(r.fold)),
         f"{r.train_start} → {r.train_end}",
         str(r.label),
         f"{r.train_sharpe:.2f}",
         f"{r.test_sharpe:.2f}",
         f"{r.test_total_return:+.1%}"]
        for r in walk_forward.itertuples()
    ]
    return _table(
        ["Fold", "Train window", "Selected", "Train Sharpe", "Test Sharpe",
         "Test return"],
        rows,
    )


def _monte_carlo_html(mc: dict) -> str:
    rows = [
        ["Median total return", f"{mc['median_return']:+.1%}"],
        ["5th / 95th percentile",
         f"{mc['p5_return']:+.1%} / {mc['p95_return']:+.1%}"],
        ["Probability of loss", f"{mc['probability_of_loss']:.0%}"],
        ["Median max drawdown", f"{mc['median_max_drawdown']:.1%}"],
        ["Resamples", str(mc["n_simulations"])],
    ]
    return _table([], rows)


def _cost_sensitivity_html(cost: pd.DataFrame) -> str:
    rows = [
        [f"{r.cost * 100:.2f}%",
         f"{r.total_return:+.1%}",
         f"{r.sharpe_ratio:.2f}",
         f"{r.max_drawdown:.1%}",
         str(int(r.trade_count))]
        for r in cost.itertuples()
    ]
    return _table(
        ["Cost / trade", "Total return", "Sharpe", "Max drawdown", "Trades"],
        rows,
    )


def _regime_html(regime: pd.DataFrame) -> str:
    rows = []
    for r in regime.itertuples():
        sharpe = "n/a" if pd.isna(r.sharpe_ratio) else f"{r.sharpe_ratio:.2f}"
        rows.append([
            r.regime.title(),
            str(int(r.n_days)),
            f"{r.active_share:.0%}",
            f"{r.total_return:+.1%}",
            sharpe,
            f"{r.max_drawdown:.1%}",
        ])
    return _table(
        ["Regime", "Days", "Active", "Total return", "Sharpe", "Max drawdown"],
        rows,
    )


def _stress_html(stress: pd.DataFrame) -> str:
    rows = [
        [r.scenario.replace("_", " ").title(),
         f"{r.total_return:+.1%}",
         f"{r.sharpe_ratio:.2f}",
         f"{r.max_drawdown:.1%}"]
        for r in stress.itertuples()
    ]
    return _table(["Scenario", "Total return", "Sharpe", "Max drawdown"], rows)


def build_html_report(
    scenario_summary: pd.DataFrame,
    best_kpis: dict,
    chart_paths: dict,
    output_dir: Path,
    in_sample_kpis: dict,
    out_of_sample_kpis: dict,
    walk_forward: pd.DataFrame,
    monte_carlo: dict,
    regime_summary: pd.DataFrame,
    cost_sensitivity: pd.DataFrame,
    stress_summary: pd.DataFrame,
    best_config,
) -> Path:
    """Build the self-contained one-page dashboard.html research report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from .backtest import config_label

    best_label = config_label(best_config)
    in_sample_sharpe = in_sample_kpis["sharpe_ratio"]
    oos_sharpe = out_of_sample_kpis["sharpe_ratio"]

    cards_html = "".join(
        f'<div class="card"><div class="card-label">{KPI_LABELS[k]}</div>'
        f'<div class="card-value">{_format_kpi(k, v)}</div></div>'
        for k, v in best_kpis.items()
        if k in KPI_LABELS
    )

    images_html = "".join(
        f'<figure>{_img_tag(p["png"], name)}'
        f'<figcaption>{name.replace("_", " ").title()}</figcaption></figure>'
        for name, p in chart_paths.items()
    )

    held = (
        float((walk_forward["test_sharpe"] > 0).mean())
        if not walk_forward.empty
        else 0.0
    )
    takeaway = (
        f"The candidate '{best_label}' was selected in-sample (across "
        f"{len(scenario_summary)} configurations spanning four signal families) "
        f"and then evaluated out-of-sample (Sharpe {oos_sharpe:.2f}). Across "
        f"walk-forward folds, {held:.0%} kept a positive out-of-sample Sharpe; "
        f"Monte Carlo resampling implies a ~{monte_carlo['probability_of_loss']:.0%} "
        f"chance of a losing path. The deliverable is this evidence trail — "
        f"feature engineering, baseline comparison, out-of-sample and "
        f"walk-forward testing, regime attribution, cost sensitivity and "
        f"robustness — not the headline number itself."
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>NeuroQuantAI — Synthetic Quant Research &amp; Analytics Lab</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{
    font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
    margin: 0; background: #f6f8fa; color: #1f2328;
  }}
  header {{ background: #0d1f3c; color: #fff; padding: 28px 32px; }}
  header h1 {{ margin: 0 0 6px; font-size: 22px; }}
  header p {{ margin: 0; color: #9bb4d8; font-size: 14px; }}
  main {{ max-width: 1040px; margin: 0 auto; padding: 24px 20px 48px; }}
  .disclaimer {{
    background: #fff8e6; border: 1px solid #f0d58c; border-radius: 8px;
    padding: 12px 16px; font-size: 13px; color: #6b5a1a; margin: 20px 0;
  }}
  h2 {{ font-size: 16px; margin: 30px 0 12px; }}
  .cards {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(165px, 1fr));
    gap: 12px;
  }}
  .card {{
    background: #fff; border: 1px solid #d0d7de; border-radius: 10px;
    padding: 16px; text-align: center;
  }}
  .card-label {{ font-size: 11px; text-transform: uppercase; color: #57606a;
    letter-spacing: .04em; }}
  .card-value {{ font-size: 22px; font-weight: 700; margin-top: 6px;
    color: #0d1f3c; }}
  .best {{
    background: #ddf4ff; border: 1px solid #54aeff; border-radius: 8px;
    padding: 14px 18px; margin: 18px 0; font-size: 14px;
  }}
  .grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }}
  @media (max-width: 720px) {{ .grid2 {{ grid-template-columns: 1fr; }} }}
  .panel {{ background: #fff; border: 1px solid #d0d7de; border-radius: 10px;
    padding: 16px 18px; }}
  .panel h3 {{ margin: 0 0 10px; font-size: 14px; }}
  table.wf {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  table.wf th, table.wf td {{ text-align: left; padding: 6px 8px;
    border-bottom: 1px solid #eaeef2; }}
  table.wf th {{ color: #57606a; font-weight: 600; }}
  .takeaway {{
    background: #fff; border-left: 4px solid #1f6feb; border-radius: 6px;
    padding: 14px 18px; margin: 18px 0; font-size: 14px; line-height: 1.55;
  }}
  figure {{ margin: 0 0 24px; background: #fff; border: 1px solid #d0d7de;
    border-radius: 10px; padding: 14px; }}
  figure img {{ width: 100%; height: auto; display: block; }}
  figcaption {{ font-size: 12px; color: #57606a; margin-top: 8px;
    text-align: center; }}
  footer {{ text-align: center; color: #8b949e; font-size: 12px; padding: 20px; }}
</style>
</head>
<body>
<header>
  <h1>NeuroQuantAI — Synthetic Quant Research &amp; Analytics Lab</h1>
  <p>Feature engineering · signal families · in/out-of-sample · walk-forward ·
     robustness · regime &amp; cost diagnostics</p>
</header>
<main>
  <div class="disclaimer">
    <strong>Disclaimer.</strong> A quant-inspired research demonstration on
    <strong>synthetic data by default</strong> (optional local CSV research
    data is supported). It is <strong>not</strong> financial advice, an
    investment recommendation, a live-trading system, or a performance
    guarantee. No brokerage connections and no API keys are used anywhere.
  </div>

  <h2>1 · Executive summary</h2>
  <div class="best">
    <strong>Selected candidate:</strong> <strong>{best_label}</strong> ·
    in-sample Sharpe <strong>{in_sample_sharpe:.2f}</strong> · out-of-sample
    Sharpe <strong>{oos_sharpe:.2f}</strong> · chosen across
    <strong>{len(scenario_summary)}</strong> configurations spanning four
    signal families.
  </div>

  <div class="grid2">
    <div class="panel">
      <h3>2 · In-sample vs out-of-sample</h3>
      {_split_summary_html(in_sample_kpis, out_of_sample_kpis)}
    </div>
    <div class="panel">
      <h3>3 · Monte Carlo robustness</h3>
      {_monte_carlo_html(monte_carlo)}
    </div>
  </div>

  <h2>4 · Walk-forward validation</h2>
  <div class="panel">{_walk_forward_html(walk_forward)}</div>

  <div class="grid2" style="margin-top:18px;">
    <div class="panel">
      <h3>5 · Cost sensitivity</h3>
      {_cost_sensitivity_html(cost_sensitivity)}
    </div>
    <div class="panel">
      <h3>6 · Regime analysis</h3>
      {_regime_html(regime_summary)}
    </div>
  </div>

  <h2>7 · Stress diagnostics</h2>
  <div class="panel">{_stress_html(stress_summary)}</div>

  <h2>8 · KPI scorecard — selected configuration (full series)</h2>
  <div class="cards">{cards_html}</div>

  <div class="takeaway"><strong>9 · Analyst takeaway.</strong> {takeaway}</div>

  <h2>10 · Visual analysis</h2>
  {images_html}

  <div class="disclaimer">
    <strong>Limitations.</strong> Synthetic data by default; a small set of
    deliberately simple, explainable signals; long-or-flat positions only; a
    flat per-trade cost/slippage assumption; no live trading, order routing, or
    real-money execution. Results are honest research diagnostics, including
    weak or negative ones — not a forecast or trading advice.
  </div>
</main>
<footer>Generated by the NeuroQuantAI research pipeline · synthetic data ·
  reproducible from a fixed seed.</footer>
</body>
</html>
"""

    report_path = output_dir / "dashboard.html"
    report_path.write_text(html, encoding="utf-8")
    return report_path
