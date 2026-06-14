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
    summary: pd.DataFrame,
    signals: pd.DataFrame,
    walk_forward: pd.DataFrame,
    output_dir: Path,
) -> dict:
    """Write the sweep summary, a sample equity curve, and walk-forward folds."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sweep_path = output_dir / "parameter_sweep_summary.csv"
    summary.round(6).to_csv(sweep_path, index=False)

    equity_path = output_dir / "equity_curve_sample.csv"
    equity_cols = [
        "close",
        "position",
        "strategy_return",
        "baseline_return",
        "strategy_equity",
        "baseline_equity",
    ]
    signals[equity_cols].round(6).to_csv(equity_path, index_label="date")

    walk_forward_path = output_dir / "walk_forward_summary.csv"
    walk_forward.round(6).to_csv(walk_forward_path, index=False)

    return {
        "summary_csv": sweep_path,
        "equity_csv": equity_path,
        "walk_forward_csv": walk_forward_path,
    }


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
    return (
        f'<img alt="{alt}" '
        f'src="data:image/png;base64,{encoded}" />'
    )


def _split_summary_html(in_sample_kpis: dict, out_of_sample_kpis: dict) -> str:
    """Small in-sample vs out-of-sample comparison block."""
    rows = [
        ("Sharpe ratio", "sharpe_ratio", "{:.2f}"),
        ("Total return", "total_return", "{:+.1%}"),
        ("Max drawdown", "max_drawdown", "{:.1%}"),
    ]
    body = "".join(
        f"<tr><td>{label}</td>"
        f"<td>{fmt.format(in_sample_kpis[key])}</td>"
        f"<td>{fmt.format(out_of_sample_kpis[key])}</td></tr>"
        for label, key, fmt in rows
    )
    return (
        "<table class='wf'><thead><tr><th>Metric</th>"
        "<th>In-sample (train)</th><th>Out-of-sample (test)</th>"
        f"</tr></thead><tbody>{body}</tbody></table>"
    )


def _walk_forward_html(walk_forward: pd.DataFrame) -> str:
    """Render the walk-forward fold table."""
    header = (
        "<tr><th>Fold</th><th>Train window</th><th>Config</th>"
        "<th>Train Sharpe</th><th>Test Sharpe</th><th>Test return</th></tr>"
    )
    body = "".join(
        f"<tr><td>{int(r.fold)}</td>"
        f"<td>{r.train_start} → {r.train_end}</td>"
        f"<td>{int(r.short_window)}/{int(r.long_window)}</td>"
        f"<td>{r.train_sharpe:.2f}</td>"
        f"<td>{r.test_sharpe:.2f}</td>"
        f"<td>{r.test_total_return:+.1%}</td></tr>"
        for r in walk_forward.itertuples()
    )
    return f"<table class='wf'><thead>{header}</thead><tbody>{body}</tbody></table>"


def _monte_carlo_html(mc: dict) -> str:
    """Render the Monte Carlo robustness summary."""
    return (
        "<table class='wf'><tbody>"
        f"<tr><td>Median total return</td><td>{mc['median_return']:+.1%}</td></tr>"
        f"<tr><td>5th / 95th percentile</td>"
        f"<td>{mc['p5_return']:+.1%} / {mc['p95_return']:+.1%}</td></tr>"
        f"<tr><td>Probability of loss</td>"
        f"<td>{mc['probability_of_loss']:.0%}</td></tr>"
        f"<tr><td>Median max drawdown</td>"
        f"<td>{mc['median_max_drawdown']:.1%}</td></tr>"
        f"<tr><td>Resamples</td><td>{mc['n_simulations']}</td></tr>"
        "</tbody></table>"
    )


def build_html_report(
    summary: pd.DataFrame,
    best_kpis: dict,
    chart_paths: dict,
    output_dir: Path,
    in_sample_kpis: dict,
    out_of_sample_kpis: dict,
    walk_forward: pd.DataFrame,
    monte_carlo: dict,
    best_config,
) -> Path:
    """Build the self-contained one-page dashboard.html research report.

    ``best_config`` is the configuration selected in-sample; every "selected
    configuration" figure references it (not the full-series sweep winner) so
    labels and numbers stay consistent.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_label = f"{best_config.short_window} / {best_config.long_window}"
    in_sample_sharpe = in_sample_kpis["sharpe_ratio"]

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

    oos_sharpe = out_of_sample_kpis["sharpe_ratio"]
    held = (
        float((walk_forward["test_sharpe"] > 0).mean())
        if not walk_forward.empty
        else 0.0
    )
    takeaway = (
        f"The {best_label}-window configuration was selected on the in-sample "
        f"period and then evaluated on unseen out-of-sample data "
        f"(out-of-sample Sharpe {oos_sharpe:.2f}). Across walk-forward folds, "
        f"{held:.0%} kept a positive out-of-sample Sharpe, and the Monte Carlo "
        f"resampling shows an estimated {monte_carlo['probability_of_loss']:.0%} "
        f"probability of a losing path. The deliverable is this evidence trail "
        f"— validation, baseline comparison, out-of-sample testing, and "
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
  header {{
    background: #0d1f3c; color: #fff; padding: 28px 32px;
  }}
  header h1 {{ margin: 0 0 6px; font-size: 22px; }}
  header p {{ margin: 0; color: #9bb4d8; font-size: 14px; }}
  main {{ max-width: 1040px; margin: 0 auto; padding: 24px 20px 48px; }}
  .disclaimer {{
    background: #fff8e6; border: 1px solid #f0d58c; border-radius: 8px;
    padding: 12px 16px; font-size: 13px; color: #6b5a1a; margin: 20px 0;
  }}
  h2 {{ font-size: 16px; margin: 28px 0 12px; }}
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
  footer {{ text-align: center; color: #8b949e; font-size: 12px;
    padding: 20px; }}
</style>
</head>
<body>
<header>
  <h1>NeuroQuantAI — Synthetic Quant Research &amp; Analytics Lab</h1>
  <p>Reproducible research workflow · validation → in/out-of-sample testing →
     walk-forward → robustness → reporting</p>
</header>
<main>
  <div class="disclaimer">
    <strong>Disclaimer.</strong> A quant-inspired research demonstration on
    <strong>synthetic data by default</strong> (optional local CSV research
    data is supported). It is <strong>not</strong> financial advice, an
    investment recommendation, a live-trading system, or a performance
    guarantee. No brokerage connections and no API keys are used anywhere.
  </div>

  <div class="best">
    <strong>Selected configuration:</strong> short/long windows
    <strong>{best_label}</strong> · in-sample Sharpe
    <strong>{in_sample_sharpe:.2f}</strong> · out-of-sample Sharpe
    <strong>{oos_sharpe:.2f}</strong> · chosen across
    <strong>{len(summary)}</strong> scenarios.
  </div>

  <div class="grid2">
    <div class="panel">
      <h3>In-sample vs out-of-sample</h3>
      {_split_summary_html(in_sample_kpis, out_of_sample_kpis)}
    </div>
    <div class="panel">
      <h3>Monte Carlo robustness</h3>
      {_monte_carlo_html(monte_carlo)}
    </div>
  </div>

  <h2>KPI Scorecard — Selected Configuration (full series)</h2>
  <div class="cards">{cards_html}</div>

  <h2>Walk-Forward Validation</h2>
  <div class="panel">{_walk_forward_html(walk_forward)}</div>

  <div class="takeaway"><strong>Analyst takeaway.</strong> {takeaway}</div>

  <h2>Visual Analysis</h2>
  {images_html}
</main>
<footer>Generated by the NeuroQuantAI research pipeline · synthetic data ·
  reproducible from a fixed seed.</footer>
</body>
</html>
"""

    report_path = output_dir / "dashboard.html"
    report_path.write_text(html, encoding="utf-8")
    return report_path
