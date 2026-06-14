"""CSV exports and a self-contained one-page HTML report.

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
    "annualized_volatility": "Annualised volatility",
    "sharpe_ratio": "Sharpe ratio",
    "sortino_ratio": "Sortino ratio",
    "max_drawdown": "Max drawdown",
    "active_days": "Active (in-market) days",
    "trade_count": "Trade count",
    "correlation_to_baseline": "Correlation to baseline",
    "win_loss_ratio": "Win / loss ratio",
}

PERCENT_KEYS = {
    "total_return",
    "baseline_return",
    "annualized_volatility",
    "max_drawdown",
}


def export_csvs(
    summary: pd.DataFrame,
    signals: pd.DataFrame,
    output_dir: Path,
) -> dict:
    """Write the parameter-sweep summary and a sample equity curve to CSV."""
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

    return {"summary_csv": sweep_path, "equity_csv": equity_path}


def _format_kpi(key: str, value) -> str:
    """Human-friendly KPI formatting for the HTML cards."""
    if key in PERCENT_KEYS:
        return f"{value * 100:+.1f}%" if key != "annualized_volatility" else f"{value * 100:.1f}%"
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


def build_html_report(
    summary: pd.DataFrame,
    best_kpis: dict,
    chart_paths: dict,
    output_dir: Path,
) -> Path:
    """Build the self-contained one-page dashboard.html report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best = summary.iloc[0]
    best_label = f"{int(best.short_window)} / {int(best.long_window)}"

    cards_html = "".join(
        f'<div class="card"><div class="card-label">{KPI_LABELS[k]}</div>'
        f'<div class="card-value">{_format_kpi(k, v)}</div></div>'
        for k, v in best_kpis.items()
    )

    images_html = "".join(
        f'<figure>{_img_tag(p["png"], name)}'
        f'<figcaption>{name.replace("_", " ").title()}</figcaption></figure>'
        for name, p in chart_paths.items()
    )

    outperformed = best_kpis["total_return"] > best_kpis["baseline_return"]
    verdict = "outperformed" if outperformed else "underperformed"
    takeaway = (
        f"The best configuration ({best_label} windows) {verdict} the "
        f"buy-and-hold baseline on this synthetic series. The analytical "
        f"value here is the reproducible workflow: validated inputs, a "
        f"transparent baseline, a parameter sweep across "
        f"{len(summary)} scenarios, and a documented KPI scorecard — not the "
        f"profit figure itself."
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>NeuroQuant Analytics Dashboard</title>
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
  .card-value {{ font-size: 24px; font-weight: 700; margin-top: 6px;
    color: #0d1f3c; }}
  .best {{
    background: #ddf4ff; border: 1px solid #54aeff; border-radius: 8px;
    padding: 14px 18px; margin: 18px 0; font-size: 14px;
  }}
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
  <h1>NeuroQuant — Synthetic Signal Analytics Case Study</h1>
  <p>Reproducible Python analytics workflow · validation → experiments →
     KPIs → reporting</p>
</header>
<main>
  <div class="disclaimer">
    <strong>Disclaimer.</strong> This is an analytics demonstration built on
    <strong>synthetic data only</strong>. It is not a trading system, not
    financial advice, and makes no market-prediction or performance claims.
  </div>

  <div class="best">
    <strong>Best configuration:</strong> short/long windows
    <strong>{best_label}</strong> · Sharpe
    <strong>{best.sharpe_ratio:.2f}</strong> · evaluated across
    <strong>{len(summary)}</strong> scenarios.
  </div>

  <h2>KPI Scorecard — Best Configuration</h2>
  <div class="cards">{cards_html}</div>

  <div class="takeaway"><strong>Analyst takeaway.</strong> {takeaway}</div>

  <h2>Visual Analysis</h2>
  {images_html}
</main>
<footer>Generated by the NeuroQuant analytics pipeline · synthetic data ·
  reproducible from a fixed seed.</footer>
</body>
</html>
"""

    report_path = output_dir / "dashboard.html"
    report_path.write_text(html, encoding="utf-8")
    return report_path
