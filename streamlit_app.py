"""NeuroQuantAI — Interactive Quant Research Lab (Streamlit).

A configurable, interactive presentation layer over the existing research
library. It lets a visitor pick a data source (synthetic or an uploaded CSV),
an **asset profile** and an **execution-assumption profile**, a candidate
signal and parameters, then run the same validation, walk-forward, Monte Carlo,
regime, cost-sensitivity and stress diagnostics used by the offline pipeline.

Asset and execution profiles model *research assumptions only* — they never
connect to a live broker, exchange, API, or market-data feed. This is a
synthetic research demo: not financial advice, not an investment
recommendation, not a live-trading system, and it makes no performance
guarantees. It performs no network calls and never persists uploaded data.

Run locally:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the src/ package importable without an editable install (works locally
# and on Streamlit Community Cloud, which runs from the repo root).
SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import streamlit as st  # noqa: E402

from neuroquant.app_helpers import (  # noqa: E402
    ASSET_PROFILES,
    EXECUTION_PROFILES,
    ResearchParams,
    analyst_warnings,
    drawdown_frame,
    equity_frame,
    load_uploaded_csv,
    make_synthetic,
    resolve_execution,
    run_research,
    scorecard_frame,
)
from neuroquant.signals import SIGNAL_FAMILIES  # noqa: E402

st.set_page_config(
    page_title="NeuroQuantAI — Interactive Quant Research Lab",
    page_icon="📈",
    layout="wide",
)

FAMILY_HELP = {
    "trend": "Trend follows a fast/slow moving-average crossover — the **short** "
    "and **long** windows matter most.",
    "momentum": "Momentum goes long on positive trailing return — the "
    "**momentum window** matters most.",
    "mean_reversion": "Mean reversion buys oversold dips — the **z-score window** "
    "and **threshold** matter most.",
    "composite": "Composite blends the trend, momentum and mean-reversion signals "
    "into one transparent score.",
}

TAB_CAPTIONS = {
    "Summary": "High-level view of the selected setup and the research takeaway.",
    "KPIs": "Risk/return metrics comparing the selected strategy to the baseline.",
    "Equity & drawdown": "Cumulative performance and downside behaviour over time.",
    "Walk-forward": "Tests whether parameter choices survive unseen periods.",
    "Monte Carlo": "Resamples observed returns to estimate robustness — not future "
    "performance.",
    "Regime": "Shows how results differ across volatility environments.",
    "Cost & stress": "Shows whether the result is fragile to costs or adverse "
    "assumptions.",
}

CONNECTIONS_NOTE = (
    "This public demo does not connect to live brokers, exchanges, or APIs. In a "
    "production/private version, broker-specific fees, spreads, margin rules, "
    "order types, latency, and data quality would materially affect results. "
    "This demo models those differences through configurable execution "
    "assumptions."
)


@st.cache_data(show_spinner=False)
def _cached_synthetic(n_days, seed, start_value, drift, volatility):
    """Cache synthetic generation so repeated runs with the same inputs are fast."""
    return make_synthetic(
        n_days=n_days, seed=seed, start_value=start_value, drift=drift,
        volatility=volatility,
    )


def _monte_carlo_figure(mc: dict):
    """Two-panel bootstrapped return / drawdown distribution figure."""
    total_returns = mc["total_returns"] * 100
    drawdowns = mc["max_drawdowns"] * 100
    fig, (ax_r, ax_d) = plt.subplots(1, 2, figsize=(10, 3.6))

    ax_r.hist(total_returns, bins=40, color="#1f6feb", alpha=0.6)
    ax_r.axvline(0, color="#57606a", lw=1)
    ax_r.axvline(mc["median_return"] * 100, color="#1a7f37", ls="--", lw=1.3)
    ax_r.axvline(mc["p5_return"] * 100, color="#d1242f", ls="--", lw=1.3)
    ax_r.axvline(mc["p95_return"] * 100, color="#1a7f37", ls="--", lw=1.3)
    ax_r.set_title("Total return distribution", fontsize=11)
    ax_r.set_xlabel("Total return (%)")
    ax_r.set_ylabel("Frequency")

    ax_d.hist(drawdowns, bins=40, color="#d1242f", alpha=0.5)
    ax_d.axvline(mc["median_max_drawdown"] * 100, color="#d1242f", ls="--", lw=1.3)
    ax_d.set_title("Max drawdown distribution", fontsize=11)
    ax_d.set_xlabel("Max drawdown (%)")
    ax_d.set_ylabel("Frequency")

    fig.tight_layout()
    return fig


def _render_note(note: dict) -> None:
    """Render an analyst note at the right severity level."""
    level = note["level"]
    if level == "warning":
        st.warning(note["text"], icon="⚠️")
    elif level == "success":
        st.success(note["text"], icon="✅")
    else:
        st.info(note["text"], icon="ℹ️")


# --- Header -----------------------------------------------------------------
st.title("NeuroQuantAI — Interactive Quant Research Lab")
st.caption(
    "A configurable synthetic research lab for testing candidate signals across "
    "asset and execution assumptions, with validation, walk-forward, Monte "
    "Carlo, regime, and cost-sensitivity diagnostics."
)
st.warning(
    "This is a synthetic research demo. It is not financial advice, not an "
    "investment recommendation, not a live-trading system, and does not make "
    "performance guarantees.",
    icon="⚠️",
)

with st.expander("How to use this lab", expanded=False):
    st.markdown(
        """
1. Choose a **data source** or upload a CSV.
2. Select an **asset profile** and an **execution-assumption profile**.
3. Choose a **signal family** and parameters.
4. Click **Run research**.
5. Review **KPIs**, **out-of-sample** results, **walk-forward** validation,
   **Monte Carlo** robustness, **regime** diagnostics, and **cost sensitivity**.
6. Treat results as **research diagnostics, not trading advice or forecasts**.
        """
    )

with st.expander("About broker / exchange connections", expanded=False):
    st.markdown(CONNECTIONS_NOTE)

# --- Sidebar controls -------------------------------------------------------
with st.sidebar:
    # A. Data source --------------------------------------------------------
    st.header("A · Data source")
    data_mode = st.radio("Data mode", ["Synthetic data", "Upload CSV"])

    # B. Asset profile ------------------------------------------------------
    st.header("B · Asset profile")
    asset_profile = st.selectbox("Asset profile", list(ASSET_PROFILES))
    asset = ASSET_PROFILES[asset_profile]
    st.caption(asset["note"] + " *(research assumption, not live data).*")

    uploaded = None
    n_days = 750
    seed = 42
    start_value = 100.0
    drift = float(asset["drift"])
    volatility = float(asset["volatility"])

    if data_mode == "Synthetic data":
        st.subheader("Synthetic controls")
        n_days = st.slider("Synthetic days", 300, 1500, 750, step=50)
        seed = st.number_input("Random seed", value=42, step=1)
        start_value = st.number_input("Start value", value=100.0, step=1.0)
        drift = st.number_input(
            "Drift", value=float(asset["drift"]), step=0.0001, format="%.4f"
        )
        volatility = st.number_input(
            "Volatility", value=float(asset["volatility"]), step=0.001,
            format="%.3f",
        )
    else:
        uploaded = st.file_uploader(
            "Upload CSV (date/timestamp + close columns)", type=["csv"]
        )
        st.caption(
            "Processed in-memory for this session only — never saved, never "
            "uploaded anywhere."
        )

    # C. Execution / connection assumptions ---------------------------------
    st.header("C · Execution profile")
    exec_names = list(EXECUTION_PROFILES)
    _default_exec = asset["execution"]
    exec_index = exec_names.index(_default_exec) if _default_exec in exec_names else 0
    execution_profile = st.selectbox(
        "Execution profile", exec_names, index=exec_index
    )
    st.caption("Simulated assumptions — **not** a real broker or exchange connection.")

    custom_cost = 0.001
    if execution_profile == "Custom assumptions":
        custom_cost = st.slider(
            "Cost per trade", 0.0, 0.01, 0.001, step=0.0005, format="%.4f"
        )
    assumptions = resolve_execution(execution_profile, custom_cost)

    if assumptions["is_custom"]:
        st.caption(f"Effective cost per trade: **{assumptions['cost']:.2%}**.")
    else:
        st.caption(
            f"Fee {assumptions['fee']:.2%} + spread {assumptions['spread']:.2%} + "
            f"slippage {assumptions['slippage']:.2%} = **{assumptions['cost']:.2%}** "
            "per trade."
        )
        st.caption(EXECUTION_PROFILES[execution_profile]["note"])

    # D. Strategy setup -----------------------------------------------------
    st.header("D · Strategy")
    _families = list(SIGNAL_FAMILIES)
    _default_family = (
        _families.index("mean_reversion") if "mean_reversion" in _families else 0
    )
    signal_family = st.selectbox("Signal family", _families, index=_default_family)
    st.caption(FAMILY_HELP[signal_family])

    # Sensible defaults; the relevant controls per family are shown prominently
    # and the rest live in an "Other parameters" expander to avoid clutter.
    short_window, long_window = 20, 60
    momentum_window, zscore_window, zscore_entry = 40, 40, 1.0

    if signal_family == "trend":
        short_window = st.slider("Short window", 2, 100, 20)
        long_window = st.slider("Long window", 5, 250, 60)
        with st.expander("Other parameters"):
            momentum_window = st.slider("Momentum window", 5, 250, 40)
            zscore_window = st.slider("Z-score window", 5, 120, 40)
            zscore_entry = st.slider("Z-score threshold", 0.5, 3.0, 1.0, 0.1)
    elif signal_family == "momentum":
        momentum_window = st.slider("Momentum window", 5, 250, 40)
        with st.expander("Other parameters"):
            short_window = st.slider("Short window", 2, 100, 20)
            long_window = st.slider("Long window", 5, 250, 60)
            zscore_window = st.slider("Z-score window", 5, 120, 40)
            zscore_entry = st.slider("Z-score threshold", 0.5, 3.0, 1.0, 0.1)
    elif signal_family == "mean_reversion":
        zscore_window = st.slider("Z-score window", 5, 120, 40)
        zscore_entry = st.slider("Z-score threshold", 0.5, 3.0, 1.0, 0.1)
        with st.expander("Other parameters"):
            short_window = st.slider("Short window", 2, 100, 20)
            long_window = st.slider("Long window", 5, 250, 60)
            momentum_window = st.slider("Momentum window", 5, 250, 40)
    else:  # composite
        short_window = st.slider("Short window", 2, 100, 20)
        long_window = st.slider("Long window", 5, 250, 60)
        momentum_window = st.slider("Momentum window", 5, 250, 40)
        zscore_window = st.slider("Z-score window", 5, 120, 40)
        zscore_entry = st.slider("Z-score threshold", 0.5, 3.0, 1.0, 0.1)

    use_volatility_filter = st.checkbox("Volatility filter", value=False)

    # E. Validation setup ---------------------------------------------------
    st.header("E · Validation")
    train_fraction = st.slider("Train fraction", 0.5, 0.9, 0.70, step=0.05)
    mc_simulations = st.slider("Monte Carlo simulations", 100, 2000, 500, step=100)
    show_cost = st.checkbox("Include cost sensitivity", value=True)
    st.caption("Walk-forward windows are sized automatically from the data length.")

    run_clicked = st.button("Run research", type="primary", use_container_width=True)


# --- Run --------------------------------------------------------------------
if run_clicked:
    try:
        if data_mode == "Upload CSV":
            if uploaded is None:
                st.error("Please upload a CSV file, or switch to synthetic data.")
                st.stop()
            frame = load_uploaded_csv(uploaded)
        else:
            frame = _cached_synthetic(
                int(n_days), int(seed), float(start_value), float(drift),
                float(volatility),
            )

        params = ResearchParams(
            signal_family=signal_family,
            short_window=int(short_window),
            long_window=int(long_window),
            momentum_window=int(momentum_window),
            zscore_window=int(zscore_window),
            zscore_entry=float(zscore_entry),
            cost_per_trade=float(assumptions["cost"]),
            train_fraction=float(train_fraction),
            use_volatility_filter=bool(use_volatility_filter),
            mc_simulations=int(mc_simulations),
        )
        with st.spinner("Running research workflow…"):
            st.session_state["result"] = run_research(frame, params)
            st.session_state["context"] = {
                "data_mode": data_mode,
                "asset_profile": asset_profile,
                "execution_profile": execution_profile,
                "assumptions": assumptions,
                "show_cost": show_cost,
            }
    except ValueError as exc:
        st.error(f"Could not run: {exc}")
    except Exception as exc:  # pragma: no cover - defensive UI guard
        st.error(f"Unexpected error: {exc}")


# --- Pre-run intro ----------------------------------------------------------
if "result" not in st.session_state:
    st.subheader("What this lab does")
    left, right = st.columns([3, 2])
    with left:
        st.markdown(
            """
- **What you can change:** the data source, asset profile, execution
  assumptions (fees / spread / slippage), signal family, parameters, and
  validation settings.
- **What the output means:** risk/return KPIs vs a baseline, out-of-sample and
  walk-forward stability, Monte Carlo robustness, regime attribution, and cost
  sensitivity.
- **What it does *not* do:** no live data, no brokerage/API connections, no
  order execution, and no forecasts or performance guarantees.
            """
        )
    with right:
        if data_mode == "Synthetic data":
            st.caption("Preview of the current synthetic series (no backtest yet):")
            preview = _cached_synthetic(
                int(n_days), int(seed), float(start_value), float(drift),
                float(volatility),
            )
            st.line_chart(preview.rename(columns={"close": "Synthetic close"}))
        else:
            st.caption("Upload a CSV in the sidebar to preview and analyse it.")
    st.info("Configure the sidebar (A → E), then click **Run research**.")
    st.stop()


# --- Output -----------------------------------------------------------------
result = st.session_state["result"]
context = st.session_state.get("context", {})
cfg = result["config"]
assumptions = context.get("assumptions", {"cost": cfg.cost_per_trade, "turnover_warn": 0.5})

# Analyst-style flags up top.
for note in analyst_warnings(result, assumptions.get("turnover_warn", 0.5)):
    _render_note(note)

tabs = st.tabs(list(TAB_CAPTIONS))

# 1 · Executive summary
with tabs[0]:
    st.subheader("Executive summary")
    st.caption(TAB_CAPTIONS["Summary"])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Signal family", cfg.signal_family)
    c2.metric("Asset profile", context.get("asset_profile", "—"))
    c3.metric("Execution", context.get("execution_profile", "—"))
    c4.metric("Cost / trade", f"{assumptions.get('cost', cfg.cost_per_trade):.2%}")
    d1, d2, d3 = st.columns(3)
    d1.metric("Data mode", context.get("data_mode", "—"))
    d2.metric("Observations", f"{result['n_obs']:,}")
    d3.metric("Train / test", f"{result['train_size']} / {result['test_size']}")
    st.write(
        f"**Selected configuration:** `{result['label']}` — short/long "
        f"{cfg.short_window}/{cfg.long_window}, momentum {cfg.momentum_window}, "
        f"z-score {cfg.zscore_window}/{cfg.zscore_entry:g}, volatility filter "
        f"{'on' if cfg.use_volatility_filter else 'off'}."
    )
    st.info(result["takeaway"], icon="🧭")

# 2 · KPI scorecards
with tabs[1]:
    st.subheader("KPI scorecard — in-sample vs out-of-sample")
    st.caption(TAB_CAPTIONS["KPIs"])
    oos = result["out_of_sample_kpis"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("OOS Sharpe", f"{oos['sharpe_ratio']:.2f}")
    m2.metric("OOS total return", f"{oos['total_return'] * 100:+.1f}%")
    m3.metric("OOS max drawdown", f"{oos['max_drawdown'] * 100:.1f}%")
    m4.metric("Trades (full)", f"{int(result['full_kpis']['trade_count'])}")
    st.dataframe(
        scorecard_frame(result["in_sample_kpis"], oos),
        use_container_width=True, hide_index=True,
    )
    st.caption(
        "Baseline = buy-and-hold over the same window; excess return is strategy "
        "minus baseline."
    )

# 3 · Equity & drawdown
with tabs[2]:
    st.subheader("Strategy vs baseline — cumulative return (%)")
    st.caption(TAB_CAPTIONS["Equity & drawdown"])
    st.line_chart(equity_frame(result["full_signals"]))
    st.subheader("Drawdown (%)")
    st.area_chart(drawdown_frame(result["full_signals"]))

# 4 · Walk-forward
with tabs[3]:
    st.subheader("Walk-forward validation")
    st.caption(TAB_CAPTIONS["Walk-forward"])
    wf = result["walk_forward"]
    if wf is None or wf.empty:
        st.warning(
            result["walk_forward_message"]
            or "Walk-forward validation is unavailable for this run."
        )
    else:
        held = float((wf["test_sharpe"] > 0).mean())
        st.metric("Folds with positive out-of-sample Sharpe", f"{held:.0%}")
        st.dataframe(
            wf[[
                "fold", "train_start", "train_end", "test_end", "label",
                "train_sharpe", "test_sharpe", "test_total_return",
            ]].round(3),
            use_container_width=True, hide_index=True,
        )
        st.bar_chart(wf.set_index("fold")[["train_sharpe", "test_sharpe"]])

# 5 · Monte Carlo
with tabs[4]:
    st.subheader("Monte Carlo robustness")
    st.caption(TAB_CAPTIONS["Monte Carlo"])
    mc = result["monte_carlo"]
    a, b, c, d, e = st.columns(5)
    a.metric("Median return", f"{mc['median_return'] * 100:+.1f}%")
    b.metric("5th pct", f"{mc['p5_return'] * 100:+.1f}%")
    c.metric("95th pct", f"{mc['p95_return'] * 100:+.1f}%")
    d.metric("P(loss)", f"{mc['probability_of_loss'] * 100:.0f}%")
    e.metric("Median max DD", f"{mc['median_max_drawdown'] * 100:.1f}%")
    st.pyplot(_monte_carlo_figure(mc))

# 6 · Regime diagnostics
with tabs[5]:
    st.subheader("Performance by volatility regime")
    st.caption(TAB_CAPTIONS["Regime"])
    regime = result["regime_summary"]
    if regime.empty:
        st.warning("No regime data available for this run.")
    else:
        display = regime.copy()
        for col in ("active_share", "total_return", "baseline_return", "max_drawdown"):
            display[col] = (display[col] * 100).round(1)
        display["sharpe_ratio"] = display["sharpe_ratio"].round(2)
        st.dataframe(display, use_container_width=True, hide_index=True)
        st.bar_chart(regime.set_index("regime")[["total_return"]])

# 7 · Cost sensitivity & stress
with tabs[6]:
    st.subheader("Cost sensitivity")
    st.caption(TAB_CAPTIONS["Cost & stress"])
    if context.get("show_cost", True):
        cost = result["cost_sensitivity"].copy()
        cost_display = cost.copy()
        cost_display["cost"] = (cost_display["cost"] * 100).map(lambda v: f"{v:.2f}%")
        for col in ("total_return", "max_drawdown"):
            cost_display[col] = (cost_display[col] * 100).round(1)
        cost_display["sharpe_ratio"] = cost_display["sharpe_ratio"].round(2)
        cost_display["turnover"] = cost_display["turnover"].round(3)
        st.dataframe(cost_display, use_container_width=True, hide_index=True)
        st.bar_chart(
            cost.assign(cost_pct=(cost["cost"] * 100)).set_index("cost_pct")[
                ["total_return"]
            ]
        )
    else:
        st.caption("Cost sensitivity hidden (toggle it on in the sidebar).")

    st.subheader("Stress diagnostics")
    st.caption("Labelled synthetic transforms — diagnostics, not predictions.")
    stress = result["stress_summary"].copy()
    for col in ("total_return", "max_drawdown"):
        stress[col] = (stress[col] * 100).round(1)
    stress["sharpe_ratio"] = stress["sharpe_ratio"].round(2)
    st.dataframe(stress, use_container_width=True, hide_index=True)

# Limitations
with st.expander("Limitations"):
    st.markdown(
        """
- **Synthetic data by default**; it has no real-world structure and results do
  not generalise to any market.
- A small set of **deliberately simple, explainable** signal families;
  long-or-flat positions only (no shorting, leverage, or sizing).
- The composite is a transparent score of simple signals — **not** a
  machine-learning model.
- **Asset and execution profiles model assumptions, not live connections.** No
  live data, no APIs, no brokerage connections, no order execution.
- Monte Carlo and stress sections are **robustness diagnostics, not forecasts**.
- **No guarantee of future results.** This is an educational/research demo only.
        """
    )
