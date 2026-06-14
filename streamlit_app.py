"""NeuroQuantAI — Interactive Quant Research Demo (Streamlit).

A thin, interactive presentation layer over the existing research library. It
lets a visitor generate synthetic data (or upload a local CSV for the session),
pick a candidate signal and parameters, and run the same validation,
walk-forward, Monte Carlo, regime, cost-sensitivity and stress diagnostics used
by the offline pipeline.

This is a synthetic research demo — not financial advice, not an investment
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
    ResearchParams,
    drawdown_frame,
    equity_frame,
    load_uploaded_csv,
    make_synthetic,
    run_research,
    scorecard_frame,
)
from neuroquant.signals import SIGNAL_FAMILIES  # noqa: E402

st.set_page_config(
    page_title="NeuroQuantAI — Interactive Quant Research Demo",
    page_icon="📈",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def _cached_synthetic(n_days, seed, start_value, drift, volatility):
    """Cache synthetic generation so repeated runs with the same inputs are fast."""
    return make_synthetic(
        n_days=n_days,
        seed=seed,
        start_value=start_value,
        drift=drift,
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


# --- Header -----------------------------------------------------------------
st.title("NeuroQuantAI — Interactive Quant Research Demo")
st.caption(
    "A synthetic quant research and analytics demo for testing candidate "
    "signals under validation, walk-forward, Monte Carlo, regime, and "
    "cost-sensitivity diagnostics."
)
st.warning(
    "This is a synthetic research demo. It is not financial advice, not an "
    "investment recommendation, not a live-trading system, and does not make "
    "performance guarantees.",
    icon="⚠️",
)

# --- Sidebar controls -------------------------------------------------------
with st.sidebar:
    st.header("Data")
    data_mode = st.radio("Data mode", ["Synthetic data", "Upload CSV"])

    uploaded = None
    n_days = 750
    seed = 42
    start_value = 100.0
    drift = 0.0002
    volatility = 0.012

    if data_mode == "Synthetic data":
        n_days = st.slider("Synthetic days", 300, 1500, 750, step=50)
        seed = st.number_input("Random seed", value=42, step=1)
        start_value = st.number_input("Start value", value=100.0, step=1.0)
        drift = st.number_input("Drift", value=0.0002, step=0.0001, format="%.4f")
        volatility = st.number_input(
            "Volatility", value=0.012, step=0.001, format="%.3f"
        )
    else:
        uploaded = st.file_uploader(
            "Upload CSV (date/timestamp + close columns)", type=["csv"]
        )
        st.caption(
            "Processed in-memory for this session only — never saved, never "
            "uploaded anywhere."
        )

    st.header("Research")
    signal_family = st.selectbox("Signal family", list(SIGNAL_FAMILIES))
    short_window = st.slider("Short window", 2, 100, 20)
    long_window = st.slider("Long window", 5, 250, 60)
    momentum_window = st.slider("Momentum window", 5, 250, 40)
    zscore_window = st.slider("Z-score window", 5, 120, 40)
    zscore_entry = st.slider("Z-score threshold", 0.5, 3.0, 1.0, step=0.1)
    cost_per_trade = st.slider(
        "Cost per trade", 0.0, 0.01, 0.001, step=0.0005, format="%.4f"
    )
    train_fraction = st.slider("Train fraction", 0.5, 0.9, 0.70, step=0.05)
    use_volatility_filter = st.checkbox("Volatility filter", value=False)
    mc_simulations = st.slider("Monte Carlo simulations", 100, 2000, 500, step=100)

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
            cost_per_trade=float(cost_per_trade),
            train_fraction=float(train_fraction),
            use_volatility_filter=bool(use_volatility_filter),
            mc_simulations=int(mc_simulations),
        )
        with st.spinner("Running research workflow…"):
            st.session_state["result"] = run_research(frame, params)
            st.session_state["data_mode"] = data_mode
    except ValueError as exc:
        st.error(f"Could not run: {exc}")
    except Exception as exc:  # pragma: no cover - defensive UI guard
        st.error(f"Unexpected error: {exc}")


# --- Output -----------------------------------------------------------------
if "result" not in st.session_state:
    st.info("Set your controls in the sidebar and click **Run research**.")
    st.stop()

result = st.session_state["result"]
cfg = result["config"]

tabs = st.tabs(
    [
        "Summary",
        "KPIs",
        "Equity & drawdown",
        "Walk-forward",
        "Monte Carlo",
        "Regime",
        "Cost & stress",
    ]
)

# 1 · Executive summary
with tabs[0]:
    st.subheader("Executive summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Signal family", cfg.signal_family)
    c2.metric("Data mode", st.session_state.get("data_mode", "—"))
    c3.metric("Observations", f"{result['n_obs']:,}")
    c4.metric("Train / test", f"{result['train_size']} / {result['test_size']}")
    st.write(
        f"**Selected configuration:** `{result['label']}` — "
        f"short/long {cfg.short_window}/{cfg.long_window}, momentum "
        f"{cfg.momentum_window}, z-score {cfg.zscore_window}/{cfg.zscore_entry:g}, "
        f"cost {cfg.cost_per_trade:.2%}, "
        f"volatility filter {'on' if cfg.use_volatility_filter else 'off'}."
    )
    st.info(result["takeaway"], icon="🧭")

# 2 · KPI scorecards
with tabs[1]:
    st.subheader("KPI scorecard — in-sample vs out-of-sample")
    oos = result["out_of_sample_kpis"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("OOS Sharpe", f"{oos['sharpe_ratio']:.2f}")
    m2.metric("OOS total return", f"{oos['total_return'] * 100:+.1f}%")
    m3.metric("OOS max drawdown", f"{oos['max_drawdown'] * 100:.1f}%")
    m4.metric("Trades (full)", f"{int(result['full_kpis']['trade_count'])}")
    st.dataframe(
        scorecard_frame(result["in_sample_kpis"], oos),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "Baseline = buy-and-hold over the same window; excess return is "
        "strategy minus baseline."
    )

# 3 · Equity curve & drawdown
with tabs[2]:
    st.subheader("Strategy vs baseline — cumulative return (%)")
    st.line_chart(equity_frame(result["full_signals"]))
    st.subheader("Drawdown (%)")
    st.area_chart(drawdown_frame(result["full_signals"]))

# 4 · Walk-forward
with tabs[3]:
    st.subheader("Walk-forward validation")
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
            wf[
                [
                    "fold", "train_start", "train_end", "test_end", "label",
                    "train_sharpe", "test_sharpe", "test_total_return",
                ]
            ].round(3),
            use_container_width=True,
            hide_index=True,
        )
        st.bar_chart(
            wf.set_index("fold")[["train_sharpe", "test_sharpe"]]
        )

# 5 · Monte Carlo
with tabs[4]:
    st.subheader("Monte Carlo robustness")
    st.caption(
        "Bootstrapped resamples of the realised return stream — robustness "
        "analysis describing the spread of outcomes, **not** a forecast."
    )
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
    cost = result["cost_sensitivity"].copy()
    cost_display = cost.copy()
    cost_display["cost"] = (cost_display["cost"] * 100).map(lambda v: f"{v:.2f}%")
    for col in ("total_return", "max_drawdown"):
        cost_display[col] = (cost_display[col] * 100).round(1)
    cost_display["sharpe_ratio"] = cost_display["sharpe_ratio"].round(2)
    cost_display["turnover"] = cost_display["turnover"].round(3)
    st.dataframe(cost_display, use_container_width=True, hide_index=True)
    st.bar_chart(cost.assign(cost_pct=(cost["cost"] * 100)).set_index("cost_pct")[
        ["total_return"]
    ])

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
- **No live data, no APIs, no brokerage connections, no trading execution**
  beyond a simplified flat cost/slippage assumption.
- Monte Carlo and stress sections are **robustness diagnostics, not forecasts**.
- **No guarantee of future results.** This is an educational/research demo only.
        """
    )
