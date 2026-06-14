"""End-to-end pipeline tests: artefact generation and a self-contained report."""

import pytest

from neuroquant.pipeline import run_pipeline

EXPECTED_ASSETS = [
    "dashboard_snapshot.png",
    "equity_curve.png",
    "scenario_comparison.png",
    "sweep_heatmap.png",
    "walk_forward.png",
    "monte_carlo.png",
    "regime_heatmap.png",
    "cost_sensitivity.png",
]

EXPECTED_OUTPUTS = [
    "dashboard.html",
    "parameter_sweep_summary.csv",
    "equity_curve_sample.csv",
    "walk_forward_summary.csv",
    "regime_summary.csv",
    "cost_sensitivity.csv",
    "stress_test_summary.csv",
    "feature_sample.csv",
]


@pytest.fixture(scope="module")
def pipeline_run(tmp_path_factory):
    """Run the pipeline once into a temp area and share across assertions."""
    base = tmp_path_factory.mktemp("nq")
    assets_dir = base / "assets"
    output_dir = base / "outputs"
    result = run_pipeline(
        n_days=400,
        assets_dir=assets_dir,
        output_dir=output_dir,
        verbose=False,
    )
    return result, assets_dir, output_dir


def test_pipeline_creates_expected_assets(pipeline_run):
    _, assets_dir, _ = pipeline_run
    for name in EXPECTED_ASSETS:
        assert (assets_dir / name).exists(), f"missing asset {name}"


def test_pipeline_creates_expected_outputs(pipeline_run):
    _, _, output_dir = pipeline_run
    for name in EXPECTED_OUTPUTS:
        assert (output_dir / name).exists(), f"missing output {name}"


def test_dashboard_is_self_contained(pipeline_run):
    _, _, output_dir = pipeline_run
    html = (output_dir / "dashboard.html").read_text(encoding="utf-8")
    # No external JavaScript and no CDN / network references.
    assert "<script" not in html.lower()
    assert "http://" not in html
    assert "https://" not in html
    # Images are embedded inline as base64 data URIs.
    assert "data:image/png;base64," in html
    # The new project title is present.
    assert "Synthetic Quant Research" in html


def test_pipeline_reports_in_and_out_of_sample(pipeline_run):
    result, _, _ = pipeline_run
    assert "in_sample_kpis" in result
    assert "out_of_sample_kpis" in result
    assert "walk_forward" in result
    assert not result["walk_forward"].empty


def test_dashboard_contains_new_sections(pipeline_run):
    _, _, output_dir = pipeline_run
    html = (output_dir / "dashboard.html").read_text(encoding="utf-8")
    for heading in (
        "Executive summary",
        "Walk-forward validation",
        "Monte Carlo robustness",
        "Cost sensitivity",
        "Regime analysis",
        "Stress diagnostics",
        "KPI scorecard",
    ):
        assert heading in html, f"missing dashboard section: {heading}"


def test_pipeline_produces_diagnostics(pipeline_run):
    result, _, _ = pipeline_run
    assert not result["regime_summary"].empty
    assert not result["cost_sensitivity"].empty
    assert not result["stress_summary"].empty
    assert "feature_frame" in result
