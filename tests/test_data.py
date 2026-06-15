"""Tests for synthetic data reproducibility and the optional CSV loader."""

import pandas as pd
import pytest

from neuroquant.data import (
    data_quality_report,
    generate_synthetic_series,
    load_csv_series,
)


def test_same_seed_is_reproducible():
    a = generate_synthetic_series(n_days=200, seed=11)
    b = generate_synthetic_series(n_days=200, seed=11)
    pd.testing.assert_frame_equal(a, b)


def test_different_seed_differs():
    a = generate_synthetic_series(n_days=200, seed=11)
    b = generate_synthetic_series(n_days=200, seed=12)
    assert not a["close"].equals(b["close"])


def _write_csv(tmp_path, text, name="data.csv"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_csv_loader_reads_valid_file(tmp_path):
    path = _write_csv(
        tmp_path,
        "date,close,volume\n2021-01-01,100\n2021-01-02,101\n2021-01-03,102\n",
    )
    frame = load_csv_series(path)
    assert list(frame.columns)[0] == "close"
    assert isinstance(frame.index, pd.DatetimeIndex)
    assert frame.index.is_monotonic_increasing
    assert len(frame) == 3


def test_csv_loader_sorts_unsorted_timestamps(tmp_path):
    path = _write_csv(
        tmp_path,
        "date,close\n2021-01-03,102\n2021-01-01,100\n2021-01-02,101\n",
    )
    frame = load_csv_series(path)
    assert frame.index.is_monotonic_increasing
    assert frame["close"].iloc[0] == 100


def test_csv_loader_rejects_missing_close(tmp_path):
    path = _write_csv(tmp_path, "date,price\n2021-01-01,100\n2021-01-02,101\n")
    with pytest.raises(ValueError, match="close"):
        load_csv_series(path)


def test_csv_loader_rejects_duplicate_timestamps(tmp_path):
    path = _write_csv(
        tmp_path, "date,close\n2021-01-01,100\n2021-01-01,101\n"
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_csv_series(path)


def test_csv_loader_rejects_non_positive_close(tmp_path):
    path = _write_csv(
        tmp_path, "date,close\n2021-01-01,100\n2021-01-02,-5\n"
    )
    with pytest.raises(ValueError, match="non-positive"):
        load_csv_series(path)


def test_csv_loader_rejects_missing_timestamp_column(tmp_path):
    path = _write_csv(tmp_path, "close\n100\n101\n")
    with pytest.raises(ValueError, match="timestamp"):
        load_csv_series(path)


def test_csv_loader_keeps_benchmark_close(tmp_path):
    path = _write_csv(
        tmp_path,
        "date,close,benchmark_close\n"
        "2021-01-01,100,50\n2021-01-02,101,51\n2021-01-03,102,52\n",
    )
    frame = load_csv_series(path)
    assert "benchmark_close" in frame.columns


def test_data_quality_report_flags_short_and_missing_benchmark():
    frame = generate_synthetic_series(n_days=120, seed=1)
    warnings = data_quality_report(frame)
    text = " ".join(warnings).lower()
    assert "short series" in text
    assert "benchmark" in text


def test_data_quality_report_flags_zero_returns():
    frame = generate_synthetic_series(n_days=400, seed=1).copy()
    # Force many repeated values -> a high share of exactly-zero returns.
    frame.iloc[100:300, frame.columns.get_loc("close")] = 123.0
    warnings = data_quality_report(frame)
    assert any("zero" in w.lower() for w in warnings)


def test_data_quality_report_clean_long_series_with_benchmark():
    frame = generate_synthetic_series(n_days=600, seed=1).copy()
    frame["benchmark_close"] = frame["close"].to_numpy() * 0.5
    assert data_quality_report(frame) == []
