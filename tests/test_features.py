"""Tests for the feature engineering module."""

import numpy as np

from neuroquant.features import (
    REGIME_ORDER,
    add_regime_labels,
    build_feature_frame,
)


def test_feature_frame_has_expected_columns(sample_data):
    feats = build_feature_frame(sample_data)
    expected = {
        "return_1d",
        "log_return",
        "ma_20",
        "price_to_ma_20",
        "zscore_20",
        "momentum_20",
        "vol_20",
        "drawdown_from_high_60",
        "vol_regime",
    }
    assert expected.issubset(feats.columns)
    # The original close is preserved and the index stays chronological.
    assert "close" in feats.columns
    assert feats.index.is_monotonic_increasing


def test_no_future_leakage(sample_data):
    """Causality: features on a prefix match features on the full series.

    If any feature peeked ahead, truncating the series would change earlier
    values. We compare a representative causal feature on a prefix vs the full
    frame and require them to match within the prefix (after warm-up).
    """
    full = build_feature_frame(sample_data)
    prefix = build_feature_frame(sample_data.iloc[:200])
    for col in ("vol_20", "momentum_20", "zscore_20", "price_to_ma_20"):
        a = full[col].iloc[:200]
        b = prefix[col]
        # Compare where both are defined (ignore warm-up NaNs).
        mask = a.notna() & b.notna()
        assert mask.sum() > 0
        assert np.allclose(a[mask].to_numpy(), b[mask].to_numpy())


def test_regime_labels_exist_after_warmup(sample_data):
    labelled = add_regime_labels(sample_data, volatility_window=60)
    labels = labelled["vol_regime"]
    # Warm-up rows are NaN; later rows carry a valid ordered regime label.
    assert labels.iloc[:59].isna().all()
    valid = labels.dropna()
    assert len(valid) > 0
    assert set(valid.unique()).issubset(set(REGIME_ORDER))
