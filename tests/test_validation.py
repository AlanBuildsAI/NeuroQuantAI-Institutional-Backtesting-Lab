"""Tests for input validation checks."""

import numpy as np
import pandas as pd
import pytest

from neuroquant.validation import (
    ValidationError,
    validate_price_frame,
    validate_window_config,
)


def test_accepts_valid_frame(sample_data):
    assert validate_price_frame(sample_data) is sample_data


def test_rejects_missing_close_column():
    frame = pd.DataFrame({"price": [100.0, 101.0, 102.0]})
    with pytest.raises(ValidationError, match="Missing required column"):
        validate_price_frame(frame)


def test_rejects_missing_values():
    frame = pd.DataFrame({"close": [100.0, np.nan, 102.0]})
    with pytest.raises(ValidationError, match="missing/NaN"):
        validate_price_frame(frame)


def test_rejects_non_positive_values():
    frame = pd.DataFrame({"close": [100.0, -5.0, 102.0]})
    with pytest.raises(ValidationError, match="non-positive"):
        validate_price_frame(frame)


def test_rejects_insufficient_rows():
    frame = pd.DataFrame({"close": [100.0]})
    with pytest.raises(ValidationError, match="Insufficient rows"):
        validate_price_frame(frame, min_rows=10)


def test_window_config_rejects_short_ge_long():
    with pytest.raises(ValidationError, match="strictly less"):
        validate_window_config(60, 20)
    with pytest.raises(ValidationError, match="strictly less"):
        validate_window_config(30, 30)


def test_window_config_accepts_valid():
    assert validate_window_config(20, 60) is None
