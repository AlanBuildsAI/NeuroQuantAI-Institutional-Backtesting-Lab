"""Shared pytest fixtures and import path setup."""

import sys
from pathlib import Path

import pytest

# Make the src/ package importable without an editable install.
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neuroquant import generate_synthetic_series  # noqa: E402


@pytest.fixture
def sample_data():
    """A small, reproducible synthetic signal series for tests."""
    return generate_synthetic_series(n_days=300, seed=7)
