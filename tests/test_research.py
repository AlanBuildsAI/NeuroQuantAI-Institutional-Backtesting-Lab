"""Tests for train/test split, walk-forward validation, and Monte Carlo."""

import numpy as np
import pytest

from neuroquant.backtest import BacktestConfig, generate_signals
from neuroquant.research import (
    monte_carlo_bootstrap,
    split_train_test,
    walk_forward_validation,
)


def test_split_train_test_no_overlap(sample_data):
    train, test = split_train_test(sample_data, train_fraction=0.7)
    # Sizes add up and the periods do not overlap in time.
    assert len(train) + len(test) == len(sample_data)
    assert train.index.max() < test.index.min()
    assert len(train) > len(test)


def test_split_train_test_rejects_bad_fraction(sample_data):
    with pytest.raises(ValueError):
        split_train_test(sample_data, train_fraction=1.0)


def test_walk_forward_returns_expected_columns(sample_data):
    wf = walk_forward_validation(
        sample_data,
        train_size=150,
        test_size=60,
    )
    expected = {
        "fold",
        "train_start",
        "train_end",
        "test_end",
        "signal_family",
        "label",
        "train_sharpe",
        "test_sharpe",
        "test_total_return",
        "test_baseline_return",
        "test_max_drawdown",
    }
    assert expected.issubset(wf.columns)
    assert len(wf) >= 1
    # Folds are numbered sequentially from zero.
    assert wf["fold"].tolist() == list(range(len(wf)))


def test_walk_forward_requires_enough_rows(sample_data):
    with pytest.raises(ValueError, match="walk-forward"):
        walk_forward_validation(
            sample_data,
            train_size=250,
            test_size=125,  # 375 > 300-row sample
        )


def test_monte_carlo_summary_fields(sample_data):
    signals = generate_signals(sample_data, BacktestConfig(20, 60))
    mc = monte_carlo_bootstrap(
        signals["strategy_return"], n_simulations=200, seed=5
    )
    for key in (
        "median_return",
        "p5_return",
        "p95_return",
        "probability_of_loss",
        "median_max_drawdown",
        "n_simulations",
    ):
        assert key in mc
    assert 0.0 <= mc["probability_of_loss"] <= 1.0
    assert mc["p5_return"] <= mc["median_return"] <= mc["p95_return"]
    assert len(mc["total_returns"]) == 200


def test_monte_carlo_is_reproducible(sample_data):
    signals = generate_signals(sample_data, BacktestConfig(20, 60))
    a = monte_carlo_bootstrap(signals["strategy_return"], n_simulations=100, seed=9)
    b = monte_carlo_bootstrap(signals["strategy_return"], n_simulations=100, seed=9)
    assert np.allclose(a["total_returns"], b["total_returns"])
