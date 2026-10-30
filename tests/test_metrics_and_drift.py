from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraud_detection.drift import drift_table, fit_profile, psi, render_markdown, status
from fraud_detection.metrics import (
    cost_optimal_threshold,
    expected_cost,
    operating_point,
    precision_at_k,
    ranking_metrics,
    recall_at_precision,
    threshold_for_precision,
)
from fraud_detection.split import SplitConfig, time_split

Y = np.array([1, 0, 1, 0, 0, 1, 0, 0, 0, 0])
S = np.array([0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
AMOUNT = np.array([100.0, 1, 50, 1, 1, 400, 1, 1, 1, 1])


def test_precision_at_k():
    assert precision_at_k(Y, S, 1) == 1.0
    assert precision_at_k(Y, S, 2) == 0.5
    assert precision_at_k(Y, S, 3) == pytest.approx(2 / 3)
    assert precision_at_k(Y, S, 100) == pytest.approx(0.3)


def test_recall_at_precision_and_threshold():
    # top-1: P=1, R=1/3 ; top-3: P=2/3, R=2/3 ; top-6: P=1/2, R=1
    assert recall_at_precision(Y, S, 0.9) == pytest.approx(1 / 3)
    assert recall_at_precision(Y, S, 0.6) == pytest.approx(2 / 3)
    assert recall_at_precision(Y, S, 0.5) == pytest.approx(1.0)
    assert threshold_for_precision(Y, S, 0.6) == pytest.approx(0.8)
    assert threshold_for_precision(Y, S, 0.5) == pytest.approx(0.5)
    # the only fraud has the lowest score: no threshold reaches 50% precision
    assert threshold_for_precision(np.array([1, 0, 0, 0]), np.linspace(0, 1, 4), 0.5) == float(
        "inf"
    )


def test_cost_optimal_threshold_is_the_brute_force_minimum():
    review = 30.0
    best = cost_optimal_threshold(Y, S, AMOUNT, review)
    candidates = [*S, float("inf")]
    brute = min(candidates, key=lambda t: expected_cost(Y, S, AMOUNT, t, review))
    assert expected_cost(Y, S, AMOUNT, best, review) == expected_cost(Y, S, AMOUNT, brute, review)
    assert best == pytest.approx(0.5)  # catching the 400 fraud is worth six reviews


def test_cost_optimal_threshold_alerts_on_nothing_when_reviews_are_too_expensive():
    assert cost_optimal_threshold(Y, S, AMOUNT, review_cost=10_000) == float("inf")


def test_operating_point_counts():
    op = operating_point(Y, S, AMOUNT, threshold=0.8, review_cost=1.0)
    assert (op.alerts, op.true_positives, op.false_positives, op.false_negatives) == (3, 2, 1, 1)
    assert op.cost == pytest.approx(3 + 400)


def test_ranking_metrics_keys():
    m = ranking_metrics(Y, S, precision_floors=(0.5,), ks=(2,))
    assert set(m) == {
        "rows",
        "fraud_rate",
        "pr_auc",
        "roc_auc",
        "recall_at_precision_50",
        "precision_at_2",
    }
    assert 0 < m["pr_auc"] <= 1


def test_psi_is_zero_for_identical_and_large_for_shifted_distributions():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 20_000)
    profile = fit_profile(ref)
    assert psi(profile.proportions, profile.shares(rng.normal(0, 1, 20_000))) < 0.01
    assert psi(profile.proportions, profile.shares(rng.normal(1.0, 1, 20_000))) > 0.25
    assert status(0.05) == "stable" and status(0.2) == "moderate" and status(0.3) == "significant"


def test_discrete_features_get_one_bin_per_value():
    profile = fit_profile(np.array([0.0] * 90 + [1.0] * 10))
    np.testing.assert_allclose(profile.proportions, [0.9, 0.1])
    np.testing.assert_allclose(profile.shares(np.array([0.0, 1.0, 1.0, 1.0])), [0.25, 0.75])


def test_drift_table_and_markdown():
    profiles = {"a": fit_profile(np.arange(100.0)), "b": fit_profile(np.arange(100.0))}
    table = drift_table(profiles, {"a": np.arange(100.0), "b": np.arange(100.0) + 60})
    assert list(table["feature"]) == ["b", "a"]
    md = render_markdown(table, "Drift", {"model": "x"})
    assert "| b |" in md and "significant" in md


def test_time_split_is_ordered_disjoint_and_skips_burn_in():
    ts = pd.date_range("2026-01-01", periods=100 * 24, freq="h")
    frame = pd.DataFrame({"timestamp": ts, "x": np.arange(len(ts))})
    split = time_split(frame, SplitConfig(burn_in_days=10))
    assert split.train["timestamp"].min() >= pd.Timestamp("2026-01-11")
    assert split.train["timestamp"].max() < split.validation["timestamp"].min()
    assert split.validation["timestamp"].max() < split.test["timestamp"].min()
    assert split.test["timestamp"].max() == ts.max()
    total = len(split.train) + len(split.validation) + len(split.test)
    assert total == len(frame) - 10 * 24


@pytest.mark.parametrize(
    "kwargs",
    [dict(train_fraction=0.0), dict(train_fraction=0.7, validation_fraction=0.3)],
)
def test_split_config_validation(kwargs):
    with pytest.raises(ValueError):
        SplitConfig(**kwargs)
