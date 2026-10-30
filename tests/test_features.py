"""Leakage tests: a feature at time t must not change when anything after t changes."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraud_detection.features import FEATURE_NAMES, FeatureConfig, build_features, order_events

CFG = FeatureConfig(label_delay_seconds=2 * 86_400, risk_lookback_seconds=7 * 86_400)


def _features(df: pd.DataFrame) -> pd.DataFrame:
    return build_features(df, CFG).set_index("transaction_id")[FEATURE_NAMES]


@pytest.fixture(scope="module")
def events(small_transactions) -> pd.DataFrame:
    return order_events(small_transactions)


@pytest.fixture(scope="module")
def full(events) -> pd.DataFrame:
    return _features(events)


@pytest.mark.parametrize("fraction", [0.2, 0.5, 0.8])
def test_truncating_the_future_does_not_change_past_features(events, full, fraction):
    prefix = events.iloc[: int(len(events) * fraction)]
    truncated = _features(prefix)
    pd.testing.assert_frame_equal(truncated, full.loc[truncated.index])


def test_rewriting_the_future_does_not_change_past_features(events, full):
    cut = events["timestamp"].quantile(0.6)
    future = events["timestamp"] > cut
    rewritten = events.copy()
    rng = np.random.default_rng(0)
    rewritten.loc[future, "amount"] = rewritten.loc[future, "amount"] * 50
    rewritten.loc[future, "is_fraud"] = 1 - rewritten.loc[future, "is_fraud"]
    rewritten.loc[future, "device_id"] = "attacker"
    rewritten.loc[future, "country"] = "XX"
    rewritten.loc[future, "merchant_id"] = rng.permutation(
        rewritten.loc[future, "merchant_id"].to_numpy()
    )
    past_ids = events.loc[~future, "transaction_id"]
    after = _features(rewritten).loc[past_ids]
    pd.testing.assert_frame_equal(after, full.loc[past_ids])
    # and the rewrite really did change the future rows (the test is not vacuous)
    future_ids = events.loc[future, "transaction_id"]
    assert not _features(rewritten).loc[future_ids].equals(full.loc[future_ids])


def test_labels_inside_the_delay_window_are_invisible(events, full):
    row = events.iloc[int(len(events) * 0.7)]
    t = row["timestamp"]
    delay = pd.Timedelta(seconds=CFG.label_delay_seconds)
    recent = events["timestamp"] >= t - delay
    flipped = events.copy()
    flipped.loc[recent, "is_fraud"] = 1 - flipped.loc[recent, "is_fraud"]
    after = _features(flipped).loc[row["transaction_id"]]
    pd.testing.assert_series_equal(after, full.loc[row["transaction_id"]])


def test_matured_labels_do_move_merchant_risk(events, full):
    # Pick a transaction late in the log at a merchant with plenty of matured history.
    late = events.iloc[int(len(events) * 0.9) :]
    row = late.loc[late["merchant_id"].map(events["merchant_id"].value_counts()).idxmax()]
    t = row["timestamp"]
    lo = t - pd.Timedelta(seconds=CFG.label_delay_seconds + CFG.risk_lookback_seconds)
    hi = t - pd.Timedelta(seconds=CFG.label_delay_seconds)
    matured = (
        (events["merchant_id"] == row["merchant_id"])
        & (events["timestamp"] >= lo)
        & (events["timestamp"] < hi)
    )
    assert matured.sum() > 0
    flipped = events.copy()
    flipped.loc[matured, "is_fraud"] = 1
    after = _features(flipped).loc[row["transaction_id"], "merchant_risk"]
    assert after > full.loc[row["transaction_id"], "merchant_risk"]


def test_input_order_does_not_matter(events, full):
    shuffled = events.sample(frac=1.0, random_state=1)
    pd.testing.assert_frame_equal(_features(shuffled).loc[full.index], full)


def _toy() -> pd.DataFrame:
    rows = [
        # id, seconds, card, merchant, amount, country, device, label
        ("a1", 0, "c1", "m1", 10.0, "DE", "d1", 0),
        ("a2", 120, "c1", "m1", 20.0, "DE", "d1", 0),
        ("a3", 300, "c1", "m2", 30.0, "FR", "d2", 1),
        ("b1", 310, "c2", "m2", 5.0, "DE", "e1", 0),
        ("a4", 3_700, "c1", "m1", 90.0, "DE", "d1", 0),
    ]
    start = pd.Timestamp("2026-01-01 08:00:00")
    return pd.DataFrame(
        {
            "transaction_id": [r[0] for r in rows],
            "timestamp": [start + pd.Timedelta(seconds=r[1]) for r in rows],
            "card_id": [r[2] for r in rows],
            "merchant_id": [r[3] for r in rows],
            "merchant_category": "grocery",
            "amount": [r[4] for r in rows],
            "country": [r[5] for r in rows],
            "device_id": [r[6] for r in rows],
            "channel": "pos",
            "is_fraud": [r[7] for r in rows],
        }
    )


def test_hand_computed_card_features():
    f = build_features(_toy(), FeatureConfig(label_delay_seconds=0)).set_index("transaction_id")
    assert f.loc["a1", "card_tx_count_10m"] == 0
    assert f.loc["a3", "card_tx_count_10m"] == 2  # a1 (300 s ago) and a2 (180 s ago)
    assert f.loc["a4", "card_tx_count_10m"] == 0
    assert f.loc["a4", "card_tx_count_1h"] == 2  # a2 and a3; a1 is 3700 s old
    assert f.loc["a4", "card_tx_count_24h"] == 3
    assert f.loc["b1", "card_tx_count_24h"] == 0  # other cards never count

    # z-score of a4 against a1..a3: mean 20, sample std 10
    assert f.loc["a4", "amount_zscore"] == pytest.approx(7.0)
    assert f.loc["a2", "amount_zscore"] == 0  # fewer than two prior transactions
    assert f.loc["a4", "log_amount_ratio"] == pytest.approx(np.log(90 / 20))
    assert f.loc["a4", "log_card_amount_24h"] == pytest.approx(np.log1p(60.0))
    assert f.loc["a4", "log_seconds_since_last"] == pytest.approx(np.log1p(3_400))

    assert f.loc["a1", ["new_device", "device_change", "new_country"]].tolist() == [0, 0, 0]
    assert f.loc["a3", ["new_device", "device_change"]].tolist() == [1, 1]
    assert f.loc["a3", ["new_country", "country_change"]].tolist() == [1, 1]
    # back on a known device / country: a change, but nothing new
    assert f.loc["a4", ["new_device", "device_change"]].tolist() == [0, 1]
    assert f.loc["a4", ["new_country", "country_change"]].tolist() == [0, 1]


def test_hand_computed_merchant_risk_respects_the_label_delay():
    cfg = FeatureConfig(label_delay_seconds=100, prior_rate=0.0, prior_strength=1.0)
    f = build_features(_toy(), cfg).set_index("transaction_id")
    # b1 at 310 s, merchant m2: a3 (300 s, fraud) is only 10 s old -> not yet labelled
    assert f.loc["b1", "merchant_risk"] == 0.0
    # with no delay the fraud label of a3 is known (strictly earlier) -> 1 fraud / (1 + 1)
    cfg0 = FeatureConfig(label_delay_seconds=0, prior_rate=0.0, prior_strength=1.0)
    f0 = build_features(_toy(), cfg0).set_index("transaction_id")
    assert f0.loc["b1", "merchant_risk"] == pytest.approx(0.5)
    assert f0.loc["b1", "log_merchant_volume"] == pytest.approx(np.log1p(1))
