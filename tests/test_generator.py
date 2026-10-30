from __future__ import annotations

from dataclasses import replace

import pandas as pd

from fraud_detection.generator import (
    COLUMNS,
    HIGH_VALUE_CATEGORIES,
    GeneratorConfig,
    generate_transactions,
)

TINY = GeneratorConfig(n_cards=120, n_merchants=50, days=40, seed=5, fraud_multiplier=6.0)


def test_same_seed_gives_identical_output():
    pd.testing.assert_frame_equal(generate_transactions(TINY), generate_transactions(TINY))


def test_different_seed_gives_different_output():
    a = generate_transactions(TINY)
    b = generate_transactions(replace(TINY, seed=6))
    assert len(a) != len(b) or not a["amount"].equals(b["amount"])


def test_schema_order_and_ids(small_transactions):
    df = small_transactions
    assert list(df.columns) == COLUMNS
    assert df["transaction_id"].is_unique
    assert df["timestamp"].is_monotonic_increasing
    assert (df["amount"] > 0).all()
    assert set(df["channel"]) <= {"pos", "online"}
    assert set(df["is_fraud"].unique()) == {0, 1}
    assert ((df["fraud_pattern"] == "none") == (df["is_fraud"] == 0)).all()


def test_all_three_fraud_patterns_are_injected(small_transactions):
    patterns = set(small_transactions["fraud_pattern"]) - {"none"}
    assert patterns == {"card_testing", "account_takeover", "high_value"}


def test_card_testing_is_a_burst_of_small_amounts(small_transactions):
    ct = small_transactions[small_transactions["fraud_pattern"] == "card_testing"]
    probes = ct[ct["amount"] <= 3.0 * TINY.amount_scale]
    assert len(probes) > 0
    gaps = probes.groupby("card_id")["timestamp"].diff().dropna()
    assert (gaps <= pd.Timedelta(minutes=10)).mean() > 0.9


def test_account_takeover_uses_a_device_the_card_never_used(small_transactions):
    df = small_transactions
    ato = df[df["fraud_pattern"] == "account_takeover"]
    legit_devices = df[df["is_fraud"] == 0].groupby("card_id")["device_id"].agg(set)
    for row in ato.itertuples():
        assert row.device_id not in legit_devices.get(row.card_id, set())


def test_high_value_fraud_targets_high_value_categories(small_transactions):
    hv = small_transactions[small_transactions["fraud_pattern"] == "high_value"]
    assert set(hv["merchant_category"]) <= set(HIGH_VALUE_CATEGORIES)


def test_drift_knobs_keep_the_population_and_scale_amounts():
    base = generate_transactions(TINY)
    shifted = generate_transactions(replace(TINY, amount_scale=2.0, start="2026-06-01"))
    assert set(base["card_id"]) == set(shifted["card_id"])
    merchants = base.groupby("merchant_id")["merchant_category"].first()
    shifted_merchants = shifted.groupby("merchant_id")["merchant_category"].first()
    common = merchants.index.intersection(shifted_merchants.index)
    assert (merchants[common] == shifted_merchants[common]).all()
    assert shifted["timestamp"].min() >= pd.Timestamp("2026-06-01")
    ratio = shifted["amount"].median() / base["amount"].median()
    assert 1.6 < ratio < 2.4


def test_fraud_rate_is_per_year_not_per_period():
    short = generate_transactions(replace(TINY, days=30))
    long = generate_transactions(replace(TINY, days=90))
    assert abs(short["is_fraud"].mean() - long["is_fraud"].mean()) < 0.02
