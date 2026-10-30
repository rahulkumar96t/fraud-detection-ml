"""The online store used by the API must reproduce the offline training features."""

from __future__ import annotations

import numpy as np

from fraud_detection.features import FEATURE_NAMES, build_features
from fraud_detection.online import OnlineFeatureStore, Transaction


def test_online_replay_matches_offline_features(small_transactions, small_feature_config):
    offline = build_features(small_transactions, small_feature_config)
    online_rows = OnlineFeatureStore(small_feature_config).replay(small_transactions)
    assert len(online_rows) == len(offline)
    for name in FEATURE_NAMES:
        online = np.array([r[name] for r in online_rows])
        np.testing.assert_allclose(
            online, offline[name].to_numpy(), rtol=1e-9, atol=1e-9, err_msg=name
        )


def test_compute_does_not_mutate_state():
    store = OnlineFeatureStore()
    tx = Transaction("t1", 1_000, "c1", "m1", "grocery", 12.5, "DE", "d1", "pos")
    first = store.compute(tx)
    assert store.compute(tx) == first
    assert store.n_cards == 0
    store.observe(tx)
    assert store.compute(tx)["card_tx_count_10m"] == 1.0
