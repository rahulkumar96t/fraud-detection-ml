from __future__ import annotations

from collections.abc import Iterator

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from fraud_detection.api import create_app


@pytest.fixture
def client(trained) -> Iterator[TestClient]:
    with TestClient(create_app(artifact=trained.artifact)) as c:
        yield c


def _tx(**overrides) -> dict:
    tx = {
        "transaction_id": "tx-1",
        "timestamp": "2026-03-01T12:00:00Z",
        "card_id": "card-1",
        "merchant_id": "m0001",
        "merchant_category": "grocery",
        "amount": 42.5,
        "country": "DE",
        "device_id": "dev-1",
        "channel": "pos",
    }
    tx.update(overrides)
    return tx


def test_health_reports_model_version(client, trained):
    body = client.get("/health").json()
    assert body == {
        "status": "ok",
        "model_name": trained.artifact.model_name,
        "model_version": trained.artifact.model_version,
        "cards_tracked": 0,
    }


def test_model_info(client, trained):
    body = client.get("/model").json()
    assert body["threshold"] == trained.artifact.threshold
    assert body["feature_names"] == trained.artifact.feature_names
    assert "pr_auc" in body["test_metrics"]


def test_score_returns_version_decision_and_no_features_by_default(client, trained):
    r = client.post("/score", json=_tx())
    assert r.status_code == 200
    body = r.json()
    assert body["model_version"] == trained.artifact.model_version
    assert 0.0 <= body["fraud_score"] <= 1.0
    expected = "review" if body["fraud_score"] >= body["threshold"] else "approve"
    assert body["decision"] == expected
    assert "features" not in body


def test_state_accumulates_across_requests(client):
    first = client.post("/score?explain=true", json=_tx()).json()["features"]
    assert first["card_tx_count_10m"] == 0
    for i in range(2, 6):  # a card-testing style burst
        body = client.post(
            "/score?explain=true",
            json=_tx(
                transaction_id=f"tx-{i}",
                timestamp=f"2026-03-01T12:0{i}:00Z",
                amount=1.0,
                device_id="dev-new",
                channel="online",
            ),
        ).json()
    assert body["features"]["card_tx_count_10m"] == 4
    assert body["features"]["device_change"] == 0  # same new device as the previous call
    assert client.get("/health").json()["cards_tracked"] == 1


def test_batch_keeps_request_order_but_updates_state_in_time_order(client):
    later = _tx(transaction_id="b-later", timestamp="2026-03-01T12:05:00Z")
    earlier = _tx(transaction_id="b-earlier", timestamp="2026-03-01T12:01:00Z")
    body = client.post("/score/batch?explain=true", json={"transactions": [later, earlier]}).json()
    ids = [r["transaction_id"] for r in body["results"]]
    assert ids == ["b-later", "b-earlier"]
    assert body["results"][0]["features"]["card_tx_count_10m"] == 1  # saw the earlier one first


@pytest.mark.parametrize(
    "bad",
    [
        {"amount": -5},
        {"amount": 0},
        {"country": "Germany"},
        {"channel": "fax"},
        {"card_id": ""},
        {"unexpected": "field"},
    ],
)
def test_invalid_payloads_are_rejected(client, bad):
    assert client.post("/score", json=_tx(**bad)).status_code == 422


def test_empty_batch_is_rejected(client):
    assert client.post("/score/batch", json={"transactions": []}).status_code == 422


def test_feedback_is_used_only_after_the_label_delay(client, trained):
    delay_days = trained.artifact.feature_config.label_delay_seconds / 86_400
    for i in range(30):
        r = client.post(
            "/feedback",
            json={
                "transaction_id": f"fb-{i}",
                "merchant_id": "m-risky",
                "merchant_category": "gift_cards",
                "timestamp": "2026-03-01T10:00:00Z",
                "is_fraud": True,
            },
        )
        assert r.status_code == 202

    def risk(ts: str) -> float:
        body = client.post(
            "/score?explain=true",
            json=_tx(
                transaction_id=f"q-{ts}",
                timestamp=ts,
                merchant_id="m-risky",
                merchant_category="gift_cards",
                card_id=f"card-{ts}",
            ),
        ).json()
        return body["features"]["merchant_risk"]

    assert delay_days < 4
    assert risk("2026-03-01T11:00:00Z") == pytest.approx(risk("2026-03-01T11:30:00Z"))
    assert risk("2026-03-05T11:00:00Z") > 10 * risk("2026-03-01T11:00:00Z")


def test_feedback_from_the_future_is_rejected(client):
    r = client.post(
        "/feedback",
        json={
            "transaction_id": "x",
            "merchant_id": "m1",
            "merchant_category": "grocery",
            "timestamp": "2999-01-01T00:00:00Z",
            "is_fraud": False,
        },
    )
    assert r.status_code == 422


def test_history_warm_start(tmp_path, trained, small_transactions):
    history = tmp_path / "history.parquet"
    small_transactions.to_parquet(history)
    app = create_app(artifact=trained.artifact, history_path=history)
    with TestClient(app) as warm:
        assert (
            warm.get("/health").json()["cards_tracked"] == small_transactions["card_id"].nunique()
        )
        card = small_transactions["card_id"].iloc[-1]
        last = small_transactions["timestamp"].max() + pd.Timedelta(minutes=1)
        body = warm.post(
            "/score?explain=true",
            json=_tx(card_id=card, timestamp=last.isoformat() + "Z"),
        ).json()
        assert body["features"]["card_history_capped"] > 0


def test_out_of_order_transaction_is_scored_not_crashed(client):
    assert client.post("/score", json=_tx(timestamp="2026-03-01T12:00:00Z")).status_code == 200
    late = client.post(
        "/score?explain=true", json=_tx(transaction_id="late", timestamp="2026-03-01T11:00:00Z")
    )
    assert late.status_code == 200
    assert late.json()["features"]["card_tx_count_10m"] == 0  # the newer one is not "before" it
