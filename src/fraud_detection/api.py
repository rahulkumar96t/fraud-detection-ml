"""FastAPI scoring service.

Run with ``fraud-ml serve`` or ``uvicorn fraud_detection.api:create_app --factory``.
The model artefact path comes from ``FRAUD_MODEL_PATH`` (default ``artifacts/model.joblib``);
``FRAUD_HISTORY_PATH`` optionally points at a transaction log (parquet) that is replayed into the
feature store at start-up, so known cards do not start cold.
"""

from __future__ import annotations

import os
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal

import pandas as pd
from fastapi import FastAPI, Query, Request, status
from pydantic import BaseModel, ConfigDict, Field, field_validator

from fraud_detection.artifact import ModelArtifact
from fraud_detection.online import OnlineFeatureStore, Transaction

DEFAULT_MODEL_PATH = "artifacts/model.joblib"

Identifier = Annotated[str, Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.:-]+$")]


def _to_epoch(value: datetime) -> int:
    if value.tzinfo is None:  # naive timestamps are interpreted as UTC
        value = value.replace(tzinfo=UTC)
    return int(value.timestamp())


class TransactionIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transaction_id: Identifier
    timestamp: datetime
    card_id: Identifier
    merchant_id: Identifier
    merchant_category: Identifier
    amount: float = Field(gt=0, le=1_000_000)
    country: str = Field(pattern=r"^[A-Z]{2}$", description="ISO 3166-1 alpha-2")
    device_id: Identifier
    channel: Literal["pos", "online"]

    def to_transaction(self) -> Transaction:
        return Transaction(
            transaction_id=self.transaction_id,
            timestamp=_to_epoch(self.timestamp),
            card_id=self.card_id,
            merchant_id=self.merchant_id,
            merchant_category=self.merchant_category,
            amount=self.amount,
            country=self.country,
            device_id=self.device_id,
            channel=self.channel,
        )


class ScoreOut(BaseModel):
    transaction_id: str
    fraud_score: float = Field(description="Model score in [0, 1]; not a calibrated probability")
    decision: Literal["approve", "review"]
    threshold: float
    model_name: str
    model_version: str
    features: dict[str, float] | None = None


class BatchIn(BaseModel):
    transactions: list[TransactionIn] = Field(min_length=1, max_length=1_000)


class BatchOut(BaseModel):
    model_version: str
    results: list[ScoreOut]


class FeedbackIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transaction_id: Identifier
    merchant_id: Identifier
    merchant_category: Identifier
    timestamp: datetime = Field(description="When the labelled transaction happened")
    is_fraud: bool

    @field_validator("timestamp")
    @classmethod
    def _not_in_future(cls, value: datetime) -> datetime:
        aware = value if value.tzinfo else value.replace(tzinfo=UTC)
        if aware > datetime.now(UTC):
            raise ValueError("label timestamp cannot be in the future")
        return value


class FeedbackOut(BaseModel):
    accepted: bool
    transaction_id: str


class HealthOut(BaseModel):
    status: Literal["ok"]
    model_name: str
    model_version: str
    cards_tracked: int


class ModelInfoOut(BaseModel):
    model_name: str
    model_version: str
    threshold: float
    threshold_policy: str
    created_at: str
    feature_names: list[str]
    test_metrics: dict[str, float]


class ScoringService:
    """Thread-safe wrapper around the artefact and the online feature store."""

    def __init__(self, artifact: ModelArtifact) -> None:
        self.artifact = artifact
        self.store = OnlineFeatureStore(artifact.feature_config)
        self._lock = threading.Lock()

    def score(self, items: list[TransactionIn], explain: bool) -> list[ScoreOut]:
        txs = [item.to_transaction() for item in items]
        with self._lock:  # feature computation + state update must be atomic per transaction
            rows = [self.store.process(tx) for tx in txs]
        scores = self.artifact.score_rows(rows)
        a = self.artifact
        return [
            ScoreOut(
                transaction_id=tx.transaction_id,
                fraud_score=float(s),
                decision="review" if s >= a.threshold else "approve",
                threshold=a.threshold,
                model_name=a.model_name,
                model_version=a.model_version,
                features=row if explain else None,
            )
            for tx, row, s in zip(txs, rows, scores, strict=True)
        ]

    def feedback(self, item: FeedbackIn) -> None:
        with self._lock:
            self.store.record_label(
                item.merchant_id, item.merchant_category, _to_epoch(item.timestamp), item.is_fraud
            )


def create_app(
    model_path: str | os.PathLike[str] | None = None,
    artifact: ModelArtifact | None = None,
    history_path: str | os.PathLike[str] | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        loaded = artifact or ModelArtifact.load(
            Path(model_path or os.environ.get("FRAUD_MODEL_PATH", DEFAULT_MODEL_PATH))
        )
        service = ScoringService(loaded)
        history = history_path or os.environ.get("FRAUD_HISTORY_PATH")
        if history:
            service.store.replay(pd.read_parquet(history))
        app.state.service = service
        yield
        del app.state.service

    app = FastAPI(
        title="Fraud scoring service",
        version="0.1.0",
        summary="Scores card transactions with online, leakage-safe features.",
        lifespan=lifespan,
    )

    def service(request: Request) -> ScoringService:
        return request.app.state.service

    @app.get("/health", response_model=HealthOut)
    def health(request: Request) -> HealthOut:
        svc = service(request)
        return HealthOut(
            status="ok",
            model_name=svc.artifact.model_name,
            model_version=svc.artifact.model_version,
            cards_tracked=svc.store.n_cards,
        )

    @app.get("/model", response_model=ModelInfoOut)
    def model_info(request: Request) -> ModelInfoOut:
        a = service(request).artifact
        return ModelInfoOut(
            model_name=a.model_name,
            model_version=a.model_version,
            threshold=a.threshold,
            threshold_policy=a.threshold_policy,
            created_at=a.created_at,
            feature_names=a.feature_names,
            test_metrics=a.metrics.get("test", {}),
        )

    @app.post("/score", response_model=ScoreOut, response_model_exclude_none=True)
    def score_one(
        request: Request,
        tx: TransactionIn,
        explain: Annotated[bool, Query(description="Include computed features")] = False,
    ) -> ScoreOut:
        return service(request).score([tx], explain)[0]

    @app.post("/score/batch", response_model=BatchOut, response_model_exclude_none=True)
    def score_batch(request: Request, batch: BatchIn, explain: bool = False) -> BatchOut:
        svc = service(request)
        items = batch.transactions
        # Update card state in time order, answer in request order.
        order = sorted(range(len(items)), key=lambda i: _to_epoch(items[i].timestamp))
        scored = svc.score([items[i] for i in order], explain)
        results: list[ScoreOut] = [None] * len(items)  # type: ignore[list-item]
        for position, result in zip(order, scored, strict=True):
            results[position] = result
        return BatchOut(model_version=svc.artifact.model_version, results=results)

    @app.post("/feedback", response_model=FeedbackOut, status_code=status.HTTP_202_ACCEPTED)
    def feedback(request: Request, item: FeedbackIn) -> FeedbackOut:
        service(request).feedback(item)
        return FeedbackOut(accepted=True, transaction_id=item.transaction_id)

    return app
