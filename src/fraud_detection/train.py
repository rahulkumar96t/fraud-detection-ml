"""Training pipeline: features -> time split -> two models -> thresholds -> report + artefact."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from fraud_detection.artifact import ModelArtifact, config_dict, data_fingerprint, model_version
from fraud_detection.drift import fit_profile
from fraud_detection.features import FEATURE_NAMES, FeatureConfig, build_features
from fraud_detection.metrics import (
    cost_optimal_threshold,
    operating_point,
    ranking_metrics,
    threshold_for_precision,
)
from fraud_detection.models import fit_model, score
from fraud_detection.split import SplitConfig, time_split

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainConfig:
    features: FeatureConfig = field(default_factory=FeatureConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    models: tuple[str, ...] = (
        "logistic_regression",
        "spline_logistic_regression",
        "hist_gradient_boosting",
    )
    seed: int = 7
    positive_weight: float = 10.0
    review_cost: float = 5.0
    precision_floor: float = 0.8
    permutation_repeats: int = 5


@dataclass
class TrainingResult:
    artifact: ModelArtifact
    report: dict[str, Any]


def _xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    return frame[FEATURE_NAMES], frame["is_fraud"].to_numpy()


def train(transactions: pd.DataFrame, cfg: TrainConfig | None = None) -> TrainingResult:
    cfg = cfg or TrainConfig()
    features = build_features(transactions, cfg.features)
    split = time_split(features, cfg.split)
    x_tr, y_tr = _xy(split.train)
    x_va, y_va = _xy(split.validation)
    x_te, y_te = _xy(split.test)
    for name, y in (("train", y_tr), ("validation", y_va), ("test", y_te)):
        if y.sum() < 2:
            raise ValueError(f"{name} split has fewer than 2 fraud cases; generate more data")

    amt_va, amt_te = split.validation["amount"].to_numpy(), split.test["amount"].to_numpy()
    fitted: dict[str, Any] = {}
    model_reports: dict[str, Any] = {}
    for name in cfg.models:
        log.info("fitting %s on %d rows", name, len(x_tr))
        model = fit_model(name, cfg.seed, cfg.positive_weight, x_tr, y_tr, x_va, y_va)
        fitted[name] = model
        s_va, s_te = score(model, x_va), score(model, x_te)
        model_reports[name] = {
            "validation": ranking_metrics(y_va, s_va),
            "test": ranking_metrics(y_te, s_te),
        }

    champion = max(cfg.models, key=lambda m: model_reports[m]["validation"]["pr_auc"])
    model = fitted[champion]
    s_va, s_te = score(model, x_va), score(model, x_te)

    # Thresholds are chosen on validation and *then* applied to test.
    t_cost = cost_optimal_threshold(y_va, s_va, amt_va, cfg.review_cost)
    t_prec = threshold_for_precision(y_va, s_va, cfg.precision_floor)
    operating = {
        "cost_optimal": {
            "validation": operating_point(y_va, s_va, amt_va, t_cost, cfg.review_cost).to_dict(),
            "test": operating_point(y_te, s_te, amt_te, t_cost, cfg.review_cost).to_dict(),
        },
        f"precision_{int(cfg.precision_floor * 100)}": {
            "validation": operating_point(y_va, s_va, amt_va, t_prec, cfg.review_cost).to_dict(),
            "test": operating_point(y_te, s_te, amt_te, t_prec, cfg.review_cost).to_dict(),
        },
    }
    baselines = {
        "no_model_cost": float(amt_te[y_te == 1].sum()),
        "review_everything_cost": float(cfg.review_cost * len(y_te)),
    }

    importance = permutation_importance(
        model,
        x_va,
        y_va,
        scoring="average_precision",
        n_repeats=cfg.permutation_repeats,
        random_state=cfg.seed,
    )
    importances = sorted(
        (
            {"feature": f, "mean": float(m), "std": float(s)}
            for f, m, s in zip(
                FEATURE_NAMES, importance.importances_mean, importance.importances_std, strict=True
            )
        ),
        key=lambda r: -r["mean"],
    )

    profiles = {name: fit_profile(split.train[name].to_numpy()) for name in FEATURE_NAMES}
    profiles["model_score"] = fit_profile(score(model, x_tr))

    fingerprint = data_fingerprint(transactions)
    params = {
        "model_params": {k: repr(v) for k, v in sorted(model.get_params().items())},
        "features": config_dict(cfg.features),
        "split": config_dict(cfg.split),
        "seed": cfg.seed,
        "positive_weight": cfg.positive_weight,
    }
    version = model_version(champion, params, fingerprint)
    report: dict[str, Any] = {
        "model_name": champion,
        "model_version": version,
        "created_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "data": {
            "fingerprint": fingerprint,
            "transactions": len(transactions),
            "boundaries": split.boundaries,
            "rows": {
                "train": len(split.train),
                "validation": len(split.validation),
                "test": len(split.test),
            },
            "frauds": {
                "train": int(y_tr.sum()),
                "validation": int(y_va.sum()),
                "test": int(y_te.sum()),
            },
        },
        "config": {
            "features": config_dict(cfg.features),
            "split": config_dict(cfg.split),
            "review_cost": cfg.review_cost,
            "precision_floor": cfg.precision_floor,
            "positive_weight": cfg.positive_weight,
            "seed": cfg.seed,
        },
        "models": model_reports,
        "operating_points": operating,
        "cost_baselines": baselines,
        "permutation_importance": importances,
        "feature_names": FEATURE_NAMES,
    }
    artifact = ModelArtifact(
        model=model,
        model_name=champion,
        model_version=version,
        feature_names=list(FEATURE_NAMES),
        feature_config=cfg.features,
        threshold=t_cost,
        threshold_policy=f"cost-optimal on validation (review cost {cfg.review_cost:g})",
        reference_profiles=profiles,
        metrics={"test": model_reports[champion]["test"], "operating": operating},
        created_at=report["created_at"],
    )
    return TrainingResult(artifact=artifact, report=report)


def write_outputs(result: TrainingResult, artifacts_dir: Path, reports_dir: Path) -> list[Path]:
    from fraud_detection.model_card import render_model_card

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifacts_dir / "model.joblib"
    metrics_path = artifacts_dir / "metrics.json"
    card_path = reports_dir / "model_card.md"
    result.artifact.save(model_path)
    metrics_path.write_text(json.dumps(result.report, indent=2, default=float) + "\n")
    card_path.write_text(render_model_card(result.report))
    return [model_path, metrics_path, card_path]
