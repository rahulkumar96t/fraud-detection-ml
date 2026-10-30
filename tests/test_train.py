"""Smoke tests for the training pipeline on a small sample."""

from __future__ import annotations

import json

import numpy as np

from fraud_detection.artifact import ModelArtifact
from fraud_detection.features import FEATURE_NAMES
from fraud_detection.train import train, write_outputs


def test_training_produces_a_usable_artifact(trained):
    artifact, report = trained.artifact, trained.report
    assert artifact.model_name in report["models"]
    assert artifact.feature_names == FEATURE_NAMES
    assert artifact.model_version.split("-")[0] in {"hgb", "logreg", "spline"}
    assert 0 < artifact.threshold < 1
    assert set(artifact.reference_profiles) == {*FEATURE_NAMES, "model_score"}


def test_models_beat_the_base_rate(trained):
    for name, by_split in trained.report["models"].items():
        test = by_split["test"]
        # a random ranking has PR-AUC ~= the fraud rate
        assert test["pr_auc"] > 3 * test["fraud_rate"], name


def test_champion_is_the_best_on_validation(trained):
    models = trained.report["models"]
    best = max(models, key=lambda m: models[m]["validation"]["pr_auc"])
    assert trained.report["model_name"] == best


def test_cost_optimal_threshold_beats_both_trivial_policies(trained):
    op = trained.report["operating_points"]["cost_optimal"]["test"]
    base = trained.report["cost_baselines"]
    assert op["cost"] < base["no_model_cost"]
    assert op["cost"] < base["review_everything_cost"]


def test_version_is_deterministic(small_transactions, small_train_config, trained):
    again = train(small_transactions, small_train_config)
    assert again.artifact.model_version == trained.artifact.model_version
    other = train(small_transactions.iloc[:-50], small_train_config)
    assert other.artifact.model_version != trained.artifact.model_version


def test_outputs_round_trip(tmp_path, trained, small_transactions, small_feature_config):
    from fraud_detection.features import build_features

    model_path, metrics_path, card_path = write_outputs(
        trained, tmp_path / "artifacts", tmp_path / "reports"
    )
    loaded = ModelArtifact.load(model_path)
    frame = build_features(small_transactions, small_feature_config).tail(200)
    np.testing.assert_allclose(loaded.score_frame(frame), trained.artifact.score_frame(frame))

    metrics = json.loads(metrics_path.read_text())
    assert metrics["model_version"] == trained.artifact.model_version

    card = card_path.read_text()
    champion_test = trained.report["models"][trained.artifact.model_name]["test"]
    assert trained.artifact.model_version in card
    assert f"{champion_test['pr_auc']:.3f}" in card  # the card quotes the measured numbers
    assert "synthetic" in card.lower()
