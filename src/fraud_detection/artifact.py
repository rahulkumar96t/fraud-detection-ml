"""The deployable model bundle: estimator + everything needed to use it consistently."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from fraud_detection.drift import BinProfile
from fraud_detection.features import FeatureConfig


@dataclass
class ModelArtifact:
    model: Any
    model_name: str
    model_version: str
    feature_names: list[str]
    feature_config: FeatureConfig
    threshold: float
    threshold_policy: str
    reference_profiles: dict[str, BinProfile]
    metrics: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""

    def score_frame(self, frame: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(frame[self.feature_names])[:, 1]

    def score_rows(self, rows: list[dict[str, float]]) -> np.ndarray:
        return self.score_frame(pd.DataFrame(rows, columns=self.feature_names))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> ModelArtifact:
        # joblib uses pickle: only load artefacts you produced yourself.
        artifact = joblib.load(path)
        if not isinstance(artifact, ModelArtifact):
            raise TypeError(f"{path} does not contain a ModelArtifact")
        return artifact


def model_version(model_name: str, params: dict[str, Any], data_fingerprint: str) -> str:
    """Deterministic version: same code config + same training data => same version string."""
    payload = json.dumps({"params": params, "data": data_fingerprint}, sort_keys=True, default=str)
    digest = hashlib.sha256(payload.encode()).hexdigest()[:10]
    short = {
        "hist_gradient_boosting": "hgb",
        "logistic_regression": "logreg",
        "spline_logistic_regression": "spline-logreg",
    }.get(model_name)
    return f"{short or model_name}-{digest}"


def data_fingerprint(frame: pd.DataFrame) -> str:
    hashed = pd.util.hash_pandas_object(frame, index=False).to_numpy()
    return hashlib.sha256(hashed.tobytes()).hexdigest()[:16]


def config_dict(cfg: Any) -> dict[str, Any]:
    return asdict(cfg)
