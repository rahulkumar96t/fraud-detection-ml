"""Model definitions: two logistic-regression baselines and a gradient-boosted model.

* ``logistic_regression`` - standardised features, linear in each feature.
* ``spline_logistic_regression`` - each feature expanded into a cubic B-spline basis first, i.e.
  an additive (GAM-style) model: still one inspectable shape function per feature, no
  interactions. It is the honest baseline: "large amount" is only suspicious in combination
  with other signals, and a straight line per feature cannot express much of that.
* ``hist_gradient_boosting`` - learns interactions directly.

Imbalance is handled with a moderate positive-class weight, not resampling. SMOTE-style
oversampling invents transactions between real ones, which breaks the per-card time structure the
features rely on; undersampling discards most legitimate behaviour. The weight is a training
hyper-parameter (``TrainConfig.positive_weight``); the operating threshold is chosen afterwards on
validation data, so the scores do not need to be calibrated probabilities.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler


def _class_weight(positive_weight: float) -> dict[int, float]:
    return {0: 1.0, 1: positive_weight}


def make_logistic_regression(seed: int, positive_weight: float):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            class_weight=_class_weight(positive_weight),
            max_iter=3_000,
            random_state=seed,
        ),
    )


def make_spline_logistic_regression(seed: int, positive_weight: float):
    return make_pipeline(
        SplineTransformer(n_knots=5, degree=3, knots="quantile", extrapolation="constant"),
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            class_weight=_class_weight(positive_weight),
            max_iter=3_000,
            random_state=seed,
        ),
    )


def make_hist_gradient_boosting(seed: int, positive_weight: float):
    return HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_iter=600,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        l2_regularization=1.0,
        class_weight=_class_weight(positive_weight),
        early_stopping=True,
        n_iter_no_change=30,
        scoring="loss",
        random_state=seed,
    )


MODEL_FACTORIES: dict[str, Callable[[int, float], Any]] = {
    "logistic_regression": make_logistic_regression,
    "spline_logistic_regression": make_spline_logistic_regression,
    "hist_gradient_boosting": make_hist_gradient_boosting,
}


def fit_model(
    name: str,
    seed: int,
    positive_weight: float,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_val: pd.DataFrame,
    y_val: np.ndarray,
):
    model = MODEL_FACTORIES[name](seed, positive_weight)
    if isinstance(model, HistGradientBoostingClassifier):
        # Early stopping on the *later* validation period instead of a random slice of training.
        model.fit(x_train, y_train, X_val=x_val, y_val=y_val)
    else:
        model.fit(x_train, y_train)
    return model


def score(model, x: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(x)[:, 1]
