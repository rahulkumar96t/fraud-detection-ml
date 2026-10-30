"""Evaluation metrics for a rare-event, cost-asymmetric problem.

ROC-AUC is reported for completeness but the decision metrics are the ones a fraud team uses:
PR-AUC, recall at a precision floor, precision in the top-k alerts, and expected cost.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


@dataclass(frozen=True)
class OperatingPoint:
    threshold: float
    alerts: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    cost: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def pr_auc(y: np.ndarray, scores: np.ndarray) -> float:
    return float(average_precision_score(y, scores))


def recall_at_precision(y: np.ndarray, scores: np.ndarray, min_precision: float) -> float:
    """Best recall achievable on (y, scores) while keeping precision >= ``min_precision``."""
    precision, recall, _ = precision_recall_curve(y, scores)
    ok = precision >= min_precision
    return float(recall[ok].max()) if ok.any() else 0.0


def threshold_for_precision(y: np.ndarray, scores: np.ndarray, min_precision: float) -> float:
    """Lowest threshold whose precision on (y, scores) is >= ``min_precision``.

    Used on the validation set; the resulting threshold is then applied unchanged to test data,
    which is the honest way to report "recall at P% precision" for a deployed model.
    Returns ``inf`` (alert on nothing) when the precision floor is never reached.
    """
    precision, _, thresholds = precision_recall_curve(y, scores)
    ok = np.flatnonzero(precision[:-1] >= min_precision)
    return float(thresholds[ok[0]]) if ok.size else float("inf")


def precision_at_k(y: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Share of fraud among the ``k`` highest-scored transactions (an analyst's alert queue)."""
    k = min(k, len(scores))
    if k == 0:
        return 0.0
    top = np.argsort(-scores, kind="stable")[:k]
    return float(np.asarray(y)[top].mean())


def expected_cost(
    y: np.ndarray, scores: np.ndarray, amount: np.ndarray, threshold: float, review_cost: float
) -> float:
    """Every alert costs ``review_cost``; every missed fraud costs its transaction amount."""
    alert = scores >= threshold
    missed = (np.asarray(y) == 1) & ~alert
    return float(review_cost * alert.sum() + np.asarray(amount)[missed].sum())


def cost_optimal_threshold(
    y: np.ndarray, scores: np.ndarray, amount: np.ndarray, review_cost: float
) -> float:
    """Threshold minimising :func:`expected_cost`, searched over every distinct score."""
    y = np.asarray(y)
    order = np.argsort(-scores, kind="stable")
    s, fraud_amount = scores[order], np.where(y[order] == 1, np.asarray(amount)[order], 0.0)
    # cut after position i => alert on s[:i+1]; only cut at the end of a group of tied scores
    last_of_group = np.r_[s[1:] != s[:-1], True]
    alerts = np.arange(1, len(s) + 1)
    cost = review_cost * alerts + fraud_amount.sum() - np.cumsum(fraud_amount)
    cost, cut_scores = cost[last_of_group], s[last_of_group]
    no_alert_cost = fraud_amount.sum()
    best = int(np.argmin(cost))
    return float(cut_scores[best]) if cost[best] < no_alert_cost else float("inf")


def operating_point(
    y: np.ndarray, scores: np.ndarray, amount: np.ndarray, threshold: float, review_cost: float
) -> OperatingPoint:
    y = np.asarray(y)
    alert = scores >= threshold
    tp = int((alert & (y == 1)).sum())
    fp = int((alert & (y == 0)).sum())
    fn = int((~alert & (y == 1)).sum())
    return OperatingPoint(
        threshold=float(threshold),
        alerts=int(alert.sum()),
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        precision=tp / (tp + fp) if tp + fp else 0.0,
        recall=tp / (tp + fn) if tp + fn else 0.0,
        cost=expected_cost(y, scores, amount, threshold, review_cost),
    )


def ranking_metrics(
    y: np.ndarray,
    scores: np.ndarray,
    precision_floors: tuple[float, ...] = (0.5, 0.8, 0.9),
    ks: tuple[int, ...] = (50, 100, 500),
) -> dict[str, float]:
    out = {
        "rows": float(len(y)),
        "fraud_rate": float(np.mean(y)),
        "pr_auc": pr_auc(y, scores),
        "roc_auc": float(roc_auc_score(y, scores)),
    }
    for p in precision_floors:
        out[f"recall_at_precision_{int(p * 100)}"] = recall_at_precision(y, scores, p)
    for k in ks:
        out[f"precision_at_{k}"] = precision_at_k(y, scores, k)
    return out
