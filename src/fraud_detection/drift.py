"""Population Stability Index (PSI) drift monitoring.

At training time we store a binned profile of every feature (and of the model score) on the
training period. Later batches are binned with the *same* edges and compared with

    PSI = sum_i (a_i - e_i) * ln(a_i / e_i)

where e_i / a_i are the reference / current bin shares. Common rule of thumb: < 0.10 stable,
0.10-0.25 moderate shift, > 0.25 significant shift. PSI says *that* a distribution moved, not
whether the model got worse; that needs labels, which arrive later.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

EPS = 1e-4


@dataclass(frozen=True)
class BinProfile:
    edges: np.ndarray  # inner bin edges; bins are (-inf, e0], (e0, e1], ... , (ek, inf)
    proportions: np.ndarray

    def shares(self, values: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(self.edges, np.asarray(values, dtype=float), side="left")
        counts = np.bincount(idx, minlength=len(self.edges) + 1)
        return counts / max(counts.sum(), 1)


def fit_profile(values: np.ndarray, n_bins: int = 10) -> BinProfile:
    values = np.asarray(values, dtype=float)
    uniq = np.unique(values)
    if len(uniq) <= n_bins:  # discrete feature: one bin per observed value
        edges = uniq[:-1]
    else:
        edges = np.unique(np.quantile(values, np.linspace(0, 1, n_bins + 1)[1:-1]))
    profile = BinProfile(edges=edges, proportions=np.zeros(len(edges) + 1))
    return BinProfile(edges=edges, proportions=profile.shares(values))


def psi(expected: np.ndarray, actual: np.ndarray) -> float:
    e = np.clip(expected, EPS, None)
    a = np.clip(actual, EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def status(value: float) -> str:
    if value < 0.10:
        return "stable"
    return "moderate" if value < 0.25 else "significant"


def drift_table(profiles: dict[str, BinProfile], current: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for name, profile in profiles.items():
        value = psi(profile.proportions, profile.shares(current[name]))
        rows.append({"feature": name, "psi": value, "status": status(value)})
    table = pd.DataFrame(rows).sort_values("psi", ascending=False, kind="stable")
    return table.reset_index(drop=True)


def render_markdown(table: pd.DataFrame, title: str, context: dict[str, str]) -> str:
    lines = [f"# {title}", ""]
    lines += [f"- **{k}**: {v}" for k, v in context.items()]
    counts = table["status"].value_counts()
    lines += [
        "",
        f"{counts.get('significant', 0)} significant, {counts.get('moderate', 0)} moderate, "
        f"{counts.get('stable', 0)} stable (PSI thresholds 0.10 / 0.25).",
        "",
        "| feature | PSI | status |",
        "|---|---:|---|",
    ]
    lines += [f"| {r.feature} | {r.psi:.4f} | {r.status} |" for r in table.itertuples()]
    return "\n".join(lines) + "\n"
