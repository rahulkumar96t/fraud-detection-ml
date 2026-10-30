"""Leakage-safe, vectorised feature engineering for the offline (training) path.

Every feature of a transaction at time ``t`` is computed only from

* transactions of the same card that precede it in *event order* (sorted by timestamp, then
  transaction id), and
* fraud labels of transactions whose label would already be known, i.e. whose timestamp is
  strictly earlier than ``t - label_delay``. Chargebacks arrive days after the purchase, so using
  "all labels before t" would still leak information that production does not have.

Features are also kept *stationary*: merchant / category risk use a trailing window of matured
labels and the history length is capped. All-time cumulative counters keep growing, so a model
trained on January would see values in April it never saw (and PSI flags them immediately).

The online scorer (:mod:`fraud_detection.online`) implements the same definitions incrementally;
``tests/test_online_parity.py`` checks that both paths agree.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

WINDOWS: dict[str, int] = {"10m": 600, "1h": 3_600, "24h": 86_400, "7d": 604_800}

FEATURE_NAMES: list[str] = [
    "log_amount",
    "amount_zscore",
    "log_amount_ratio",
    "card_history_capped",
    "log_seconds_since_last",
    *[f"card_tx_count_{w}" for w in WINDOWS],
    "log_card_amount_24h",
    "hour_sin",
    "hour_cos",
    "is_night",
    "is_online",
    "device_change",
    "new_device",
    "country_change",
    "new_country",
    "merchant_risk",
    "log_merchant_volume",
    "category_risk",
]


@dataclass(frozen=True)
class FeatureConfig:
    label_delay_seconds: int = 3 * 86_400
    prior_rate: float = 0.003
    prior_strength: float = 200.0
    zscore_std_floor: float = 1.0
    max_gap_seconds: int = 30 * 86_400
    risk_lookback_seconds: int = 28 * 86_400
    history_cap: int = 20

    @property
    def warmup_seconds(self) -> int:
        """Time until every feature is fully populated (label delay + risk lookback)."""
        return self.label_delay_seconds + self.risk_lookback_seconds


def epoch_seconds(ts: pd.Series) -> np.ndarray:
    """Timestamps (naive UTC) as int64 seconds since the Unix epoch."""
    return ts.astype("datetime64[s]").astype("int64").to_numpy()


def order_events(df: pd.DataFrame) -> pd.DataFrame:
    """Canonical event order: timestamp, then transaction id as a deterministic tie-breaker."""
    return df.sort_values(["timestamp", "transaction_id"], kind="stable").reset_index(drop=True)


def _group_sorted(codes: np.ndarray) -> np.ndarray:
    """Permutation that groups rows by ``codes`` while keeping event order inside a group."""
    return np.lexsort((np.arange(len(codes)), codes))


def _window_counts_and_sums(
    codes: np.ndarray, ts: np.ndarray, amount: np.ndarray, window: int
) -> tuple[np.ndarray, np.ndarray]:
    """For rows already grouped by ``codes``: prior rows of the same group within ``window``.

    A composite key ``code * span + ts`` is monotonic across the grouped array, so one
    ``searchsorted`` finds the first in-window row without crossing into the previous group.
    """
    rel = ts - ts.min()
    span = int(rel.max()) + window + 1
    key = codes.astype(np.int64) * span + rel
    lo = np.searchsorted(key, key - window, side="left")
    idx = np.arange(len(key))
    csum = np.concatenate([[0.0], np.cumsum(amount)])
    return idx - lo, csum[idx] - csum[lo]


def _smoothed_past_rate(
    codes: np.ndarray, ts: np.ndarray, labels: np.ndarray, cfg: FeatureConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Bayesian-smoothed fraud rate per group from labels that are known at time ``t``.

    A row j of the same group counts for row i when
    ``t_i - delay - lookback <= ts_j < t_i - delay``: its label has arrived (delay) and it is
    recent enough (lookback). Rows must be grouped by ``codes`` and time-ordered within a group.
    Returns the smoothed rate and the number of labelled rows it was computed from.
    """
    delay, lookback = cfg.label_delay_seconds, cfg.risk_lookback_seconds
    rel = ts - ts.min()
    span = int(rel.max()) + delay + lookback + 1  # keeps groups from overlapping
    base = codes.astype(np.int64) * span
    key = base + rel + delay  # key_j compared against t_i:  ts_j + delay  vs  t_i
    hi = np.searchsorted(key, base + rel, side="left")  # ts_j + delay < t_i
    lo = np.searchsorted(key, base + rel - lookback, side="left")  # ts_j + delay >= t_i - lookback
    cum_fraud = np.concatenate([[0], np.cumsum(labels)])
    n = hi - lo
    frauds = cum_fraud[hi] - cum_fraud[lo]
    rate = (frauds + cfg.prior_rate * cfg.prior_strength) / (n + cfg.prior_strength)
    return rate, n


def build_features(df: pd.DataFrame, cfg: FeatureConfig | None = None) -> pd.DataFrame:
    """Return one row per transaction (in event order) with identifiers, label and features."""
    cfg = cfg or FeatureConfig()
    events = order_events(df)
    n = len(events)
    ts = epoch_seconds(events["timestamp"])
    amount = events["amount"].to_numpy(dtype=float)
    labels = events["is_fraud"].to_numpy(dtype=np.int64)

    out = pd.DataFrame(
        {
            "transaction_id": events["transaction_id"].to_numpy(),
            "timestamp": events["timestamp"].to_numpy(),
            "card_id": events["card_id"].to_numpy(),
            "amount": amount,
            "is_fraud": labels.astype(np.int8),
        }
    )

    # ---- per-card history -------------------------------------------------------------------
    card_codes = pd.factorize(events["card_id"])[0]
    perm = _group_sorted(card_codes)
    c_codes, c_ts, c_amt = card_codes[perm], ts[perm], amount[perm]
    starts = np.searchsorted(c_codes, c_codes, side="left")
    n_prior = np.arange(n) - starts

    grouped_amt = pd.Series(c_amt).groupby(c_codes)
    s_prior = grouped_amt.cumsum().to_numpy() - c_amt
    ss_prior = pd.Series(c_amt * c_amt).groupby(c_codes).cumsum().to_numpy() - c_amt * c_amt
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = np.where(n_prior > 0, s_prior / np.maximum(n_prior, 1), np.nan)
        var = (ss_prior - n_prior * mean * mean) / np.maximum(n_prior - 1, 1)
        std = np.sqrt(np.clip(var, 0.0, None))
        z = np.where(n_prior >= 2, (c_amt - mean) / np.maximum(std, cfg.zscore_std_floor), 0.0)
        ratio = np.where(n_prior > 0, np.log(c_amt / mean), 0.0)

    prev_ts = np.where(n_prior > 0, np.roll(c_ts, 1), -1)
    gap = np.where(n_prior > 0, c_ts - prev_ts, cfg.max_gap_seconds)
    gap = np.minimum(gap, cfg.max_gap_seconds)

    feats: dict[str, np.ndarray] = {
        "amount_zscore": z,
        "log_amount_ratio": ratio,
        "card_history_capped": np.minimum(n_prior, cfg.history_cap).astype(float),
        "log_seconds_since_last": np.log1p(gap),
    }
    for name, window in WINDOWS.items():
        count, total = _window_counts_and_sums(c_codes, c_ts, c_amt, window)
        feats[f"card_tx_count_{name}"] = count.astype(float)
        if name == "24h":
            feats["log_card_amount_24h"] = np.log1p(total)

    for col, change_name, new_name in (
        ("device_id", "device_change", "new_device"),
        ("country", "country_change", "new_country"),
    ):
        values = events[col].to_numpy()[perm]
        prev = np.roll(values, 1)
        has_prior = n_prior > 0
        feats[change_name] = (has_prior & (values != prev)).astype(float)
        first_seen = (
            pd.DataFrame({"c": c_codes, "v": values}).groupby(["c", "v"], sort=False).cumcount()
            == 0
        ).to_numpy()
        feats[new_name] = (has_prior & first_seen).astype(float)

    inverse = np.empty(n, dtype=np.int64)
    inverse[perm] = np.arange(n)
    for name, values in feats.items():
        out[name] = values[inverse]

    # ---- transaction-level context -------------------------------------------------------------
    hour = (ts % 86_400) / 3_600.0
    out["log_amount"] = np.log1p(amount)
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["is_night"] = (hour < 6).astype(float)
    out["is_online"] = (events["channel"].to_numpy() == "online").astype(float)

    # ---- merchant / category risk from matured labels only --------------------------------
    for col, rate_name, count_name in (
        ("merchant_id", "merchant_risk", "log_merchant_volume"),
        ("merchant_category", "category_risk", None),
    ):
        codes = pd.factorize(events[col])[0]
        p = _group_sorted(codes)
        rate, labeled = _smoothed_past_rate(codes[p], ts[p], labels[p], cfg)
        inv = np.empty(n, dtype=np.int64)
        inv[p] = np.arange(n)
        out[rate_name] = rate[inv]
        if count_name:
            out[count_name] = np.log1p(labeled[inv].astype(float))

    return out[["transaction_id", "timestamp", "card_id", "amount", "is_fraud", *FEATURE_NAMES]]
