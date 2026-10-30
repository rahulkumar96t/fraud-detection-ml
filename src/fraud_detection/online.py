"""Incremental feature computation for the scoring service.

:class:`OnlineFeatureStore` keeps a bounded amount of state per card (a 7-day deque plus running
sums) and per merchant/category (timestamps of labelled transactions). It implements exactly the
definitions of :func:`fraud_detection.features.build_features`; a parity test replays a generated
log through both paths and compares the results.

The store expects transactions of one card to arrive in time order, which is what an
authorisation stream gives you. A late (out-of-order) transaction is still scored: its time gap is
clamped to zero and window counts only look at ``[t - window, t]``, but the running amount
statistics already include the newer transactions.

The store is in-memory and single-process by design; see the README for what a production
version would need (a shared low-latency store, TTLs, persistence).
"""

from __future__ import annotations

import math
from bisect import bisect_left, insort
from collections import defaultdict, deque
from dataclasses import dataclass, field

import pandas as pd

from fraud_detection.features import WINDOWS, FeatureConfig, epoch_seconds, order_events

MAX_WINDOW = max(WINDOWS.values())


@dataclass(frozen=True)
class Transaction:
    transaction_id: str
    timestamp: int  # Unix seconds, UTC
    card_id: str
    merchant_id: str
    merchant_category: str
    amount: float
    country: str
    device_id: str
    channel: str


@dataclass
class _CardState:
    recent: deque[tuple[int, float]] = field(default_factory=deque)
    n: int = 0
    s: float = 0.0
    ss: float = 0.0
    last_ts: int | None = None
    last_device: str | None = None
    last_country: str | None = None
    devices: set[str] = field(default_factory=set)
    countries: set[str] = field(default_factory=set)


@dataclass
class _LabelIndex:
    """Sorted timestamps of labelled transactions (all, and fraud only) for one key."""

    all_ts: list[int] = field(default_factory=list)
    fraud_ts: list[int] = field(default_factory=list)

    def add(self, ts: int, is_fraud: bool) -> None:
        insort(self.all_ts, ts)
        if is_fraud:
            insort(self.fraud_ts, ts)

    def known_between(self, lo: int, hi: int) -> tuple[int, int]:
        """Labelled / fraud counts with ``lo <= ts < hi``."""
        n = bisect_left(self.all_ts, hi) - bisect_left(self.all_ts, lo)
        frauds = bisect_left(self.fraud_ts, hi) - bisect_left(self.fraud_ts, lo)
        return n, frauds


class OnlineFeatureStore:
    def __init__(self, cfg: FeatureConfig | None = None) -> None:
        self.cfg = cfg or FeatureConfig()
        self._cards: dict[str, _CardState] = defaultdict(_CardState)
        self._merchants: dict[str, _LabelIndex] = defaultdict(_LabelIndex)
        self._categories: dict[str, _LabelIndex] = defaultdict(_LabelIndex)

    # -- reads ---------------------------------------------------------------------------------
    def compute(self, tx: Transaction) -> dict[str, float]:
        """Features for ``tx`` from current state; does not mutate the store."""
        cfg = self.cfg
        card = self._cards.get(tx.card_id) or _CardState()
        n = card.n
        mean = card.s / n if n else math.nan
        if n >= 2:
            var = (card.ss - n * mean * mean) / (n - 1)
            std = math.sqrt(max(var, 0.0))
            z = (tx.amount - mean) / max(std, cfg.zscore_std_floor)
        else:
            z = 0.0
        gap = cfg.max_gap_seconds if card.last_ts is None else max(tx.timestamp - card.last_ts, 0)
        hour = (tx.timestamp % 86_400) / 3_600.0

        feats: dict[str, float] = {
            "log_amount": math.log1p(tx.amount),
            "amount_zscore": z,
            "log_amount_ratio": math.log(tx.amount / mean) if n else 0.0,
            "card_history_capped": float(min(n, cfg.history_cap)),
            "log_seconds_since_last": math.log1p(min(gap, cfg.max_gap_seconds)),
        }
        for name, window in WINDOWS.items():
            lo = tx.timestamp - window
            in_window = [a for t, a in card.recent if lo <= t <= tx.timestamp]
            feats[f"card_tx_count_{name}"] = float(len(in_window))
            if name == "24h":
                feats["log_card_amount_24h"] = math.log1p(sum(in_window))
        feats.update(
            hour_sin=math.sin(2 * math.pi * hour / 24),
            hour_cos=math.cos(2 * math.pi * hour / 24),
            is_night=float(hour < 6),
            is_online=float(tx.channel == "online"),
            device_change=float(n > 0 and tx.device_id != card.last_device),
            new_device=float(n > 0 and tx.device_id not in card.devices),
            country_change=float(n > 0 and tx.country != card.last_country),
            new_country=float(n > 0 and tx.country not in card.countries),
        )
        hi = tx.timestamp - cfg.label_delay_seconds
        lo = hi - cfg.risk_lookback_seconds
        m_n, m_f = self._known(self._merchants, tx.merchant_id, lo, hi)
        c_n, c_f = self._known(self._categories, tx.merchant_category, lo, hi)
        feats["merchant_risk"] = self._smooth(m_f, m_n)
        feats["log_merchant_volume"] = math.log1p(m_n)
        feats["category_risk"] = self._smooth(c_f, c_n)
        return feats

    # -- writes --------------------------------------------------------------------------------
    def observe(self, tx: Transaction) -> None:
        """Add ``tx`` to its card's history (call after scoring it)."""
        card = self._cards[tx.card_id]
        card.recent.append((tx.timestamp, tx.amount))
        while card.recent and card.recent[0][0] < tx.timestamp - MAX_WINDOW:
            card.recent.popleft()
        card.n += 1
        card.s += tx.amount
        card.ss += tx.amount * tx.amount
        card.last_ts = max(tx.timestamp, card.last_ts or tx.timestamp)
        card.last_device = tx.device_id
        card.last_country = tx.country
        card.devices.add(tx.device_id)
        card.countries.add(tx.country)

    def record_label(self, merchant_id: str, category: str, timestamp: int, is_fraud: bool) -> None:
        """Register a confirmed label for a transaction that happened at ``timestamp``."""
        self._merchants[merchant_id].add(timestamp, is_fraud)
        self._categories[category].add(timestamp, is_fraud)

    def process(self, tx: Transaction) -> dict[str, float]:
        feats = self.compute(tx)
        self.observe(tx)
        return feats

    def replay(self, transactions: pd.DataFrame, labels: bool = True) -> list[dict[str, float]]:
        """Feed a historical log through the store in event order (warm start, parity checks).

        With ``labels=True`` each transaction's label is recorded right after it is processed;
        the store's label-delay logic keeps it invisible until it would really have arrived.
        """
        events = order_events(transactions)
        seconds = epoch_seconds(events["timestamp"])
        rows = []
        for row, ts in zip(events.itertuples(index=False), seconds, strict=True):
            tx = Transaction(
                transaction_id=row.transaction_id,
                timestamp=int(ts),
                card_id=row.card_id,
                merchant_id=row.merchant_id,
                merchant_category=row.merchant_category,
                amount=float(row.amount),
                country=row.country,
                device_id=row.device_id,
                channel=row.channel,
            )
            rows.append(self.process(tx))
            if labels:
                self.record_label(
                    tx.merchant_id, tx.merchant_category, tx.timestamp, bool(row.is_fraud)
                )
        return rows

    @property
    def n_cards(self) -> int:
        return len(self._cards)

    # -- helpers -------------------------------------------------------------------------------
    @staticmethod
    def _known(index: dict[str, _LabelIndex], key: str, lo: int, hi: int) -> tuple[int, int]:
        entry = index.get(key)
        return entry.known_between(lo, hi) if entry else (0, 0)

    def _smooth(self, frauds: int, n: int) -> float:
        cfg = self.cfg
        return (frauds + cfg.prior_rate * cfg.prior_strength) / (n + cfg.prior_strength)
