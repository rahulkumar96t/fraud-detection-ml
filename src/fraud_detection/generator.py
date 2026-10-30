"""Seeded synthetic card-transaction generator.

The generator produces a transaction log with legitimate behaviour that is *not* trivially
separable from fraud (travel, second devices, commuters with many micro-payments, occasional
large purchases) and three injected fraud patterns:

* ``card_testing``  - bursts of tiny online authorisations at a few weak-verification merchants,
  often followed by a cash-out at a high-value category.
* ``account_takeover`` - from some point on, a new device in a new country spends several
  multiples of the card's usual amount.
* ``high_value`` - one or two large night-time purchases at jewellery / electronics / gift cards.

Everything is driven by a single ``numpy.random.Generator`` so a seed fully determines the output.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

COUNTRIES = np.array(["DE", "FR", "CH", "IT", "ES", "NL", "GB", "US", "PL", "SE"])
COUNTRY_WEIGHTS = np.array([0.22, 0.15, 0.10, 0.10, 0.09, 0.08, 0.10, 0.08, 0.05, 0.03])
FRAUD_COUNTRIES = np.array(["RU", "NG", "BR", "VN", "RO", "US", "CN"])

# category -> (typical amount, share of merchants, online share)
CATEGORIES: dict[str, tuple[float, float, float]] = {
    "grocery": (38.0, 0.20, 0.05),
    "restaurant": (32.0, 0.16, 0.02),
    "fuel": (55.0, 0.08, 0.0),
    "pharmacy": (24.0, 0.06, 0.10),
    "transit": (3.5, 0.04, 0.50),
    "apparel": (75.0, 0.10, 0.45),
    "online_retail": (60.0, 0.10, 1.0),
    "digital_goods": (12.0, 0.06, 1.0),
    "travel": (320.0, 0.05, 0.80),
    "electronics": (260.0, 0.07, 0.55),
    "jewelry": (420.0, 0.03, 0.30),
    "gift_cards": (90.0, 0.04, 0.70),
}
HIGH_VALUE_CATEGORIES = ("electronics", "jewelry", "gift_cards")
TESTING_CATEGORIES = ("digital_goods", "online_retail")

COLUMNS = [
    "transaction_id",
    "timestamp",
    "card_id",
    "merchant_id",
    "merchant_category",
    "amount",
    "country",
    "device_id",
    "channel",
    "is_fraud",
    "fraud_pattern",
]

DAY = 86_400
HOUR = 3_600


@dataclass(frozen=True)
class GeneratorConfig:
    """Knobs for the synthetic population. Fraud rates are incidents per card per year.

    Cards and merchants are drawn before any transaction, so two configs that differ only in
    ``start``, ``amount_scale`` or ``fraud_multiplier`` share the same population - handy for
    simulating drift on "the same bank, a later month".
    """

    n_cards: int = 2_000
    n_merchants: int = 400
    days: int = 150
    start: str = "2026-01-01"
    seed: int = 42
    card_testing_rate: float = 0.09
    account_takeover_rate: float = 0.09
    high_value_rate: float = 0.15
    commuter_share: float = 0.06
    gamer_share: float = 0.08
    travel_probability: float = 0.25
    amount_scale: float = 1.0
    fraud_multiplier: float = 1.0


@dataclass
class _Merchants:
    ids: np.ndarray
    category: np.ndarray
    country: np.ndarray
    online: np.ndarray
    popularity: np.ndarray
    weak_verification: np.ndarray


@dataclass
class _Card:
    card_id: str
    home: str
    spend: float
    rate: float
    devices: list[str]
    favourites: np.ndarray
    commuter: bool
    gamer: bool


def _make_merchants(cfg: GeneratorConfig, rng: np.random.Generator) -> _Merchants:
    names = list(CATEGORIES)
    shares = np.array([CATEGORIES[c][1] for c in names])
    category = rng.choice(names, size=cfg.n_merchants, p=shares / shares.sum())
    # Make sure every category exists so the fraud patterns always have a target.
    category[: len(names)] = names
    online_share = np.array([CATEGORIES[c][2] for c in category])
    online = rng.random(cfg.n_merchants) < online_share
    country = rng.choice(COUNTRIES, size=cfg.n_merchants, p=COUNTRY_WEIGHTS)
    popularity = rng.pareto(1.5, size=cfg.n_merchants) + 1.0
    weak = online & np.isin(category, TESTING_CATEGORIES) & (rng.random(cfg.n_merchants) < 0.35)
    ids = np.array([f"m{i:04d}" for i in range(cfg.n_merchants)])
    return _Merchants(ids, category, country, online, popularity, weak)


def _make_cards(cfg: GeneratorConfig, m: _Merchants, rng: np.random.Generator) -> list[_Card]:
    cards = []
    for i in range(cfg.n_cards):
        home = str(rng.choice(COUNTRIES, p=COUNTRY_WEIGHTS))
        n_devices = 1 + int(rng.random() < 0.45)
        devices = [f"d{i:05d}-{k}" for k in range(n_devices)]
        local = (m.country == home) | m.online
        weights = m.popularity * local
        favourites = rng.choice(
            len(m.ids), size=min(15, int(local.sum())), replace=False, p=weights / weights.sum()
        )
        cards.append(
            _Card(
                card_id=f"c{i:05d}",
                home=home,
                spend=float(rng.lognormal(0.0, 0.45)),
                rate=float(rng.gamma(2.0, 0.75)),
                devices=devices,
                favourites=favourites,
                commuter=bool(rng.random() < cfg.commuter_share),
                gamer=bool(rng.random() < cfg.gamer_share),
            )
        )
    return cards


def _hours(rng: np.random.Generator, n: int, night_bias: float = 0.0) -> np.ndarray:
    """Seconds into the day: daytime-peaked for legit traffic, shifted to night when biased."""
    day = np.clip(rng.normal(14.5, 3.8, size=n), 6.0, 23.99)
    night = rng.uniform(0.0, 5.99, size=n)
    use_night = rng.random(n) < (0.03 + night_bias)
    return (np.where(use_night, night, day) * HOUR).astype(np.int64)


def _amounts(rng: np.random.Generator, category: np.ndarray, spend: float, scale: float):
    base = np.array([CATEGORIES[c][0] for c in category])
    return np.round(base * spend * scale * rng.lognormal(0.0, 0.55, size=len(category)), 2)


def _legit_for_card(card: _Card, cfg: GeneratorConfig, m: _Merchants, rng) -> list[dict]:
    n = int(rng.poisson(card.rate * cfg.days))
    if n == 0:
        return []
    day = rng.integers(0, cfg.days, size=n)
    ts = day * DAY + _hours(rng, n)
    use_fav = rng.random(n) < 0.8
    merchant = np.where(
        use_fav,
        rng.choice(card.favourites, size=n),
        rng.choice(len(m.ids), size=n, p=m.popularity / m.popularity.sum()),
    )
    category = m.category[merchant]
    amount = _amounts(rng, category, card.spend, cfg.amount_scale)
    # the occasional legit splurge - much more common at high-value merchants
    p_big = np.where(np.isin(category, HIGH_VALUE_CATEGORIES), 0.04, 0.004)
    big = rng.random(n) < p_big
    amount = np.where(big, np.round(amount * rng.uniform(2, 9, size=n), 2), amount)
    country = np.where(m.online[merchant], card.home, m.country[merchant])
    device = rng.choice(card.devices, size=n)
    new_device = rng.random(n) < 0.004  # a legit new phone / browser
    device = np.where(new_device, f"{card.devices[0]}-new{rng.integers(1_000)}", device)

    if rng.random() < cfg.travel_probability:  # a trip abroad, sometimes on a borrowed device
        trip_start = int(rng.integers(0, max(1, cfg.days - 10)))
        trip_len = int(rng.integers(3, 11))
        dest = str(rng.choice(np.setdiff1d(COUNTRIES, [card.home])))
        on_trip = (day >= trip_start) & (day < trip_start + trip_len)
        country = np.where(on_trip & (~m.online[merchant] | (rng.random(n) < 0.5)), dest, country)
        if rng.random() < 0.3:
            device = np.where(on_trip & m.online[merchant], f"{card.devices[0]}-trip", device)

    rows = [
        dict(ts=ts, merchant=merchant, amount=amount, country=country, device=device),
    ]
    if card.commuter:  # many micro-payments, a legit look-alike of card testing
        k = int(rng.poisson(2.5 * cfg.days))
        transit = np.flatnonzero(m.category == "transit")
        tm = rng.choice(transit, size=k)
        rows.append(
            dict(
                ts=rng.integers(0, cfg.days, size=k) * DAY + _hours(rng, k),
                merchant=tm,
                amount=np.round(rng.uniform(1.2, 4.8, size=k) * cfg.amount_scale, 2),
                country=np.where(m.online[tm], card.home, m.country[tm]),
                device=rng.choice(card.devices, size=k),
            )
        )
    if card.gamer:  # bursts of small in-app purchases, another card-testing look-alike
        digital = np.flatnonzero((m.category == "digital_goods") & ~m.weak_verification)
        for _ in range(int(rng.poisson(0.12 * cfg.days))):
            k = int(rng.integers(3, 9))
            t0 = int(rng.integers(0, cfg.days)) * DAY + int(_hours(rng, 1, night_bias=0.2)[0])
            gm = int(rng.choice(digital))
            rows.append(
                dict(
                    ts=t0 + np.cumsum(rng.integers(20, 300, size=k)),
                    merchant=np.full(k, gm),
                    amount=np.round(rng.choice([0.99, 1.99, 4.99, 9.99], size=k), 2),
                    country=np.full(k, card.home),
                    device=np.full(k, card.devices[-1]),
                )
            )
    out = []
    for r in rows:
        out.append(
            pd.DataFrame(
                {
                    "ts": r["ts"],
                    "card_id": card.card_id,
                    "merchant_idx": r["merchant"],
                    "amount": r["amount"],
                    "country": r["country"],
                    "device_id": r["device"],
                    "is_fraud": 0,
                    "fraud_pattern": "none",
                }
            )
        )
    return out


def _pick(rng, m: _Merchants, categories, mask=None) -> int:
    pool = np.isin(m.category, categories)
    if mask is not None and (pool & mask).any():
        pool &= mask
    idx = np.flatnonzero(pool)
    return int(rng.choice(idx, p=m.popularity[idx] / m.popularity[idx].sum()))


def _fraud_frame(rows: list[tuple], card: _Card, pattern: str) -> pd.DataFrame:
    ts, merchant, amount, country, device = zip(*rows, strict=True)
    return pd.DataFrame(
        {
            "ts": np.array(ts, dtype=np.int64),
            "card_id": card.card_id,
            "merchant_idx": np.array(merchant, dtype=np.int64),
            "amount": np.round(np.array(amount, dtype=float), 2),
            "country": list(country),
            "device_id": list(device),
            "is_fraud": 1,
            "fraud_pattern": pattern,
        }
    )


def _card_testing(card: _Card, cfg: GeneratorConfig, m: _Merchants, rng, k: int) -> pd.DataFrame:
    t = int(rng.integers(3 * DAY, cfg.days * DAY - DAY))
    device = f"fx{k:04d}"
    country = str(rng.choice(FRAUD_COUNTRIES)) if rng.random() < 0.7 else card.home
    targets = [_pick(rng, m, TESTING_CATEGORIES, m.weak_verification) for _ in range(3)]
    rows = []
    for _ in range(int(rng.integers(2, 14))):
        t += int(rng.integers(8, 400))
        amount = rng.uniform(0.5, 3.0) * cfg.amount_scale
        rows.append((t, int(rng.choice(targets)), amount, country, device))
    if rng.random() < 0.6:  # cash-out once the card is confirmed live
        for _ in range(int(rng.integers(1, 4))):
            t += int(rng.integers(10 * 60, 20 * HOUR))
            merchant = _pick(rng, m, HIGH_VALUE_CATEGORIES, m.online)
            amount = CATEGORIES[m.category[merchant]][0] * rng.uniform(2, 6) * cfg.amount_scale
            rows.append((t, merchant, amount, country, device))
    return _fraud_frame(rows, card, "card_testing")


def _account_takeover(card: _Card, cfg: GeneratorConfig, m: _Merchants, rng, k: int):
    t = int(rng.integers(5 * DAY, cfg.days * DAY - 2 * DAY))
    device = f"fa{k:04d}"
    # a quarter of takeovers come from a proxy in the card's own country
    local = rng.random() < 0.25
    foreign = np.setdiff1d(FRAUD_COUNTRIES, [card.home])
    country = card.home if local else str(rng.choice(foreign))
    rows = []
    for _ in range(int(rng.integers(2, 7))):
        t += int(rng.integers(5 * 60, 12 * HOUR))
        merchant = _pick(rng, m, (*HIGH_VALUE_CATEGORIES, "travel", "apparel"), m.online)
        amount = CATEGORIES[m.category[merchant]][0] * card.spend * rng.uniform(1.2, 6)
        rows.append((t, merchant, amount * cfg.amount_scale, country, device))
    return _fraud_frame(rows, card, "account_takeover")


def _high_value(card: _Card, cfg: GeneratorConfig, m: _Merchants, rng, k: int) -> pd.DataFrame:
    t = int(rng.integers(2 * DAY, cfg.days * DAY - DAY)) // DAY * DAY
    rows = []
    for _ in range(1 + int(rng.random() < 0.35)):
        t_i = t + int(_hours(rng, 1, night_bias=0.6)[0])
        merchant = _pick(rng, m, HIGH_VALUE_CATEGORIES)
        cloned = rng.random() < 0.5  # cloned card at a POS vs card-not-present
        device = card.devices[0] if cloned else f"fh{k:04d}"
        country = m.country[merchant] if not m.online[merchant] else card.home
        amount = CATEGORIES[m.category[merchant]][0] * card.spend * rng.uniform(1.5, 8)
        rows.append((t_i, merchant, amount * cfg.amount_scale, country, device))
        t += int(rng.integers(HOUR, 6 * HOUR))
    return _fraud_frame(rows, card, "high_value")


def generate_transactions(cfg: GeneratorConfig | None = None) -> pd.DataFrame:
    """Generate a deterministic transaction log for ``cfg`` (sorted by time, stable ids)."""
    cfg = cfg or GeneratorConfig()
    rng = np.random.default_rng(cfg.seed)
    merchants = _make_merchants(cfg, rng)
    cards = _make_cards(cfg, merchants, rng)

    frames: list[pd.DataFrame] = []
    for card in cards:
        frames.extend(_legit_for_card(card, cfg, merchants, rng))

    patterns = (
        (_card_testing, cfg.card_testing_rate),
        (_account_takeover, cfg.account_takeover_rate),
        (_high_value, cfg.high_value_rate),
    )
    for make, rate in patterns:
        n_incidents = round(rate * cfg.fraud_multiplier * cfg.n_cards * cfg.days / 365)
        victims = rng.integers(0, cfg.n_cards, size=n_incidents)
        for k, victim in enumerate(victims):
            frames.append(make(cards[int(victim)], cfg, merchants, rng, k))

    df = pd.concat(frames, ignore_index=True)
    df = df[(df["ts"] >= 0) & (df["ts"] < cfg.days * DAY)]
    df = df.sort_values(["ts", "card_id", "merchant_idx", "amount"], kind="stable")
    df = df.reset_index(drop=True)

    start = pd.Timestamp(cfg.start)
    idx = df["merchant_idx"].to_numpy()
    out = pd.DataFrame(
        {
            "transaction_id": [f"t{i:08d}" for i in range(len(df))],
            "timestamp": start + pd.to_timedelta(df["ts"].to_numpy(), unit="s"),
            "card_id": df["card_id"].to_numpy(),
            "merchant_id": merchants.ids[idx],
            "merchant_category": merchants.category[idx],
            "amount": df["amount"].to_numpy().clip(min=0.5),
            "country": df["country"].to_numpy(),
            "device_id": df["device_id"].to_numpy(),
            "channel": np.where(merchants.online[idx], "online", "pos"),
            "is_fraud": df["is_fraud"].to_numpy().astype(np.int8),
            "fraud_pattern": df["fraud_pattern"].to_numpy(),
        }
    )
    out["timestamp"] = out["timestamp"].astype("datetime64[s]")
    return out[COLUMNS]
