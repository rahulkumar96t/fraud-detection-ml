"""Time-based train / validation / test split.

Random splits leak the future into training (a card's later transactions sit next to its
earlier ones) and over-state performance. We cut the timeline instead and drop a burn-in period
at the start, where per-card history features are not yet meaningful.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SplitConfig:
    train_fraction: float = 0.6
    validation_fraction: float = 0.2
    # >= label delay + risk lookback (31 days by default), so every training row has
    # fully populated merchant / category risk features.
    burn_in_days: int = 31

    def __post_init__(self) -> None:
        if not 0 < self.train_fraction < 1 or not 0 < self.validation_fraction < 1:
            raise ValueError("fractions must be in (0, 1)")
        if self.train_fraction + self.validation_fraction >= 1:
            raise ValueError("train + validation must leave room for a test period")


@dataclass(frozen=True)
class TimeSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    boundaries: dict[str, str]


def time_split(features: pd.DataFrame, cfg: SplitConfig | None = None) -> TimeSplit:
    cfg = cfg or SplitConfig()
    ts = features["timestamp"]
    start, end = ts.min(), ts.max()
    span = end - start
    burn_end = start + pd.Timedelta(days=cfg.burn_in_days)
    train_end = start + span * cfg.train_fraction
    val_end = start + span * (cfg.train_fraction + cfg.validation_fraction)
    if burn_end >= train_end:
        raise ValueError("burn-in period swallows the whole training window")

    def _slice(lo, hi, inclusive_hi: bool = False) -> pd.DataFrame:
        mask = (ts >= lo) & ((ts <= hi) if inclusive_hi else (ts < hi))
        return features.loc[mask].reset_index(drop=True)

    fmt = "%Y-%m-%d %H:%M"
    return TimeSplit(
        train=_slice(burn_end, train_end),
        validation=_slice(train_end, val_end),
        test=_slice(val_end, end, inclusive_hi=True),
        boundaries={
            "data_start": start.strftime(fmt),
            "train_start": burn_end.strftime(fmt),
            "validation_start": train_end.strftime(fmt),
            "test_start": val_end.strftime(fmt),
            "data_end": end.strftime(fmt),
        },
    )
