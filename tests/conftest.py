from __future__ import annotations

import pandas as pd
import pytest

from fraud_detection.features import FeatureConfig
from fraud_detection.generator import GeneratorConfig, generate_transactions
from fraud_detection.split import SplitConfig
from fraud_detection.train import TrainConfig, TrainingResult, train

SMALL = GeneratorConfig(n_cards=300, n_merchants=80, days=60, seed=11, fraud_multiplier=5.0)
SMALL_FEATURES = FeatureConfig(label_delay_seconds=2 * 86_400, risk_lookback_seconds=7 * 86_400)
SMALL_TRAIN = TrainConfig(
    features=SMALL_FEATURES,
    split=SplitConfig(burn_in_days=9),
    permutation_repeats=2,
)


@pytest.fixture(scope="session")
def small_transactions() -> pd.DataFrame:
    return generate_transactions(SMALL)


@pytest.fixture(scope="session")
def trained(small_transactions: pd.DataFrame) -> TrainingResult:
    return train(small_transactions, SMALL_TRAIN)


@pytest.fixture(scope="session")
def small_feature_config() -> FeatureConfig:
    return SMALL_FEATURES


@pytest.fixture(scope="session")
def small_train_config() -> TrainConfig:
    return SMALL_TRAIN
