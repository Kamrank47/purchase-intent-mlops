from __future__ import annotations

import pytest

from intent_mlops.config import DEFAULT_DATA_PATH, FEATURE_COLUMNS
from intent_mlops.data import load_dataset, split_dataset
from intent_mlops.features import build_pipeline


@pytest.fixture(scope="session")
def dataset():
    return load_dataset(DEFAULT_DATA_PATH)


@pytest.fixture(scope="session")
def sample(dataset):
    return dataset.head(800).reset_index(drop=True)


@pytest.fixture(scope="session")
def trained_pipeline(sample):
    X_train, X_test, y_train, y_test = split_dataset(sample, test_size=0.25)
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    return pipeline, X_test, y_test


@pytest.fixture(scope="session")
def feature_names():
    return list(FEATURE_COLUMNS)
