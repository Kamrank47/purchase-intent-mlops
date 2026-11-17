from __future__ import annotations

import pandas as pd
import pytest

from intent_mlops.config import DEFAULT_DATA_PATH, FEATURE_COLUMNS, TARGET_COLUMN
from intent_mlops.data import (
    DatasetError,
    dataset_fingerprint,
    load_dataset,
    split_dataset,
    validate_dataset,
)


def test_dataset_shape_and_schema(dataset):
    assert len(dataset) == 12330
    assert set(FEATURE_COLUMNS + [TARGET_COLUMN]).issubset(dataset.columns)
    assert dataset[TARGET_COLUMN].isin([0, 1]).all()
    assert dataset["weekend"].isin([0, 1]).all()


def test_split_is_stratified(dataset):
    X_train, X_test, y_train, y_test = split_dataset(dataset)
    assert len(X_train) + len(X_test) == len(dataset)
    assert abs(y_train.mean() - y_test.mean()) < 0.02
    assert list(X_train.columns) == FEATURE_COLUMNS


def test_validation_rejects_missing_columns(dataset):
    broken = dataset.drop(columns=["month"])
    with pytest.raises(DatasetError):
        validate_dataset(broken)


def test_validation_rejects_multiclass_target(dataset):
    broken = dataset.copy()
    broken.loc[broken.index[0], TARGET_COLUMN] = 2
    with pytest.raises(DatasetError):
        validate_dataset(broken)


def test_fingerprint_is_stable(dataset, tmp_path):
    csv = tmp_path / "data.csv"
    dataset.to_csv(csv, index=False)
    assert dataset_fingerprint(csv) == dataset_fingerprint(csv)
    assert len(dataset_fingerprint(csv)) == 64


def test_loader_rejects_unknown_boolean(tmp_path):
    csv = tmp_path / "data.csv"
    raw = pd.read_csv(DEFAULT_DATA_PATH)
    raw["Revenue"] = raw["Revenue"].astype(object)
    raw.loc[0, "Revenue"] = "MAYBE"
    raw.to_csv(csv, index=False)
    with pytest.raises(DatasetError):
        load_dataset(csv)
