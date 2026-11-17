"""Dataset loading, validation and splitting."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import (
    COLUMN_RENAMES,
    FEATURE_COLUMNS,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)


class DatasetError(ValueError):
    """Raised when the dataset does not match the expected schema."""


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.astype(int)
    normalized = series.astype(str).str.strip().str.upper()
    mapping = {"TRUE": 1, "FALSE": 0, "1": 1, "0": 0, "YES": 1, "NO": 0}
    mapped = normalized.map(mapping)
    if mapped.isna().any():
        unknown = sorted(set(normalized[mapped.isna()]))[:5]
        raise DatasetError(f"Unrecognized boolean values: {unknown}")
    return mapped.astype(int)


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load the raw CSV and normalize names and dtypes."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}. Run `make data` to fetch it.")

    frame = pd.read_csv(path)
    missing = set(COLUMN_RENAMES) - set(frame.columns)
    if missing:
        raise DatasetError(f"Missing expected columns: {sorted(missing)}")

    frame = frame.rename(columns=COLUMN_RENAMES)
    frame["weekend"] = _as_bool(frame["weekend"])
    frame[TARGET_COLUMN] = _as_bool(frame[TARGET_COLUMN])

    for column in FEATURE_COLUMNS:
        if frame[column].isna().any():
            raise DatasetError(f"Column {column!r} contains null values")

    frameworked = frame[[*FEATURE_COLUMNS, TARGET_COLUMN]].copy()
    for column in FEATURE_COLUMNS:
        if column not in ("month", "visitor_type"):
            frameworked[column] = pd.to_numeric(frameworked[column], errors="raise")

    return frameworked


def validate_dataset(frame: pd.DataFrame) -> None:
    """Validate a normalized frame before training."""
    missing = set([*FEATURE_COLUMNS, TARGET_COLUMN]) - set(frame.columns)
    if missing:
        raise DatasetError(f"Frame is missing columns: {sorted(missing)}")
    if frame.empty:
        raise DatasetError("Dataset is empty")
    target = set(frame[TARGET_COLUMN].unique())
    if not target.issubset({0, 1}):
        raise DatasetError(f"Target must be binary, found {sorted(target)}")
    if frame[TARGET_COLUMN].nunique() < 2:
        raise DatasetError("Target has a single class; cannot train a classifier")


def split_dataset(
    frame: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split."""
    validate_dataset(frame)
    X = frame[FEATURE_COLUMNS]
    y = frame[TARGET_COLUMN]
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def dataset_fingerprint(path: str | Path) -> str:
    """SHA-256 of the raw file, recorded with every training run."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
