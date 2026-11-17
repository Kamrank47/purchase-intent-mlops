from __future__ import annotations

from intent_mlops.config import TARGET_COLUMN
from intent_mlops.monitor import compute_drift


def test_identical_frames_show_no_drift(dataset):
    features = dataset.drop(columns=[TARGET_COLUMN]).head(500)
    result = compute_drift(features, features.copy())
    assert result["drift_share"] == 0.0
    assert result["alert"] is False
    assert result["drifted_features"] == []


def test_numeric_shift_is_detected(dataset):
    reference = dataset.drop(columns=[TARGET_COLUMN]).head(500)
    current = reference.copy()
    current["administrative"] = current["administrative"] + 50
    result = compute_drift(reference, current)
    assert "administrative" in result["drifted_features"]
    assert result["features"]["administrative"]["drifted"] is True


def test_category_shift_is_detected(dataset):
    reference = dataset.drop(columns=[TARGET_COLUMN]).head(500)
    current = reference.copy()
    current["month"] = "Feb"
    result = compute_drift(reference, current)
    assert "month" in result["drifted_features"]


def test_report_counts_rows(dataset):
    features = dataset.drop(columns=[TARGET_COLUMN]).head(200)
    result = compute_drift(features, features.head(100))
    assert result["rows_reference"] == 200
    assert result["rows_current"] == 100
