"""Project configuration and shared constants."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path.cwd())).resolve()
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "online_shoppers_intention.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "artifacts" / "model.joblib"
DEFAULT_METRICS_PATH = PROJECT_ROOT / "artifacts" / "metrics.json"
DEFAULT_META_PATH = PROJECT_ROOT / "artifacts" / "model_meta.json"

TARGET_COLUMN = "revenue"
RANDOM_STATE = 42
TEST_SIZE = 0.2

CATEGORICAL_FEATURES = ["month", "visitor_type"]
NUMERIC_FEATURES = [
    "administrative",
    "administrative_duration",
    "informational",
    "informational_duration",
    "product_related",
    "product_related_duration",
    "bounce_rates",
    "exit_rates",
    "page_values",
    "special_day",
    "operating_systems",
    "browser",
    "region",
    "traffic_type",
    "weekend",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

COLUMN_RENAMES = {
    "Administrative": "administrative",
    "Administrative_Duration": "administrative_duration",
    "Informational": "informational",
    "Informational_Duration": "informational_duration",
    "ProductRelated": "product_related",
    "ProductRelated_Duration": "product_related_duration",
    "BounceRates": "bounce_rates",
    "ExitRates": "exit_rates",
    "PageValues": "page_values",
    "SpecialDay": "special_day",
    "Month": "month",
    "OperatingSystems": "operating_systems",
    "Browser": "browser",
    "Region": "region",
    "TrafficType": "traffic_type",
    "VisitorType": "visitor_type",
    "Weekend": "weekend",
    "Revenue": "revenue",
}

VALID_MONTHS = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
VALID_VISITOR_TYPES = ["Returning_Visitor", "New_Visitor", "Other"]


def model_path() -> Path:
    """Model artifact location, overridable for containers and tests."""
    return Path(os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH))


def data_path() -> Path:
    """Training dataset location."""
    return Path(os.getenv("DATA_PATH", DEFAULT_DATA_PATH))


def metrics_path() -> Path:
    """Metrics report location written after training."""
    return Path(os.getenv("METRICS_PATH", DEFAULT_METRICS_PATH))


def mlflow_tracking_uri() -> str:
    """MLflow tracking server, defaults to a local server for compose runs."""
    return os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")


def experiment_name() -> str:
    return os.getenv("MLFLOW_EXPERIMENT", "purchase-intent")


def registered_model_name() -> str | None:
    return os.getenv("MLFLOW_MODEL_NAME", "purchase-intent-classifier")
