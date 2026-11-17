"""Data drift reporting for production monitoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET_COLUMN
from .data import load_dataset

DRIFT_SHARE_ALERT = 0.3


def _numeric_drift(reference: pd.Series, current: pd.Series) -> dict:
    ref_std = float(reference.std()) or 1.0
    shift = abs(float(current.mean()) - float(reference.mean())) / ref_std
    return {"statistic": round(shift, 4), "drifted": shift >= 0.5}


def _categorical_drift(reference: pd.Series, current: pd.Series) -> dict:
    ref_share = reference.value_counts(normalize=True)
    cur_share = current.value_counts(normalize=True)
    categories = ref_share.index.union(cur_share.index)
    distance = 0.5 * float(
        np.abs(
            ref_share.reindex(categories, fill_value=0)
            - cur_share.reindex(categories, fill_value=0)
        ).sum()
    )
    return {"statistic": round(distance, 4), "drifted": distance >= 0.1}


def compute_drift(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    """Feature-level drift with a summary share of drifted features."""
    features = {}
    for column in NUMERIC_FEATURES:
        if column in reference and column in current:
            features[column] = {
                "type": "numeric",
                **_numeric_drift(reference[column], current[column]),
            }
    for column in CATEGORICAL_FEATURES:
        if column in reference and column in current:
            features[column] = {
                "type": "categorical",
                **_categorical_drift(reference[column], current[column]),
            }

    drifted = [name for name, result in features.items() if result["drifted"]]
    share = len(drifted) / len(features) if features else 0.0
    return {
        "rows_reference": int(len(reference)),
        "rows_current": int(len(current)),
        "drifted_features": drifted,
        "drift_share": round(share, 4),
        "alert": share >= DRIFT_SHARE_ALERT,
        "features": features,
    }


def _evidently_report(reference: pd.DataFrame, current: pd.DataFrame, output: Path) -> bool:
    try:
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report
    except Exception:
        return False

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)
    report.save_html(str(output.with_suffix(".html")))
    return True


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compare production data with the training reference"
    )
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/drift.json"))
    args = parser.parse_args(argv)

    reference = load_dataset(args.reference)
    current = load_dataset(args.current)
    result = compute_drift(
        reference.drop(columns=[TARGET_COLUMN]), current.drop(columns=[TARGET_COLUMN])
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    html = _evidently_report(reference, current, args.output)
    print(
        f"drift share={result['drift_share']:.2f} alert={result['alert']} "
        f"drifted={len(result['drifted_features'])} report={args.output}"
        + (" (+ html)" if html else "")
    )


if __name__ == "__main__":
    main()
