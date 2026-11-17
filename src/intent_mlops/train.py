"""Train the purchase-intent classifier and record a reproducible run."""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import sklearn

from .config import (
    DEFAULT_DATA_PATH,
    DEFAULT_META_PATH,
    DEFAULT_METRICS_PATH,
    DEFAULT_MODEL_PATH,
    experiment_name,
    mlflow_tracking_uri,
    registered_model_name,
)
from .data import dataset_fingerprint, load_dataset, split_dataset
from .evaluate import evaluate_model
from .features import build_pipeline


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the purchase-intent model")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--model-out", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--metrics-out", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--meta-out", type=Path, default=DEFAULT_META_PATH)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--track",
        action="store_true",
        help="Log params, metrics and the model to MLflow (requires the tracking extra)",
    )
    parser.add_argument("--register-model", action="store_true")
    return parser.parse_args(argv)


def train(
    data_path: Path,
    model_out: Path,
    metrics_out: Path,
    meta_out: Path,
    test_size: float = 0.2,
    seed: int = 42,
) -> dict:
    frame = load_dataset(data_path)
    X_train, X_test, y_train, y_test = split_dataset(frame, test_size, seed)

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    metrics = evaluate_model(pipeline, X_test, y_test)

    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_out)

    trained_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    metadata = {
        "trained_at": trained_at,
        "dataset_sha256": dataset_fingerprint(data_path),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "feature_count": int(len(X_train.columns)),
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
    }
    report = {**metrics, **metadata}

    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(json.dumps(report, indent=2) + "\n")
    meta_out.write_text(json.dumps(metadata, indent=2) + "\n")

    return report


def _log_to_mlflow(
    pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    metrics: dict,
    seed: int,
    test_size: float,
    register: bool,
) -> None:
    import mlflow
    import mlflow.sklearn
    from mlflow.models import infer_signature

    mlflow.set_tracking_uri(mlflow_tracking_uri())
    mlflow.set_experiment(experiment_name())

    with mlflow.start_run(run_name=f"rf-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"):
        mlflow.log_params(
            {
                "model": "RandomForestClassifier",
                "n_estimators": 300,
                "max_depth": 14,
                "min_samples_leaf": 4,
                "seed": seed,
                "test_size": test_size,
            }
        )
        mlflow.log_metrics(
            {
                key: metrics[key]
                for key in (
                    "roc_auc",
                    "average_precision",
                    "accuracy",
                    "precision",
                    "recall",
                    "f1",
                    "log_loss",
                    "best_f1",
                )
            }
        )
        mlflow.set_tag("dataset_sha256", metrics["dataset_sha256"])
        signature = infer_signature(X_train, pipeline.predict_proba(X_train)[:, 1])
        model_kwargs = {
            "sk_model": pipeline,
            "signature": signature,
            "input_example": X_train.head(3),
            "registered_model_name": registered_model_name() if register else None,
        }
        # The tree types are produced by this training run, not loaded from an
        # untrusted source, so skops may serialise them.
        trusted_types = ["sklearn.tree._tree.Tree"]
        try:  # MLflow 3 renamed artifact_path to name and uses skops
            mlflow.sklearn.log_model(
                name="model", skops_trusted_types=trusted_types, **model_kwargs
            )
        except TypeError:  # MLflow 2 signature
            mlflow.sklearn.log_model(artifact_path="model", **model_kwargs)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = train(
        args.data,
        args.model_out,
        args.metrics_out,
        args.meta_out,
        test_size=args.test_size,
        seed=args.seed,
    )

    if args.track:
        frame = load_dataset(args.data)
        X_train, _, y_train, _ = split_dataset(frame, args.test_size, args.seed)
        pipeline = joblib.load(args.model_out)
        _log_to_mlflow(
            pipeline,
            X_train,
            y_train,
            report,
            args.seed,
            args.test_size,
            args.register_model,
        )

    print(
        "trained model | roc_auc={roc_auc:.4f} f1={f1:.4f} "
        "best_f1={best_f1:.4f} (threshold {best_f1_threshold:.2f})".format(**report)
    )
    print(f"model: {args.model_out}")
    print(f"metrics: {args.metrics_out}")


if __name__ == "__main__":
    main()
