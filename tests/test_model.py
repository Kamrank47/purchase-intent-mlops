from __future__ import annotations

import numpy as np

from intent_mlops.evaluate import best_f1_threshold, binary_metrics, evaluate_model


def test_pipeline_predicts_probabilities(trained_pipeline):
    pipeline, X_test, _ = trained_pipeline
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    assert probabilities.shape[0] == len(X_test)
    assert np.all((probabilities >= 0) & (probabilities <= 1))


def test_metrics_have_expected_keys(trained_pipeline):
    pipeline, X_test, y_test = trained_pipeline
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    metrics = binary_metrics(y_test, probabilities)
    for key in ("roc_auc", "f1", "precision", "recall", "accuracy", "log_loss"):
        assert 0 <= metrics[key] <= 1
    assert len(metrics["confusion_matrix"]) == 2


def test_best_threshold_is_within_sweep(trained_pipeline):
    pipeline, X_test, y_test = trained_pipeline
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    threshold, score = best_f1_threshold(y_test, probabilities)
    assert 0.1 <= threshold <= 0.9
    assert 0 <= score <= 1


def test_evaluate_model_includes_threshold_report(trained_pipeline):
    pipeline, X_test, y_test = trained_pipeline
    report = evaluate_model(pipeline, X_test, y_test)
    assert "best_threshold_metrics" in report
    assert report["best_f1"] >= report["f1"]


def test_pipeline_handles_unseen_category(trained_pipeline):
    pipeline, X_test, _ = trained_pipeline
    unseen = X_test.head(5).copy()
    unseen["month"] = "Dec"
    unseen["visitor_type"] = "Other"
    assert pipeline.predict_proba(unseen).shape == (5, 2)
