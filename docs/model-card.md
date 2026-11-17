# Model card: purchase-intent classifier

## Overview

Random forest classifier that estimates the probability an e-commerce session
ends in a purchase, using 17 session-level behavioural features.

- Version: 0.1.0
- Owner: Kamran Hashmi
- Type: binary classification, scikit-learn `RandomForestClassifier` in a preprocessing pipeline
- Artifact: `artifacts/model.joblib` (joblib), logged to MLflow when tracking is enabled

## Intended use

- Ranking sessions for remarketing or onsite personalisation experiments
- Demonstrating an end-to-end MLOps workflow: training, tracking, serving, deployment, monitoring
- Batch scoring of historical sessions for analysis

## Out of scope

- Individual credit, employment, housing or pricing decisions
- Identifying people. The dataset is session-level and contains no user identifiers
- Real-time bidding or fully automated discounts without human review

## Training data

UCI Online Shoppers Purchasing Intention dataset, 12,330 sessions from a single
retailer over one year, 15.5% positive class. See `data/README.md` for source,
license and the SHA-256 pinned in every training run.

## Evaluation

Stratified 80/20 split, random state 42. Metrics from the committed run:

| Metric | Value |
| --- | --- |
| ROC-AUC | 0.9266 |
| Average precision | 0.7274 |
| Accuracy | 0.8816 |
| Precision | 0.5962 |
| Recall | 0.7304 |
| F1 | 0.6565 |
| Best F1 | 0.6658 at threshold 0.55 |

The class imbalance means accuracy is not the headline metric. Recall and
average precision matter more when the cost of missing a likely buyer is high;
precision matters when outreach is expensive.

## Limitations

- Single retailer, one year of traffic. Seasonality and channel mix may not transfer.
- No temporal split, so the reported score is optimistic for future sessions.
- Features are behavioural aggregates, not customer identity or product catalogue data.
- Random forest probabilities are not calibrated. Run calibration before using them as
  expected values rather than ranking scores.

## Ethical considerations

Session-level behavioural data can indirectly reveal browsing patterns. Keep
predictions at the session level, avoid linking them to identities, retain data
only as long as needed and follow the retailer's consent and privacy policy.

## Monitoring and retraining

`intent-monitor` compares production feature distributions with the training
reference using per-feature mean shift for numeric columns and total variation
distance for categorical columns. A drift share of 30% or more raises the alert
flag. Retrain when drift is sustained, when F1 drops below the agreed floor, or
on a fixed cadence, and promote only after the CI evaluation passes.
