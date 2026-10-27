# Model Card

## Model

- Type: multi-layer perceptron reconstruction model used as an autoencoder.
- Purpose: prioritize unusual manufacturing sensor records for human review.
- Baseline: Isolation Forest.
- Version: 0.1.0.

## Training data

Synthetic, correlated telemetry with injected bearing-failure, overheating, and pressure-leak patterns. No real people, customers, or employers are represented.

## Evaluation

Run `python -m crowe_anomaly.train`, then copy the final values from `artifacts/metrics.json` here:

- Precision:
- Recall:
- F1:
- False positives:
- False negatives:
- Recall/flags by anomaly type:

## Intended use

Learning, portfolio demonstration, and shadow-mode experimentation. The output supports review; it is not a maintenance or safety decision.

## Out-of-scope use

- Automatic equipment shutdown.
- Employee performance monitoring.
- Scoring personal, health, financial, or customer data.
- Deployment without site-specific validation and monitoring.

## Known limitations

- Synthetic anomalies are cleaner than many real-world failures.
- Reconstruction error can drift when normal operating conditions change.
- A global threshold may perform differently across machines and seasons.
- The project does not yet include authentication, encryption termination, or a production feature store.

## Monitoring plan

- Input schema failures and missingness.
- Feature distribution drift.
- Score distribution and alert rate.
- Confirmed-event recall and false-alert rate.
- Service availability and latency.
- Model version and rollback status.

