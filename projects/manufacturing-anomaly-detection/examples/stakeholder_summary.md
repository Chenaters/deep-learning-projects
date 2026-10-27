# Manufacturing Anomaly Pilot - Stakeholder Brief

## Decision

Proceed with a limited shadow-mode pilot of `autoencoder_24x8x24`. Do not automate maintenance shutdowns from this model.

## Business objective

Identify unusual equipment behavior early enough for an operator to investigate, while keeping false alerts low enough that the system remains usable.

## Test results

- Autoencoder precision: 80.0%
- Autoencoder recall: 100.0%
- Autoencoder F1: 88.9%
- False positives: 23
- Missed anomalies: 0
- Isolation Forest baseline F1: 3.6%

## Controls and caveats

- The data are synthetic and demonstrate workflow quality, not production performance.
- The service accepts only five approved numeric sensor fields and rejects common PII fields.
- Alerts require human review; the score is not a safety decision.
- Production use would require site-specific validation, drift monitoring, access controls, and a rollback plan.

## Next experiment

Run a four-week shadow test on de-identified historical telemetry, review false alerts with an engineer, and set the threshold from the cost of misses versus unnecessary inspections.
