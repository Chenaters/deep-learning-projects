# Problem Statement

## User and decision

The primary user is a manufacturing reliability engineer. Every five minutes, the system scores five sensor readings and decides whether the observation should enter a human review queue.

The model does **not** decide whether to stop equipment. A qualified operator retains that decision.

## Business objective

Detect unusual combinations of temperature, vibration, pressure, current, and throughput early enough to investigate potential failures.

## Costs

- False negative: a failure pattern is missed, possibly increasing downtime or damage.
- False positive: an engineer spends time investigating healthy equipment and may lose trust in the alerting system.
- Privacy or compliance failure: an unapproved field is accepted or logged.
- Deployment failure: a bad model version becomes unavailable or cannot be rolled back.

## Initial success criteria

- Recall of at least 70% on the held-out synthetic test period.
- Report precision, F1, false positives, and false negatives alongside recall.
- No training on future rows or test labels.
- No personal data in training or scoring.
- Every alert requires human review.

## Assumptions to challenge

1. Synthetic failure patterns resemble real failures.
2. Five sensor features are sufficient.
3. The normal operating distribution is stable.
4. A global threshold works across equipment and seasons.
5. Review capacity is large enough for the resulting alert rate.

