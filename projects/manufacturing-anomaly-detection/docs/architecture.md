# Architecture and Data Flow

## Components

1. `data.py` generates safe synthetic telemetry with three failure modes.
2. `schema.py` enforces the approved five-feature contract and rejects common PII fields.
3. `train.py` performs a chronological split and trains only on normal training rows.
4. `modeling.py` compares neural-network autoencoders and an Isolation Forest baseline.
5. `report.py` creates technical metrics and a business-facing decision brief.
6. `score.py` provides batch inference.
7. `service.py` provides `/health` and `/score` endpoints without logging payloads.
8. The Dockerfile trains in a builder stage and serves from a smaller non-root runtime stage.
9. Kubernetes manages two replicas with rolling updates, health probes, resource bounds, and a restricted security context.

## Training boundary

The scaler and autoencoder are fit only on normal rows from the first 60% of the timeline. Model architecture and threshold are selected using the next 20%. The final 20% is held out for one-time test reporting.

## Serving boundary

Only these fields cross the scoring boundary:

- `temperature_c`
- `vibration_mm_s`
- `pressure_bar`
- `motor_current_a`
- `throughput_units_h`

The service returns a score, threshold, model version, anomaly flag, and human-review flag.

