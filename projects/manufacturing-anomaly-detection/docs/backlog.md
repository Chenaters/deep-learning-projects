# Agile Backlog

## Story 1 - Establish a reproducible baseline

**As a** data scientist, **I want** a deterministic dataset and Isolation Forest baseline **so that** the neural model must demonstrate incremental value.

Acceptance criteria:

- Same seed produces identical data.
- Baseline metrics are stored in `metrics.json`.
- Train, validation, and test periods are documented.

## Story 2 - Train a neural anomaly detector

**As a** reliability engineer, **I want** a model that scores unusual sensor combinations **so that** I can prioritize investigations.

Acceptance criteria:

- At least two autoencoder architectures are compared on validation data.
- Threshold uses validation-normal scores only.
- Test recall, precision, F1, and confusion counts are reported.

## Story 3 - Package reusable scoring

**As an** application developer, **I want** batch and HTTP scoring interfaces **so that** other workflows can consume the model consistently.

Acceptance criteria:

- Both interfaces load the same versioned artifact.
- Invalid schemas return clear errors.
- Scoring does not log payloads.

## Story 4 - Add delivery controls

**As a** platform engineer, **I want** automated tests and safe container settings **so that** changes are reviewable and deployments are repeatable.

Acceptance criteria:

- CI runs on pull requests.
- A recall regression blocks the integration job.
- Container runs as non-root.
- Deployment defines probes, resources, and a rolling-update policy.

## Stretch stories

- Add threshold tuning based on an explicit cost matrix.
- Add feature-drift monitoring.
- Add authenticated scoring and structured audit events without payloads.
- Add an approved model registry and rollback command.
- Replace synthetic data with an authorized de-identified public dataset.

