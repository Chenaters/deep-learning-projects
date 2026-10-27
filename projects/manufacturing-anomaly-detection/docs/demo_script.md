# Five-Minute Demo Script

## 0:00-0:45 - Business problem

"A reliability engineer receives a review queue of unusual sensor records. The goal is early investigation, not automated shutdown. Missing a failure and creating alert fatigue have different costs, so I report recall and false alerts together."

## 0:45-1:30 - Data and controls

Show `data.py` and `schema.py`.

"The dataset is synthetic. I use a five-field allowlist, reject common PII fields, and avoid logging payloads. The split is chronological so the model never learns from future operating periods."

## 1:30-2:30 - Model selection

Show `metrics.json`.

"I train only on normal operating rows. Two neural architectures compete on validation F1, and the alert threshold comes from validation-normal reconstruction errors. I retain Isolation Forest as a baseline and use the test period only for final reporting."

## 2:30-3:30 - Software and testing

Run the unit tests and show the modules.

"The model is packaged with its scaler, threshold, feature order, and version. Batch scoring and the HTTP service use the same artifact. Tests cover reproducible data, chronological splitting, schema rejection, training, persistence, and scoring."

## 3:30-4:20 - Deployment

Show the Dockerfile and Kubernetes manifests.

"The multi-stage Docker build separates training from runtime. The service runs as non-root. Kubernetes provides two replicas, rolling updates, health probes, resource bounds, and a restricted security context. CI blocks a regression if recall falls below the agreed gate."

## 4:20-5:00 - Decision and next step

Show `stakeholder_summary.md`.

"I recommend shadow mode only. The next step is authorized de-identified telemetry, operator review of false alerts, an explicit miss-versus-inspection cost, drift monitoring, and a rollback plan."

## Interview questions to prepare

1. Why is this an autoencoder rather than a classifier?
2. Why fit the scaler only on normal training data?
3. What happens when normal operating conditions drift?
4. How would you tune the threshold with limited review capacity?
5. Why can a strong synthetic-data metric still be misleading?
6. What does the readiness probe protect during a rolling update?

