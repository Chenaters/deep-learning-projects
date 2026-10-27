# Risk Register

| ID | Risk | Impact | Control in this project | Next production control | Owner |
|---|---|---|---|---|---|
| R1 | Future-data leakage | Inflated metrics | Chronological 60/20/20 split | Dataset lineage and review | Data scientist |
| R2 | Test-label tuning | Inflated metrics | Candidate and threshold selection use validation only | Locked test set and experiment registry | Data scientist |
| R3 | PII enters scoring | Privacy/compliance incident | Five-field allowlist; common PII names rejected; payloads not logged | Authentication, DLP, encryption, audit policy | Security lead |
| R4 | Alert fatigue | Operators ignore alerts | Precision, false positives, and flag rate reported | Capacity-based threshold and feedback loop | Product owner |
| R5 | Missed failure | Downtime or damage | Recall and false negatives reported; human process retained | Failure-cost analysis and incident review | Reliability lead |
| R6 | Distribution drift | Performance degrades | Model card defines drift monitoring | Automated drift alerts and retraining approval | ML engineer |
| R7 | Bad deployment | Service outage or regression | Readiness/liveness probes; rolling update; CI recall gate | Canary release and automated rollback | Platform engineer |
| R8 | Excess container privilege | Security compromise | Non-root image and restricted Kubernetes security context | Image scanning and admission policy | Platform engineer |
| R9 | Model artifact tampering | Incorrect scores | Version stored with artifact | Signed images, checksums, access controls | ML platform owner |

## Change-control exercise

For every threshold or feature change, record:

1. Request and business reason.
2. Before/after validation metrics.
3. Risk review and approver.
4. Model and image version.
5. Deployment plan, monitoring window, and rollback trigger.

