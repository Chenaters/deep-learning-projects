# Code Review Checklist

## Correctness

- [ ] The change has a clear user or business outcome.
- [ ] Train, validation, and test boundaries remain intact.
- [ ] Feature order and model version are preserved.
- [ ] Edge cases and failure messages are tested.

## Modeling

- [ ] The baseline remains visible.
- [ ] The selected metric matches the operational cost.
- [ ] Threshold changes include false-positive and false-negative impact.
- [ ] No test result was used to choose the model.

## Engineering

- [ ] Functions are focused, named clearly, and type-hinted.
- [ ] No secrets, credentials, personal data, generated models, or raw data are committed.
- [ ] Unit and integration tests pass.
- [ ] Container and deployment changes retain least privilege and health checks.

## Communication

- [ ] The pull request explains what changed, why, risk, evidence, and rollback.
- [ ] Technical language is translated for the stakeholder brief when relevant.
- [ ] Documentation is updated with behavior changes.

