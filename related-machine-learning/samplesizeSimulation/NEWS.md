# samplesizeSimulation 0.1.0

- Introduce a standard R package layout, explicit namespace, registered S3
  methods, runnable help examples, and a self-contained HTML vignette.
- Preserve the original study entry points through compatibility loaders.
- Validate simulation inputs and make unsupported parallel execution explicit.
- Sort sample sizes and handle missing or deteriorating efficiency references.
- Correct binary single-class metrics, zero-variance R-squared, Brier scores,
  and zero F1 scores; remove the mislabeled adjusted R-squared placeholder.
- Calculate manual AUC using ranks rather than a quadratic pairwise matrix.
- Preserve caller RNG state for seeded simulations and restore plot settings.
- Invalidate cross-validation repeats with failed folds and report unavailable
  metric counts in simulation aggregates and reusable tables.
- Preserve generated mixed-model group identifiers and fit random intercepts;
  use response-scale predictions for generalized mixed models.
- Reject ordered validation with row bootstrap or synthetic resampling.
- Version cache keys for the revised methods and reject duplicate cache rows.
- Correct ridge/lasso prediction matrices to select predictor names rather than
  regularization-path labels, and remove the unused mandatory Matrix dependency.
