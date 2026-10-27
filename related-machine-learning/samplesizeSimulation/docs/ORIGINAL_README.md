# samplesizeSimulation

Simulation based sample size planning for tabular prediction models, maintained
by Louis Chen. Licensed under GPL-3.

The framework compares prediction performance across candidate sample sizes,
with random or forward chaining validation. Recommendations describe relative
performance under the specified assumptions. They do not represent formal
power, guaranteed external accuracy, or a statistically valid online stopping
rule.

This source tree is being prepared for its first CRAN submission; it is not
currently available through `install.packages("samplesizeSimulation")`.

## Install from source

From this directory, with R and the required dependencies installed:

```sh
R CMD build .
R CMD INSTALL samplesizeSimulation_0.1.0.tar.gz
```

Only `MASS` is required beyond R's standard packages. Optional models use the
packages listed by `get_model_registry()`. Building the vignette needs `knitr`;
running the tests needs `testthat`.

```r
library(samplesizeSimulation)

result <- run_general_simulation(
  sample_sizes = c(30, 60, 90),
  n_simulations = 3,
  target_models = "lm",
  metrics = c("mse", "r_squared"),
  infinite_n = NULL,
  cv_folds = 3,
  cv_repeats = 1,
  seed = 123,
  verbose = FALSE
)

summary(result)
result$results$lm$mse
get_model_registry(available_only = TRUE)
```

These small settings demonstrate the interface. Substantive studies need more
repetitions, sensitivity analysis, an adequate candidate grid, and independent
validation. A seeded simulation restores the caller's random-number state.

## Interpret the results

- `sample_size` counts total generated rows, rather than the rows in a CV
  training fold.
- `q025` and `q975` are simulation percentiles, rather than confidence limits
  for the mean. `n_success` and `n_failed` expose unavailable metric values.
- Error efficiency measures the fraction of observed improvement between the
  smallest and largest sample sizes. Other metrics use the ratio to performance
  at the largest size. A threshold of 0.95 is a descriptive convention.
- `infinite_n` is a finite reference run. Its adequacy must be assessed.
- Missing or deteriorating reference performance may yield `NA` recommendations.
  Equal endpoint errors use efficiency one by convention; inspect the curve.
- `lookup_table` accepts prior results or validated CSV tables. Only matching
  keys and complete metric rows are reused. Retain R/dependency version
  provenance and regenerate caches after changing the modeling environment.

## Evaluate original ordered data

```r
score <- fit_and_evaluate_model(
  data = observations,
  model_type = "lm",
  response_var = "y",
  predictor_vars = c("x1", "x2"),
  metrics = "mse",
  cv_folds = 3,
  cv_repeats = 1,
  validation_strategy = "ordered",
  order_var = "time"
)
```

This evaluator uses earlier observations to predict later blocks. It does not
provide grouped folds, temporal gaps, purging of overlapping labels, or
fold-specific feature preprocessing. The simulation function rejects ordered
validation with bootstrap/synthetic resampling because chronology alone does
not make those splits independent. Random empirical bootstrap folds can also
share original records; their scores are simulation results for a plug-in
population, not proof of real-data generalization.

Classification metrics use numeric 0/1 labels. Mixed models fit random
intercepts from a `.group` column and predict population-level outcomes.
Parametric categorical generation requires numeric scores. Adjusted R-squared,
standalone `efficiency`, and parallel execution are unsupported.

## Development

Canonical implementations are in `R/`, help files in `man/`, and package tests
in `tests/testthat/`. The old `source("functionSamplesizeSimu.R")` entry point
and helper paths remain as compatibility loaders. Study-specific examples and
historical notes are excluded from the source package.

Run `Rscript tools/check-package.R` for a source build and local CRAN-style
check. See [the submission record](documentation/CRAN-READINESS.md) for the
checks actually performed and remaining release requirements. See [the
extension assessment](documentation/EXTENSIONS.md) for image and momentum
applications.
