# pmsesampling

`pmsesampling` estimates the training sample size needed to reach a prediction
mean-squared error (PMSE) target. It keeps two model classes separate:

- `pmse_samplesize()` is an analytic calculation for two nested ordinary
  least-squares (OLS) models under the assumptions of Ma (2023).
- `pmse_learning_curve()` empirically evaluates ridge, k-nearest-neighbor, OLS,
  or custom machine-learning models on a shared holdout set.

The distinction matters: the OLS formula is not silently applied to algorithms
with different learning behavior.

## Compare machine-learning models

```r
models <- list(
  pmse_model("linear"),
  pmse_model("ridge", lambda = 0.5),
  pmse_model("knn", neighbors = 5)
)

pmse_table <- pmse_learning_curve(
  mtcars,
  outcome = "mpg",
  predictors = c("wt", "hp", "disp"),
  models = models,
  sample_sizes = c(12, 18, 24),
  repeats = 20,
  seed = 42
)

pmse_table_samplesize(pmse_table, target_pmse = 8)
```

`metric = "upper"` is the conservative lookup default. It uses a one-sided
upper confidence bound rather than only the mean PMSE. Empirical estimates are
limited by the representativeness and size of the holdout data.

## Reuse the table instead of refitting

The learning-curve result is a regular data frame, so it can be persisted with
`saveRDS()` or `write.csv()`. Many target PMSE values can then be checked with
`pmse_table_samplesize()` without fitting any models again.

An existing table can also be extended. Compatible rows are reused and only
missing model/sample-size combinations are computed:

```r
extended <- pmse_learning_curve(
  mtcars,
  outcome = "mpg",
  predictors = c("wt", "hp", "disp"),
  models = models,
  sample_sizes = c(12, 18, 24, 26),
  repeats = 20,
  seed = 42,
  previous = pmse_table
)
```

For expensive packages such as boosted trees or random forests, define a
custom adapter once:

```r
model <- pmse_model(
  "custom",
  name = "my model",
  fit = function(x, y) my_fit_function(x, y),
  predict = function(object, newx) my_predict_function(object, newx)
)
```

The custom predictor must return one finite numeric prediction per row of
`newx`.
