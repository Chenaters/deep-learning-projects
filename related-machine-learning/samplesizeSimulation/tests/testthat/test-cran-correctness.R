test_that("invalid simulation configurations fail before fitting", {
  expect_error(run_with_overrides(list(n_simulations = 0)), "positive integer")
  expect_error(run_with_overrides(list(n_simulations = Inf)), "positive integer")
  expect_error(run_with_overrides(list(cv_repeats = 0)), "positive integer")
  expect_error(run_with_overrides(list(cv_folds = 1)), "cv_folds")
  expect_error(run_with_overrides(list(cv_folds = 30)), "smaller than every")
  expect_error(run_with_overrides(list(efficiency_threshold = NA_real_)), "efficiency_threshold")
  expect_error(run_with_overrides(list(infinite_n = 10)), "at least as large")
  expect_error(run_with_overrides(list(target_models = character(0))), "nonempty")
  expect_error(run_with_overrides(list(metrics = "adj_r_squared")), "not standalone")
  expect_error(run_with_overrides(list(parallel = TRUE)), "not implemented")
})

test_that("sample-size order cannot change recommendations", {
  expect_equal(process_sample_sizes(c(100, 30, 60, 30)), c(30, 60, 100))
  expect_error(process_sample_sizes(list(min = 10, max = 20, step = 0)), "step > 0")
  result <- list(lm = list(mse = data.frame(
    sample_size = c(100, 30, 60), mean = c(1, 3, 1.05)
  )))
  processed <- calculate_efficiency_metrics(result, 0.95)
  expect_equal(determine_minimum_sample_sizes(processed, 0.95, "mse")$lm$mse, 60)
})

test_that("missing or worsening reference performance cannot produce recommendations", {
  make_result <- function(values) list(lm = list(mse = data.frame(
    sample_size = c(30, 60, 100), mean = values
  )))
  for (values in list(c(3, 2, NA), c(NA, 2, 1), c(1, 2, 3))) {
    result <- calculate_efficiency_metrics(make_result(values), 0.95)
    expect_true(all(is.na(result$lm$mse$efficiency)))
    expect_true(is.na(determine_minimum_sample_sizes(result, 0.95, "mse")$lm$mse))
  }
  result <- list(lm = list(r_squared = data.frame(sample_size = c(30, 60), mean = c(-2, -1))))
  expect_true(all(is.na(calculate_efficiency_metrics(result, 0.95)$lm$r_squared$efficiency)))
})

test_that("binary edge cases have defined metric outputs", {
  scores <- calculate_metrics(c(1, 1), c(0.9, 0.8), c("auc", "accuracy", "specificity", "brier"))
  expect_true(is.na(scores$auc))
  expect_equal(scores$accuracy, 1)
  expect_true(is.na(scores$specificity))
  expect_equal(scores$brier, 0.025)
  expect_equal(calculate_metrics(c(0, 1), c(0.9, 0.1), "f1")$f1, 0)
  expect_true(is.na(calculate_metrics(c(1, 1), c(1, 1), "r_squared")$r_squared))
  expect_error(calculate_metrics(1:3, 1:2, "mse"), "equal length")
  expect_equal(calculate_auc_manual(c(0, 1, 0, 1), rep(0.5, 4)), 0.5)
  expect_equal(calculate_auc_manual(c(0, 1), c(1, 0)), 0)
})

test_that("seeded simulations restore caller RNG state", {
  set.seed(12)
  before <- .Random.seed
  result <- run_with_overrides()
  expect_identical(.Random.seed, before)
  expect_output(printed <- print(result), "Sample Size Simulation Results")
  expect_identical(printed, result)
  expect_s3_class(summary(result), "data.frame")
})

test_that("ordered evaluation rejects in-sample and row-bootstrap leakage", {
  expect_error(run_with_overrides(list(validation_strategy = "ordered", cv_folds = 0)), "out-of-sample")
  expect_error(run_with_overrides(list(
    validation_strategy = "ordered", data_generation = list(method = "empirical")
  )), "row-bootstrap")
  observations <- data.frame(y = 1:12, x = 1:12)
  expect_error(fit_and_evaluate_model(observations, "lm", "y", "x", "mse", 0, 1,
                                   validation_strategy = "ordered"), "out-of-sample")
})

test_that("simulation failures remain visible after aggregation and cache reuse", {
  result <- initialize_results_storage(30, "lm", "mse", list())
  result <- aggregate_results(result, list(lm = list(mse = c(1, 2, NA, Inf))), 30, 1)
  expect_equal(result$lm$mse$n_success, 2)
  expect_equal(result$lm$mse$n_failed, 2)
  table <- results_to_simulation_table(result, c(lm = "key"))
  restored <- restore_results_from_table(initialize_results_storage(30, "lm", "mse", list()),
                                        table, c(lm = "key"), "lm", "mse", 30, 1)
  expect_equal(restored$results$lm$mse$n_failed, 2)
  expect_error(load_simulation_table(rbind(table, table)), "duplicate")
})

test_that("mixed simulations preserve group IDs needed by fitted models", {
  skip_if_not_installed("lme4")
  args <- base_joint_args()
  args$reference_model <- "lmm"
  args$predictor_config$effects$n_groups <- 5
  args$predictor_config$effects$group_variance <- 2
  observations <- generate_data(200, args$data_generation, args$predictor_config, "lmm", 12)
  expect_true(".group" %in% names(observations))
  expect_equal(nlevels(observations$.group), 5)
  score <- fit_and_evaluate_model(observations, "lmm", "y", c("x1", "x2"), "mse", NULL, 1)
  expect_true(is.finite(score$mse))
})

test_that("regularized predictions use feature rows rather than lambda columns", {
  skip_if_not_installed("glmnet")
  set.seed(71)
  observations <- data.frame(x1 = rnorm(50), x2 = rnorm(50))
  observations$y <- 2 * observations$x1 - observations$x2 + rnorm(50, sd = 0.1)
  for (model_type in c("ridge", "lasso")) {
    model <- fit_model(model_type, y ~ x1 + x2, observations)
    predictions <- predict_model(model, model_type, observations)
    expect_length(predictions, 50)
    expect_true(all(is.finite(predictions)))
    expect_lt(mean((observations$y - predictions)^2), 0.2)
  }
})
