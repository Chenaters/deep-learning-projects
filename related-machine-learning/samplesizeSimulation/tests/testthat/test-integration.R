# Integration tests for the simulation framework
# Focus on broad behavior coverage with small, fast configurations.

library(testthat)

test_that("run_general_simulation completes a full workflow", {
  skip_if_not_installed("MASS")
  skip_if_not_installed("Matrix")

  results <- run_with_overrides()

  expect_s3_class(results, "sample_size_simulation")
  expect_true(all(c("results", "min_sample_sizes", "configuration") %in% names(results)))
  expect_true("lm" %in% names(results$results))
  expect_true(all(c("mse", "r_squared") %in% names(results$results$lm)))
  expect_equal(results$results$lm$mse$sample_size, c(30, 40))
  expect_false(all(is.na(results$results$lm$mse$mean)))
})

test_that("run_general_simulation supports multiple target models", {
  skip_if_not_installed("MASS")
  skip_if_not_installed("Matrix")

  results <- run_with_overrides(list(
    target_models = c("lm", "glm"),
    metrics = c("mse"),
    sample_sizes = c(35),
    n_simulations = 1
  ))

  expect_true(all(c("lm", "glm") %in% names(results$results)))
  expect_true(all(c("lm", "glm") %in% names(results$min_sample_sizes)))
})

test_that("run_general_simulation supports GAM and neural-network targets", {
  skip_if_not_installed("MASS")
  skip_if_not_installed("Matrix")
  skip_if_not_installed("mgcv")
  skip_if_not_installed("nnet")

  results <- run_with_overrides(list(
    target_models = c("gam", "nnet"),
    model_options = list(
      gam = list(k = 4),
      nnet = list(size = 2, maxit = 30)
    ),
    metrics = "mse",
    sample_sizes = 40,
    n_simulations = 1,
    cv_folds = 0
  ))

  expect_true(all(c("gam", "nnet") %in% names(results$results)))
  expect_false(is.na(results$results$gam$mse$mean))
  expect_false(is.na(results$results$nnet$mse$mean))
})

test_that("run_general_simulation supports random forest aliases", {
  skip_if_not_installed("MASS")
  skip_if_not_installed("Matrix")
  skip_if_not_installed("randomForest")
  skip_on_ci()

  results <- run_with_overrides(list(
    target_models = c("randomForest"),
    metrics = c("mse"),
    sample_sizes = c(35),
    n_simulations = 1
  ))

  expect_true("rf" %in% names(results$results))
  expect_true("mse" %in% names(results$results$rf))
  expect_false(all(is.na(results$results$rf$mse$mean)))
  expect_equal(results$configuration$target_models, "rf")
})

test_that("run_general_simulation handles parametric and empirical generation", {
  skip_if_not_installed("MASS")
  skip_if_not_installed("Matrix")

  param_results <- run_with_overrides(list(
    data_generation = list(
      method = "parametric",
      params = list(
        x1 = list(distribution = "normal", mean = 0, sd = 1),
        x2 = list(distribution = "uniform", min = -1, max = 1)
      )
    ),
    sample_sizes = c(30),
    n_simulations = 1,
    metrics = c("mse")
  ))
  expect_s3_class(param_results, "sample_size_simulation")

  empirical_data <- data.frame(
    y = rnorm(80, 0, 1),
    x1 = rnorm(80, 0, 1),
    x2 = rnorm(80, 0, 1)
  )
  empirical_results <- run_with_overrides(list(
    data_generation = list(
      method = "empirical",
      params = list(empirical_data = empirical_data)
    ),
    predictor_config = list(
      names = c("x1", "x2"),
      regenerate_outcome = FALSE,
      effects = list(fixed = c(0.5, -0.3), residual_sd = 1.0)
    ),
    sample_sizes = c(30),
    n_simulations = 1,
    metrics = c("mse")
  ))
  expect_s3_class(empirical_results, "sample_size_simulation")
})

test_that("run_general_simulation handles binary glm metrics", {
  skip_if_not_installed("MASS")
  skip_if_not_installed("Matrix")

  results <- run_with_overrides(list(
    reference_model = "glm",
    target_models = c("glm"),
    predictor_config = list(
      names = c("x1", "x2"),
      types = c("continuous", "continuous"),
      effects = list(
        fixed = c(0.8, -0.6),
        family = "binomial"
      )
    ),
    sample_sizes = c(100),
    n_simulations = 1,
    metrics = c("auc", "accuracy"),
    cv_folds = 0
  ))

  expect_s3_class(results, "sample_size_simulation")
  expect_true(all(c("auc", "accuracy") %in% names(results$results$glm)))
})

test_that("run_general_simulation includes and stores explicit infinite_n", {
  skip_if_not_installed("MASS")
  skip_if_not_installed("Matrix")

  results <- run_with_overrides(list(
    sample_sizes = c(25, 35),
    infinite_n = 45,
    n_simulations = 1,
    metrics = c("mse")
  ))

  expect_equal(results$configuration$sample_sizes, c(25, 35, 45))
  expect_equal(results$infinite_results$n, 45)
  expect_equal(nrow(results$results$lm$mse), 3)
  expect_equal(nrow(results$infinite_results$metrics$lm$mse), 1)
})

test_that("run_general_simulation reuses exact simulation-table rows", {
  skip_if_not_installed("MASS")
  skip_if_not_installed("Matrix")

  first <- run_with_overrides(list(
    sample_sizes = c(25, 35),
    n_simulations = 1,
    metrics = "mse"
  ))
  second <- run_with_overrides(list(
    sample_sizes = c(25, 35),
    n_simulations = 1,
    metrics = "mse",
    lookup_table = first$simulation_table
  ))

  expect_equal(second$cache$fraction_reused, 1)
  expect_equal(second$results$lm$mse, first$results$lm$mse)
  expect_equal(second$simulation_table, first$simulation_table)
})

test_that("run_general_simulation supports ordinal outcomes", {
  skip_if_not_installed("MASS")
  skip_if_not_installed("Matrix")

  results <- run_with_overrides(list(
    reference_model = "ordinal",
    target_models = "ordinal",
    predictor_config = list(
      names = c("x1", "x2"),
      types = c("continuous", "continuous"),
      effects = list(
        fixed = c(0.8, -0.5),
        thresholds = c(-0.75, 0.75),
        labels = c("low", "middle", "high")
      )
    ),
    sample_sizes = 100,
    n_simulations = 1,
    metrics = c("mse", "mae"),
    cv_folds = 0
  ))

  expect_true("ordinal" %in% names(results$results))
  expect_true(all(is.finite(c(
    results$results$ordinal$mse$mean,
    results$results$ordinal$mae$mean
  ))))
})

test_that("run_general_simulation can save results to disk", {
  skip_if_not_installed("MASS")
  skip_if_not_installed("Matrix")

  output_path <- file.path(tempdir(), "integration_test_results")
  on.exit(unlink(output_path, recursive = TRUE), add = TRUE)

  run_with_overrides(list(
    sample_sizes = c(30),
    n_simulations = 1,
    metrics = c("mse"),
    output_path = output_path
  ))

  expect_true(dir.exists(output_path))
  files <- list.files(output_path)
  expect_true(any(grepl("^results_lm_mse_", files)))
  expect_true(any(grepl("^min_sample_sizes_", files)))
  expect_true(any(grepl("^full_results_", files)))
  expect_true(any(grepl("^simulation_table_", files)))
})

test_that("configuration is stored correctly", {
  skip_if_not_installed("MASS")
  skip_if_not_installed("Matrix")

  results <- run_with_overrides(list(
    sample_sizes = list(min = 20, max = 30, step = 10),
    n_simulations = 2,
    metrics = c("mse"),
    efficiency_threshold = 0.9,
    seed = 99
  ))

  config <- results$configuration
  expect_equal(config$reference_model, "lm")
  expect_equal(config$target_models, c("lm"))
  expect_equal(config$n_simulations, 2)
  expect_equal(config$metrics, c("mse"))
  expect_equal(config$efficiency_threshold, 0.9)
  expect_equal(config$seed, 99)
  expect_equal(config$sample_sizes, c(20, 30))
})
