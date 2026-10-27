# Tests for results_processing.R
# Test post-processing and efficiency calculations

library(testthat)

# Source helpers if not already loaded
if (!exists("calculate_efficiency_metrics")) {
  source("helpers/results_processing.R")
  source("helpers/storage_utils.R")
  source("helpers/utilities.R")
}

setup_test_results <- function() {
  list(
    lm = list(
      mse = data.frame(
        sample_size = c(50, 100, 150, 200),
        mean = c(5.0, 3.0, 2.5, 2.3),
        sd = c(1.0, 0.8, 0.6, 0.5),
        median = c(4.9, 2.9, 2.4, 2.2),
        q025 = c(3.5, 2.0, 1.8, 1.7),
        q975 = c(6.5, 4.0, 3.2, 2.9),
        min = c(3.0, 1.5, 1.2, 1.0),
        max = c(7.0, 4.5, 3.8, 3.5)
      ),
      r_squared = data.frame(
        sample_size = c(50, 100, 150, 200),
        mean = c(0.70, 0.85, 0.90, 0.92),
        sd = c(0.10, 0.05, 0.03, 0.02),
        median = c(0.71, 0.86, 0.91, 0.93),
        q025 = c(0.55, 0.75, 0.85, 0.88),
        q975 = c(0.85, 0.95, 0.95, 0.96),
        min = c(0.50, 0.70, 0.80, 0.85),
        max = c(0.90, 0.98, 0.98, 0.99)
      )
    )
  )
}

test_that("calculate_efficiency_metrics computes correctly for MSE", {
  results <- setup_test_results()
  results <- calculate_efficiency_metrics(results, threshold = 0.95)
  
  expect_true("efficiency" %in% names(results$lm$mse))
  expect_length(results$lm$mse$efficiency, 4)
  
  # For MSE (lower is better), efficiency should increase with sample size
  eff <- results$lm$mse$efficiency
  expect_true(all(eff >= 0 & eff <= 1))
  expect_true(eff[4] > eff[1])  # Larger n should have higher efficiency
})

test_that("calculate_efficiency_metrics computes correctly for R-squared", {
  results <- setup_test_results()
  results <- calculate_efficiency_metrics(results, threshold = 0.95)
  
  expect_true("efficiency" %in% names(results$lm$r_squared))
  
  # For R² (higher is better), efficiency = current / max
  eff <- results$lm$r_squared$efficiency
  expect_true(all(eff >= 0 & eff <= 1))
  expect_equal(eff[4], 1.0)  # Last sample size should have efficiency = 1
})

test_that("calculate_efficiency_metrics handles all NA values", {
  results <- list(
    lm = list(
      mse = data.frame(
        sample_size = c(50, 100),
        mean = c(NA_real_, NA_real_),
        sd = c(NA_real_, NA_real_),
        median = c(NA_real_, NA_real_),
        q025 = c(NA_real_, NA_real_),
        q975 = c(NA_real_, NA_real_),
        min = c(NA_real_, NA_real_),
        max = c(NA_real_, NA_real_)
      )
    )
  )
  
  results <- calculate_efficiency_metrics(results, threshold = 0.95)
  
  expect_true("efficiency" %in% names(results$lm$mse))
  expect_true(all(is.na(results$lm$mse$efficiency)))
})

test_that("determine_minimum_sample_sizes finds correct thresholds", {
  results <- setup_test_results()
  results <- calculate_efficiency_metrics(results, threshold = 0.95)
  
  min_sizes <- determine_minimum_sample_sizes(
    results,
    threshold = 0.95,
    metrics = c("mse", "r_squared")
  )
  
  expect_type(min_sizes, "list")
  expect_true("lm" %in% names(min_sizes))
  expect_true(all(c("mse", "r_squared") %in% names(min_sizes$lm)))
  
  # Should find sample sizes where efficiency >= 0.95
  expect_true(is.numeric(min_sizes$lm$mse) || is.na(min_sizes$lm$mse))
})

test_that("determine_minimum_sample_sizes returns NA when threshold not met", {
  results <- list(
    lm = list(
      mse = data.frame(
        sample_size = c(50, 100),
        mean = c(5.0, 4.5),
        sd = c(1.0, 0.9),
        median = c(5.0, 4.5),
        q025 = c(4.0, 3.5),
        q975 = c(6.0, 5.5),
        min = c(3.5, 3.0),
        max = c(6.5, 6.0),
        efficiency = c(0.10, 0.20)  # Never reaches 0.95
      )
    )
  )
  
  min_sizes <- determine_minimum_sample_sizes(
    results,
    threshold = 0.95,
    metrics = c("mse")
  )
  
  expect_true(is.na(min_sizes$lm$mse))
})

test_that("create_summary_table produces correct structure", {
  results <- setup_test_results()
  results <- calculate_efficiency_metrics(results, threshold = 0.95)
  
  min_sizes <- determine_minimum_sample_sizes(
    results,
    threshold = 0.95,
    metrics = c("mse", "r_squared")
  )
  
  summary_table <- create_summary_table(results, min_sizes)
  
  expect_s3_class(summary_table, "data.frame")
  expect_true(all(c("model", "metric", "min_sample_size", "max_sample_size") %in% 
                  names(summary_table)))
  expect_equal(nrow(summary_table), 2)  # 1 model × 2 metrics
})

test_that("plot_performance_curve creates plot without errors", {
  plot_file <- tempfile(fileext = ".pdf")
  grDevices::pdf(plot_file)
  on.exit({
    grDevices::dev.off()
    unlink(plot_file)
  }, add = TRUE)
  results <- setup_test_results()
  results <- calculate_efficiency_metrics(results, threshold = 0.95)
  
  min_sizes <- determine_minimum_sample_sizes(
    results,
    threshold = 0.95,
    metrics = c("mse")
  )
  
  # Should not throw error
  expect_silent(
    plot_performance_curve(
      results = results,
      model = "lm",
      metric = "mse",
      add_efficiency = TRUE,
      min_sample_size = min_sizes$lm$mse
    )
  )
})

test_that("plot_performance_curve handles missing model or metric", {
  results <- setup_test_results()
  
  expect_error(
    plot_performance_curve(results, "nonexistent_model", "mse"),
    "Model not found"
  )
  
  expect_error(
    plot_performance_curve(results, "lm", "nonexistent_metric"),
    "Metric not found"
  )
})

test_that("generate_summary_plots handles multiple models and metrics", {
  results <- setup_test_results()
  results <- calculate_efficiency_metrics(results, threshold = 0.95)
  
  min_sizes <- determine_minimum_sample_sizes(
    results,
    threshold = 0.95,
    metrics = c("mse", "r_squared")
  )
  
  plots <- generate_summary_plots(results, min_sizes)
  
  # Should return list or NULL
  expect_true(is.null(plots) || is.list(plots))
})

test_that("efficiency calculation handles equal performance", {
  results <- list(
    lm = list(
      mse = data.frame(
        sample_size = c(50, 100, 150),
        mean = c(2.5, 2.5, 2.5),  # No improvement
        sd = c(0.5, 0.5, 0.5),
        median = c(2.5, 2.5, 2.5),
        q025 = c(2.0, 2.0, 2.0),
        q975 = c(3.0, 3.0, 3.0),
        min = c(1.8, 1.8, 1.8),
        max = c(3.2, 3.2, 3.2)
      )
    )
  )
  
  results <- calculate_efficiency_metrics(results, threshold = 0.95)
  
  # Should handle division by zero
  expect_true("efficiency" %in% names(results$lm$mse))
  expect_false(any(is.na(results$lm$mse$efficiency)))
})

test_that("print.sample_size_simulation displays information", {
  simulation_result <- list(
    results = setup_test_results(),
    min_sample_sizes = list(
      lm = list(mse = 150, r_squared = 100)
    ),
    validation = NULL,
    configuration = list(
      reference_model = "lm",
      target_models = c("lm"),
      sample_sizes = c(50, 100, 150, 200),
      n_simulations = 100,
      metrics = c("mse", "r_squared"),
      efficiency_threshold = 0.95
    )
  )
  class(simulation_result) <- "sample_size_simulation"
  
  # Should print without error
  expect_output(print(simulation_result), "Sample Size Simulation Results")
  expect_output(print(simulation_result), "lm")
})

test_that("summary.sample_size_simulation returns data frame", {
  simulation_result <- list(
    results = setup_test_results(),
    min_sample_sizes = list(
      lm = list(mse = 150, r_squared = 100)
    )
  )
  class(simulation_result) <- "sample_size_simulation"
  
  summary_df <- summary(simulation_result)
  
  expect_s3_class(summary_df, "data.frame")
  expect_true(nrow(summary_df) > 0)
})

test_that("efficiency bounds are maintained", {
  results <- setup_test_results()
  
  # Add extreme values
  results$lm$mse$mean <- c(10, 5, 2, 1)
  
  results <- calculate_efficiency_metrics(results, threshold = 0.95)
  
  eff <- results$lm$mse$efficiency
  expect_true(all(eff >= 0))
  expect_true(all(eff <= 1))
})
