# Tests for storage_utils.R
# Test storage initialization and result saving/loading

library(testthat)

# Source helpers if not already loaded
if (!exists("initialize_results_storage")) {
  source("helpers/storage_utils.R")
  source("helpers/utilities.R")
}

test_that("initialize_results_storage creates correct structure", {
  storage <- initialize_results_storage(
    sample_sizes = c(50, 100, 150),
    target_models = c("lm", "rf"),
    metrics = c("mse", "r_squared"),
    predictor_config = list()
  )
  
  # Check top-level structure
  expect_type(storage, "list")
  expect_equal(names(storage), c("lm", "rf"))
  
  # Check model-level structure
  expect_equal(names(storage$lm), c("mse", "r_squared"))
  expect_equal(names(storage$rf), c("mse", "r_squared"))
  
  # Check metric data frames
  expect_s3_class(storage$lm$mse, "data.frame")
  expect_equal(nrow(storage$lm$mse), 3)
  expect_true(all(c("sample_size", "mean", "sd", "median", "q025", "q975") %in% 
                  names(storage$lm$mse)))
  
  # Check sample sizes are correct
  expect_equal(storage$lm$mse$sample_size, c(50, 100, 150))
  
  # Check all values initialized as NA
  expect_true(all(is.na(storage$lm$mse$mean)))
})

test_that("initialize_sim_storage creates correct structure", {
  sim_storage <- initialize_sim_storage(
    n_simulations = 100,
    target_models = c("lm", "rf"),
    metrics = c("mse", "r_squared")
  )
  
  # Check structure
  expect_type(sim_storage, "list")
  expect_equal(names(sim_storage), c("lm", "rf"))
  expect_equal(names(sim_storage$lm), c("mse", "r_squared"))
  
  # Check vectors
  expect_type(sim_storage$lm$mse, "double")
  expect_length(sim_storage$lm$mse, 100)
  expect_true(all(is.na(sim_storage$lm$mse)))
})

test_that("aggregate_results computes statistics correctly", {
  # Initialize storage
  results <- initialize_results_storage(
    sample_sizes = c(50, 100),
    target_models = c("lm"),
    metrics = c("mse"),
    predictor_config = list()
  )
  
  # Create simulation results
  sim_results <- list(
    lm = list(
      mse = c(2.5, 2.3, 2.7, 2.4, 2.6)
    )
  )
  
  # Aggregate
  results <- aggregate_results(results, sim_results, n = 50, i_size = 1)
  
  # Check aggregated values
  expect_false(is.na(results$lm$mse$mean[1]))
  expect_equal(results$lm$mse$mean[1], mean(sim_results$lm$mse))
  expect_equal(results$lm$mse$sd[1], sd(sim_results$lm$mse))
  expect_equal(results$lm$mse$median[1], median(sim_results$lm$mse))
  
  # Check quantiles
  expect_equal(results$lm$mse$q025[1], unname(quantile(sim_results$lm$mse, 0.025)))
  expect_equal(results$lm$mse$q975[1], unname(quantile(sim_results$lm$mse, 0.975)))
})

test_that("aggregate_results handles NA and Inf values", {
  results <- initialize_results_storage(
    sample_sizes = c(50),
    target_models = c("lm"),
    metrics = c("mse"),
    predictor_config = list()
  )
  
  # Create simulation results with NA and Inf
  sim_results <- list(
    lm = list(
      mse = c(2.5, NA, 2.7, Inf, 2.6)
    )
  )
  
  # Aggregate
  results <- aggregate_results(results, sim_results, n = 50, i_size = 1)
  
  # Should compute statistics on valid values only
  valid_values <- c(2.5, 2.7, 2.6)
  expect_equal(results$lm$mse$mean[1], mean(valid_values))
  expect_false(is.na(results$lm$mse$mean[1]))
})

test_that("simulation tables round-trip aggregate results", {
  results <- initialize_results_storage(
    sample_sizes = c(50, 100),
    target_models = c("lm"),
    metrics = c("mse"),
    predictor_config = list()
  )
  results$lm$mse[, c("mean", "sd", "median", "q025", "q975", "min", "max")] <-
    rbind(
      c(2.5, 0.2, 2.5, 2.1, 2.9, 2.0, 3.0),
      c(2.0, 0.1, 2.0, 1.8, 2.2, 1.7, 2.3)
    )
  results$lm$mse$efficiency <- c(0.8, 1.0)

  table <- results_to_simulation_table(
    results,
    simulation_keys = c(lm = "test-key")
  )

  expect_equal(nrow(table), 2)
  expect_true(all(c("simulation_key", "model", "metric", "efficiency") %in%
                    names(table)))
  expect_equal(unique(table$simulation_key), "test-key")

  empty <- initialize_results_storage(
    sample_sizes = c(50, 100),
    target_models = "lm",
    metrics = "mse",
    predictor_config = list()
  )
  restored <- restore_results_from_table(
    results = empty,
    lookup_table = table,
    simulation_keys = c(lm = "test-key"),
    target_models = "lm",
    metrics = "mse",
    n = 50,
    i_size = 1
  )

  expect_equal(restored$cached_models, "lm")
  expect_equal(restored$results$lm$mse$mean[1], 2.5)
})

test_that("table recommendations avoid rerunning simulations", {
  table <- data.frame(
    simulation_key = rep("key-1", 3),
    model = rep("lm", 3),
    metric = rep("mse", 3),
    sample_size = c(50, 100, 150),
    mean = c(3, 2, 1),
    sd = c(0.2, 0.2, 0.1),
    median = c(3, 2, 1),
    q025 = c(2.5, 1.5, 0.8),
    q975 = c(3.5, 2.5, 1.2),
    min = c(2, 1, 0.7),
    max = c(4, 3, 1.3),
    efficiency = c(0, 0.96, 1)
  )

  recommendation <- recommend_from_simulation_table(table, 0.95)
  expect_equal(recommendation$min_sample_size, 100)

  updated <- table[2:3, ]
  updated$mean <- c(1.9, 0.9)
  merged <- merge_simulation_tables(table, updated)
  expect_equal(nrow(merged), 3)
  expect_equal(merged$mean[merged$sample_size == 100], 1.9)
})

test_that("save_simulation_results creates files correctly", {
  skip_on_cran()
  
  # Create temporary directory
  temp_dir <- tempdir()
  output_path <- file.path(temp_dir, "test_results")
  
  # Create minimal results
  results <- list(
    lm = list(
      mse = data.frame(
        sample_size = c(50, 100),
        mean = c(2.5, 2.0),
        sd = c(0.5, 0.3),
        median = c(2.4, 1.9),
        q025 = c(2.0, 1.5),
        q975 = c(3.0, 2.5),
        min = c(1.8, 1.3),
        max = c(3.2, 2.7)
      )
    )
  )
  
  min_sample_sizes <- list(
    lm = list(mse = 100)
  )
  
  timestamp <- "20251112_120000"
  
  # Save results
  save_simulation_results(
    results = results,
    min_sample_sizes = min_sample_sizes,
    validation_results = NULL,
    output_path = output_path,
    timestamp = timestamp
  )
  
  # Check files exist
  expect_true(dir.exists(output_path))
  expect_true(file.exists(file.path(output_path, 
                                    paste0("results_lm_mse_", timestamp, ".csv"))))
  expect_true(file.exists(file.path(output_path, 
                                    paste0("min_sample_sizes_", timestamp, ".csv"))))
  expect_true(file.exists(file.path(output_path, 
                                    paste0("full_results_", timestamp, ".rds"))))
  
  # Check content
  saved_csv <- read.csv(file.path(output_path, 
                                  paste0("results_lm_mse_", timestamp, ".csv")))
  expect_equal(nrow(saved_csv), 2)
  expect_equal(saved_csv$sample_size, c(50, 100))
  
  # Clean up
  unlink(output_path, recursive = TRUE)
})

test_that("load_simulation_results reads saved files", {
  skip_on_cran()
  
  # Create and save results
  temp_dir <- tempdir()
  output_path <- file.path(temp_dir, "test_load_results")
  
  original_results <- list(
    lm = list(
      mse = data.frame(
        sample_size = c(50, 100),
        mean = c(2.5, 2.0),
        sd = c(0.5, 0.3),
        median = c(2.4, 1.9),
        q025 = c(2.0, 1.5),
        q975 = c(3.0, 2.5),
        min = c(1.8, 1.3),
        max = c(3.2, 2.7)
      )
    )
  )
  
  timestamp <- "20251112_130000"
  
  save_simulation_results(
    results = original_results,
    min_sample_sizes = list(lm = list(mse = 100)),
    validation_results = NULL,
    output_path = output_path,
    timestamp = timestamp
  )
  
  # Load results
  loaded_results <- load_simulation_results(output_path, timestamp)
  
  # Check loaded data
  expect_equal(loaded_results$lm$mse$sample_size, c(50, 100))
  expect_equal(loaded_results$lm$mse$mean, c(2.5, 2.0))
  
  # Clean up
  unlink(output_path, recursive = TRUE)
})
