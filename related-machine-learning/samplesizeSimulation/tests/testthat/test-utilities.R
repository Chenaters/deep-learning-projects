# Tests for utilities.R
# Test general utility functions

library(testthat)

# Source helpers if not already loaded
if (!exists("%||%")) {
  source("helpers/utilities.R")
}

test_that("null-coalescing operator works correctly", {
  # Test with NULL
  expect_equal(NULL %||% "default", "default")
  
  # Test with non-NULL
  expect_equal("value" %||% "default", "value")
  
  # Test with empty string
  expect_equal("" %||% "default", "")
  
  # Test with numeric
  expect_equal(0 %||% 10, 0)
  expect_equal(NULL %||% 10, 10)
  
  # Test with FALSE
  expect_equal(FALSE %||% TRUE, FALSE)
})

test_that("check_packages validates package availability", {
  # Test with installed packages
  expect_silent(check_packages(c("base", "stats")))
  
  # Test with non-existent package (required)
  expect_error(
    check_packages("nonexistent_package_12345", required = TRUE),
    "required but not installed"
  )
  
  # Test with non-existent package (optional)
  expect_warning(
    check_packages("nonexistent_package_12345", required = FALSE),
    "not installed"
  )
})

test_that("generate_timestamp creates valid timestamp strings", {
  timestamp <- generate_timestamp()
  
  # Should be a character string
  expect_type(timestamp, "character")
  
  # Should have length 15 (YYYYMMDD_HHMMSS)
  expect_equal(nchar(timestamp), 15)
  
  # Should match pattern
  expect_match(timestamp, "^\\d{8}_\\d{6}$")
  
  # Custom format
  custom <- generate_timestamp("%Y-%m-%d")
  expect_match(custom, "^\\d{4}-\\d{2}-\\d{2}$")
})

test_that("derived seeds are stable across loop reordering", {
  seed_a <- derive_simulation_seed(2025, 100, 2, "data")
  seed_b <- derive_simulation_seed(2025, 100, 2, "data")

  expect_equal(seed_a, seed_b)
  expect_type(seed_a, "integer")
  expect_false(identical(seed_a, derive_simulation_seed(2025, 100, 2, "nnet")))
  expect_null(derive_simulation_seed(NULL, 100, 2, "data"))
})

test_that("process_sample_sizes handles different inputs", {
  # Test list input
  sizes1 <- process_sample_sizes(list(min = 50, max = 150, step = 50))
  expect_equal(sizes1, c(50, 100, 150))
  
  # Test numeric vector input
  sizes2 <- process_sample_sizes(c(50, 100, 150))
  expect_equal(sizes2, c(50, 100, 150))
  
  # Test single value
  sizes3 <- process_sample_sizes(100)
  expect_equal(sizes3, 100)
  
  # Test invalid input
  expect_error(
    process_sample_sizes(list(min = 50, max = 150)),
    "must contain 'min', 'max', and 'step'"
  )
  
  expect_error(
    process_sample_sizes("invalid"),
    "must be either a numeric vector or list"
  )

  expect_error(
    process_sample_sizes(c(1, 25.5, NA)),
    "finite whole numbers greater than 1"
  )
})

test_that("validate_configuration checks input parameters", {
  # Valid configuration
  valid_config <- list(
    predictor_config = list(
      names = c("x1", "x2"),
      types = c("continuous", "continuous")
    ),
    data_generation = list(
      method = "joint_mvn",
      params = list()
    ),
    metrics = c("mse", "r_squared")
  )
  
  expect_true(validate_configuration(valid_config))
  
  # Missing predictor names
  invalid_config1 <- list(
    predictor_config = list(
      types = c("continuous", "continuous")
    )
  )
  expect_error(
    validate_configuration(invalid_config1),
    "must contain 'names' and 'types'"
  )
  
  # Mismatched lengths
  invalid_config2 <- list(
    predictor_config = list(
      names = c("x1", "x2"),
      types = c("continuous")
    )
  )
  expect_error(
    validate_configuration(invalid_config2),
    "Length of predictor names and types must match"
  )
  
  # Invalid data generation method
  invalid_config3 <- list(
    data_generation = list(
      method = "invalid_method"
    )
  )
  expect_error(
    validate_configuration(invalid_config3),
    "must be one of"
  )
  
  # Unknown metrics (should warn, not error)
  config_with_unknown <- list(
    metrics = c("mse", "unknown_metric")
  )
  expect_warning(
    validate_configuration(config_with_unknown),
    "Unknown metrics"
  )
})
