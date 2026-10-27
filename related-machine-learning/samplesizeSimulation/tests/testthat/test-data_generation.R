# Tests for data_generation.R
# Test data generation methods

library(testthat)

# Source helpers if not already loaded
if (!exists("generate_data")) {
  source("helpers/utilities.R")
  source("helpers/data_generation.R")
}

test_that("generate_mvn_data creates correct data structure", {
  skip_if_not_installed("MASS")
  
  corr_matrix <- matrix(c(
    1.0, 0.3,
    0.3, 1.0
  ), nrow = 2, byrow = TRUE)
  
  data <- generate_mvn_data(
    n = 100,
    params = list(
      correlation_matrix = corr_matrix,
      mean_vector = c(50, 10),
      sd_vector = c(10, 3)
    ),
    predictor_config = list(
      names = c("x1", "x2"),
      types = c("continuous", "continuous"),
      effects = list(
        fixed = c(0.5, -0.3),
        residual_sd = 1.0
      )
    ),
    reference_model = "lm"
  )
  
  # Check structure
  expect_s3_class(data, "data.frame")
  expect_equal(nrow(data), 100)
  expect_true(all(c("y", "x1", "x2") %in% names(data)))
  
  # Check approximate means (with tolerance for sampling variability)
  expect_equal(mean(data$x1), 50, tolerance = 5)
  expect_equal(mean(data$x2), 10, tolerance = 2)
})

test_that("generate_mvn_data uses covariance matrix", {
  skip_if_not_installed("MASS")
  
  cov_matrix <- matrix(c(
    100, 15,
    15, 9
  ), nrow = 2, byrow = TRUE)
  
  data <- generate_mvn_data(
    n = 100,
    params = list(
      covariance_matrix = cov_matrix,
      mean_vector = c(50, 10)
    ),
    predictor_config = list(
      names = c("x1", "x2"),
      effects = list(
        fixed = c(0.5, -0.3),
        residual_sd = 1.0
      )
    ),
    reference_model = "lm"
  )
  
  expect_s3_class(data, "data.frame")
  expect_equal(nrow(data), 100)
})

test_that("generate_outcome creates outcomes for different models", {
  X <- matrix(c(1, 2, 3, 4, 5, 6), nrow = 3, ncol = 2)
  colnames(X) <- c("x1", "x2")
  
  predictor_config <- list(
    effects = list(
      fixed = c(0.5, -0.3),
      residual_sd = 1.0
    )
  )
  
  # Linear model
  y_lm <- generate_outcome(X, predictor_config, "lm")
  expect_type(y_lm, "double")
  expect_length(y_lm, 3)
  
  # GLM - binomial
  predictor_config$effects$family <- "binomial"
  y_glm <- generate_outcome(X, predictor_config, "glm")
  expect_true(all(y_glm %in% c(0, 1)))
  
  # GLM - poisson
  predictor_config$effects$family <- "poisson"
  y_pois <- generate_outcome(X, predictor_config, "glm")
  expect_true(all(y_pois >= 0))
  expect_true(all(y_pois == floor(y_pois)))

  # Ordinal proportional-odds outcome
  predictor_config$effects$thresholds <- c(-0.5, 0.5)
  predictor_config$effects$labels <- c("low", "medium", "high")
  y_ordinal <- generate_outcome(X, predictor_config, "ordinal")
  expect_true(is.ordered(y_ordinal))
  expect_equal(levels(y_ordinal), c("low", "medium", "high"))
})

test_that("parametric generation preserves ordered outcome type", {
  generated <- generate_parametric_data(
    n = 80,
    params = list(
      x1 = list(distribution = "normal", mean = 0, sd = 1),
      x2 = list(distribution = "normal", mean = 0, sd = 1)
    ),
    predictor_config = list(
      names = c("x1", "x2"),
      types = c("continuous", "continuous"),
      effects = list(
        fixed = c(0.8, -0.4),
        thresholds = c(-0.75, 0.75),
        labels = c("low", "middle", "high")
      )
    ),
    reference_model = "ordinal"
  )

  expect_true(is.ordered(generated$y))
  expect_equal(levels(generated$y), c("low", "middle", "high"))
})

test_that("get_effect_sizes handles fixed and ranges", {
  # Fixed effects
  config1 <- list(
    effects = list(
      fixed = c(0.5, -0.3, 0.2)
    )
  )
  effects1 <- get_effect_sizes(config1)
  expect_equal(effects1, c(0.5, -0.3, 0.2))
  
  # Range effects
  set.seed(123)
  config2 <- list(
    effects = list(
      ranges = list(
        x1 = c(0.3, 0.7),
        x2 = c(-0.5, -0.1),
        x3 = c(0.1, 0.4)
      )
    )
  )
  effects2 <- get_effect_sizes(config2)
  expect_length(effects2, 3)
  expect_true(effects2[1] >= 0.3 && effects2[1] <= 0.7)
  expect_true(effects2[2] >= -0.5 && effects2[2] <= -0.1)
  
  # Error when neither provided
  config3 <- list(effects = list())
  expect_error(
    get_effect_sizes(config3),
    "Must provide either fixed effects or ranges"
  )
})

test_that("generate_parametric_data creates data from distributions", {
  data <- generate_parametric_data(
    n = 100,
    params = list(
      x1 = list(distribution = "normal", mean = 50, sd = 10),
      x2 = list(distribution = "uniform", min = 0, max = 20)
    ),
    predictor_config = list(
      names = c("x1", "x2"),
      types = c("continuous", "continuous"),
      effects = list(
        fixed = c(0.5, -0.3),
        residual_sd = 1.0
      )
    ),
    reference_model = "lm"
  )
  
  expect_s3_class(data, "data.frame")
  expect_equal(nrow(data), 100)
  expect_true(all(c("y", "x1", "x2") %in% names(data)))
  
  # Check x2 is within uniform bounds
  expect_true(all(data$x2 >= 0 & data$x2 <= 20))
})

test_that("resample_empirical_data bootstraps correctly", {
  empirical_data <- data.frame(
    y = 1:200,
    x1 = rnorm(200),
    x2 = rnorm(200)
  )
  
  resampled <- resample_empirical_data(
    n = 50,
    empirical_data = empirical_data,
    predictor_config = list(
      names = c("x1", "x2"),
      regenerate_outcome = FALSE
    ),
    reference_model = "lm"
  )
  
  expect_s3_class(resampled, "data.frame")
  expect_equal(nrow(resampled), 50)
  expect_true(all(c("y", "x1", "x2") %in% names(resampled)))
})

test_that("generate_data dispatches to correct method", {
  skip_if_not_installed("MASS")
  
  corr_matrix <- matrix(c(1, 0.3, 0.3, 1), nrow = 2)
  
  # Test MVN method
  data_mvn <- generate_data(
    n = 50,
    data_generation = list(
      method = "joint_mvn",
      params = list(correlation_matrix = corr_matrix)
    ),
    predictor_config = list(
      names = c("x1", "x2"),
      effects = list(fixed = c(0.5, -0.3), residual_sd = 1),
      types = c("continuous", "continuous")
    ),
    reference_model = "lm",
    seed = 123
  )
  expect_s3_class(data_mvn, "data.frame")
  
  # Test parametric method
  data_param <- generate_data(
    n = 50,
    data_generation = list(
      method = "parametric",
      params = list(
        x1 = list(distribution = "normal", mean = 0, sd = 1),
        x2 = list(distribution = "normal", mean = 0, sd = 1)
      )
    ),
    predictor_config = list(
      names = c("x1", "x2"),
      effects = list(fixed = c(0.5, -0.3), residual_sd = 1),
      types = c("continuous", "continuous")
    ),
    reference_model = "lm",
    seed = 123
  )
  expect_s3_class(data_param, "data.frame")
  
  # Test empirical method
  empirical <- data.frame(y = rnorm(100), x1 = rnorm(100), x2 = rnorm(100))
  data_emp <- generate_data(
    n = 50,
    data_generation = list(
      method = "empirical",
      params = list(empirical_data = empirical)
    ),
    predictor_config = list(
      names = c("x1", "x2"),
      regenerate_outcome = FALSE
    ),
    reference_model = "lm",
    seed = 123
  )
  expect_s3_class(data_emp, "data.frame")
  
  # Test invalid method
  expect_error(
    generate_data(
      n = 50,
      data_generation = list(method = "invalid_method"),
      predictor_config = list(),
      reference_model = "lm",
      seed = 123
    ),
    "Unknown data generation method"
  )
})

test_that("MVN generation requires correlation or covariance matrix", {
  expect_error(
    generate_mvn_data(
      n = 50,
      params = list(),
      predictor_config = list(
        names = c("x1", "x2"),
        effects = list(fixed = c(0.5, -0.3))
      ),
      reference_model = "lm"
    ),
    "Must provide either correlation_matrix or covariance_matrix"
  )
})
