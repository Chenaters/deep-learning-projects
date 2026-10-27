# Tests for model_fitting.R
# Exercise branch behavior in model fitting, prediction, and evaluation flow.

library(testthat)

setup_test_data <- function(n = 60) {
  set.seed(123)
  data.frame(
    y = rnorm(n, 10, 2),
    x1 = rnorm(n, 50, 10),
    x2 = rnorm(n, 10, 3)
  )
}

test_that("normalize_model_type maps aliases to canonical names", {
  expect_equal(normalize_model_type("rf"), "rf")
  expect_equal(normalize_model_type("randomforest"), "rf")
  expect_equal(normalize_model_type("randomForest"), "rf")
  expect_equal(normalize_model_type("random_forest"), "rf")
  expect_equal(normalize_model_type("gradient_boosting"), "gbm")
  expect_equal(normalize_model_type("lmer"), "lmm")
  expect_equal(normalize_model_type("proportional_odds"), "ordinal")
  expect_equal(normalize_model_type("neural_network"), "nnet")
})

test_that("model registry defines model behavior and tuning defaults", {
  registry <- get_model_registry()

  expect_true(all(c(
    "model", "label", "category", "outcomes", "order_aware",
    "neural_network", "package", "available"
  ) %in% names(registry)))
  expect_true(all(c("gam", "ordinal", "nnet") %in% registry$model))
  expect_true(registry$order_aware[registry$model == "ordinal"])
  expect_true(registry$neural_network[registry$model == "nnet"])

  nnet_spec <- get_model_spec("mlp")
  expect_equal(nnet_spec$model, "nnet")
  expect_equal(nnet_spec$defaults$size, 5)
  expect_equal(
    resolve_model_options("nnet", list(size = 3))$size,
    3
  )
  expect_error(
    resolve_model_options("nnet", list(unknown = 1)),
    "Unknown model_options"
  )
})

test_that("fit_model works for core model families", {
  test_data <- setup_test_data()
  lm_model <- fit_model("lm", y ~ x1 + x2, test_data)
  expect_s3_class(lm_model, "lm")

  binary_data <- test_data
  binary_data$y <- rbinom(nrow(binary_data), 1, 0.5)
  glm_binom <- fit_model("glm", y ~ x1 + x2, binary_data)
  expect_s3_class(glm_binom, "glm")
  expect_equal(glm_binom$family$family, "binomial")

  count_data <- test_data
  count_data$y <- rpois(nrow(count_data), lambda = 3)
  glm_pois <- fit_model("glm", y ~ x1 + x2, count_data)
  expect_equal(glm_pois$family$family, "poisson")

  gauss_data <- test_data
  gauss_data$y <- rexp(nrow(gauss_data), rate = 0.5)
  glm_gauss <- fit_model("glm", y ~ x1 + x2, gauss_data)
  expect_equal(glm_gauss$family$family, "gaussian")
})

test_that("fit_model supports selected optional model aliases", {
  test_data <- setup_test_data()

  skip_on_ci()
  if (requireNamespace("randomForest", quietly = TRUE)) {
    rf_model <- fit_model("randomforest", y ~ x1 + x2, test_data)
    expect_s3_class(rf_model, "randomForest")

    rf_model_alias <- fit_model("randomForest", y ~ x1 + x2, test_data)
    expect_s3_class(rf_model_alias, "randomForest")
  }
  if (requireNamespace("gbm", quietly = TRUE)) {
    gbm_model <- fit_model("gradient_boosting", y ~ x1 + x2, test_data)
    expect_s3_class(gbm_model, "gbm")
  }
})

test_that("GAM captures smooth nonlinear effects", {
  skip_if_not_installed("mgcv")
  set.seed(42)
  nonlinear_data <- data.frame(
    x1 = seq(-2, 2, length.out = 80),
    x2 = rnorm(80)
  )
  nonlinear_data$y <- sin(2 * nonlinear_data$x1) + 0.1 * nonlinear_data$x2

  model <- fit_model(
    "gam", y ~ x1 + x2, nonlinear_data,
    model_options = list(k = 6)
  )
  predictions <- predict_model(model, "gam", nonlinear_data)

  expect_s3_class(model, "gam")
  expect_equal(get_model_type(model), "gam")
  expect_length(predictions, nrow(nonlinear_data))
  expect_false(anyNA(predictions))
})

test_that("ordinal model uses response category order", {
  set.seed(43)
  ordinal_data <- data.frame(x1 = rnorm(120), x2 = rnorm(120))
  latent <- ordinal_data$x1 - 0.4 * ordinal_data$x2 + rlogis(120)
  ordinal_data$y <- cut(
    latent,
    breaks = c(-Inf, -0.5, 0.5, Inf),
    labels = c("low", "medium", "high"),
    ordered_result = TRUE
  )

  model <- fit_model("ordinal", y ~ x1 + x2, ordinal_data)
  predictions <- predict_model(model, "ordinal", ordinal_data[1:10, ])

  expect_s3_class(model, "polr")
  expect_equal(get_model_type(model), "ordinal")
  expect_length(predictions, 10)
  expect_true(all(predictions >= 1 & predictions <= 3))
})

test_that("neural network supports regression with explicit options", {
  skip_if_not_installed("nnet")
  test_data <- setup_test_data(80)

  model <- fit_model(
    "nnet", y ~ x1 + x2, test_data,
    model_options = list(size = 3, maxit = 50, decay = 0.001)
  )
  predictions <- predict_model(model, "nnet", test_data[1:12, ])

  expect_s3_class(model, "nnet")
  expect_equal(get_model_type(model), "nnet")
  expect_length(predictions, 12)
  expect_false(anyNA(predictions))
})

test_that("predict_model returns correctly shaped outputs", {
  test_data <- setup_test_data()
  train_data <- test_data[1:45, ]
  test_subset <- test_data[46:60, ]

  lm_model <- fit_model("lm", y ~ x1 + x2, train_data)
  lm_preds <- predict_model(lm_model, "lm", test_subset)
  expect_type(lm_preds, "double")
  expect_length(lm_preds, 15)
  expect_false(any(is.na(lm_preds)))

  binary_data <- setup_test_data()
  binary_data$y <- rbinom(nrow(binary_data), 1, 0.5)
  glm_model <- fit_model("glm", y ~ x1 + x2, binary_data[1:45, ])
  glm_preds <- predict_model(glm_model, "glm", binary_data[46:60, ])
  expect_type(glm_preds, "double")
  expect_length(glm_preds, 15)
  expect_true(all(glm_preds >= 0 & glm_preds <= 1))
})

test_that("fit_and_evaluate_model covers split, in-sample, and CV modes", {
  test_data <- setup_test_data()

  split_results <- fit_and_evaluate_model(
    data = test_data,
    model_type = "lm",
    response_var = "y",
    predictor_vars = c("x1", "x2"),
    metrics = c("mse", "r_squared"),
    cv_folds = NULL,
    cv_repeats = 1
  )
  expect_true(all(c("mse", "r_squared") %in% names(split_results)))
  expect_false(is.na(split_results$mse))

  in_sample_results <- fit_and_evaluate_model(
    data = test_data,
    model_type = "lm",
    response_var = "y",
    predictor_vars = c("x1", "x2"),
    metrics = c("mse", "r_squared"),
    cv_folds = 0,
    cv_repeats = 1
  )
  expect_true(all(c("mse", "r_squared") %in% names(in_sample_results)))
  expect_false(is.na(in_sample_results$r_squared))

  cv_results <- fit_and_evaluate_model(
    data = test_data,
    model_type = "lm",
    response_var = "y",
    predictor_vars = c("x1", "x2"),
    metrics = c("mse"),
    cv_folds = 3,
    cv_repeats = 2
  )
  expect_true("mse" %in% names(cv_results))
  expect_false(is.na(cv_results$mse))
})

test_that("ordered validation is deterministic and forward-looking", {
  ordered_data <- data.frame(
    y = 1:60 + sin(1:60),
    x1 = 1:60,
    x2 = cos(1:60),
    time = 1:60
  )

  set.seed(1)
  first <- fit_and_evaluate_model(
    data = ordered_data,
    model_type = "lm",
    response_var = "y",
    predictor_vars = c("x1", "x2"),
    metrics = "mse",
    cv_folds = 3,
    cv_repeats = 5,
    validation_strategy = "ordered",
    order_var = "time"
  )
  set.seed(999)
  second <- fit_and_evaluate_model(
    data = ordered_data[nrow(ordered_data):1, ],
    model_type = "lm",
    response_var = "y",
    predictor_vars = c("x1", "x2"),
    metrics = "mse",
    cv_folds = 3,
    cv_repeats = 2,
    validation_strategy = "ordered",
    order_var = "time"
  )

  expect_equal(first$mse, second$mse)
  expect_true(is.finite(first$mse))
})

test_that("fit_evaluate_cv handles model fit failures gracefully", {
  test_data <- setup_test_data()

  expect_warning(
    results <- fit_evaluate_cv(
      data = test_data,
      model_type = "unsupported_model",
      formula_obj = y ~ x1 + x2,
      metrics = c("mse"),
      cv_folds = 3,
      cv_repeats = 1,
      response_var = "y"
    ),
    "Error in fold"
  )
  expect_true(is.na(results$mse))
})

test_that("fit_evaluate_split supports in-sample evaluation", {
  test_data <- setup_test_data()

  out_of_sample <- fit_evaluate_split(
    data = test_data,
    model_type = "lm",
    formula_obj = y ~ x1 + x2,
    metrics = c("mse", "r_squared"),
    response_var = "y",
    in_sample = FALSE
  )
  in_sample <- fit_evaluate_split(
    data = test_data,
    model_type = "lm",
    formula_obj = y ~ x1 + x2,
    metrics = c("mse", "r_squared"),
    response_var = "y",
    in_sample = TRUE
  )

  expect_true(all(c("mse", "r_squared") %in% names(out_of_sample)))
  expect_true(all(c("mse", "r_squared") %in% names(in_sample)))
  expect_false(is.na(in_sample$mse))
})

test_that("get_model_type identifies known and unknown classes", {
  test_data <- setup_test_data()
  model_lm <- lm(y ~ x1 + x2, data = test_data)
  expect_equal(get_model_type(model_lm), "lm")

  binary_data <- test_data
  binary_data$y <- rbinom(nrow(binary_data), 1, 0.5)
  model_glm <- glm(y ~ x1 + x2, data = binary_data, family = binomial())
  expect_equal(get_model_type(model_glm), "glm")

  expect_equal(get_model_type(list(foo = "bar")), "unknown")
})

test_that("fit_model and predict_model expose helpful errors", {
  test_data <- setup_test_data()
  expect_error(
    fit_model("unsupported_model", y ~ x1 + x2, test_data),
    "Unsupported model type"
  )

  model <- fit_model("lm", y ~ x1 + x2, test_data)
  bad_data <- data.frame(x1 = rnorm(10))
  expect_warning(
    preds <- predict_model(model, "lm", bad_data),
    "Prediction error"
  )
  expect_length(preds, 10)
  expect_true(all(is.na(preds)))
})
