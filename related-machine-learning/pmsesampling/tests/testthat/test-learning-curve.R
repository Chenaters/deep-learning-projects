test_that("built-in model specifications are explicit and validated", {
  expect_s3_class(pmse_model("linear"), "pmse_model")
  expect_equal(pmse_model("ridge", lambda = 0.5)$parameters$lambda, 0.5)
  expect_equal(pmse_model("knn", neighbors = 3)$parameters$neighbors, 3L)
  expect_error(pmse_model("ridge", lambda = 0), "positive")
  expect_error(pmse_model("custom", name = "incomplete"), "requires both")
})

test_that("OLS, ridge, and KNN produce a reusable comparison table", {
  models <- list(
    pmse_model("linear"),
    pmse_model("ridge", lambda = 0.5),
    pmse_model("knn", neighbors = 3)
  )
  curve <- pmse_learning_curve(
    mtcars,
    outcome = "mpg",
    predictors = c("wt", "hp", "disp"),
    models = models,
    sample_sizes = c(12, 20),
    repeats = 3,
    seed = 42
  )

  expect_s3_class(curve, "pmse_learning_curve")
  expect_equal(nrow(curve), 6)
  expect_setequal(curve$method, c("linear", "ridge", "knn"))
  expect_true(all(is.finite(curve$mean_pmse)))
  expect_true(all(curve$failures == 0))

  extended <- pmse_learning_curve(
    mtcars,
    outcome = "mpg",
    predictors = c("wt", "hp", "disp"),
    models = models,
    sample_sizes = c(12, 20, 24),
    repeats = 3,
    seed = 42,
    previous = curve
  )
  expect_equal(nrow(extended), 9)
  cached <- extended[extended$n %in% c(12, 20), names(curve)]
  cached <- as.data.frame(cached)
  expected <- as.data.frame(curve)
  rownames(cached) <- NULL
  rownames(expected) <- NULL
  expect_equal(cached, expected)
})

test_that("previous table skips model fits already computed", {
  calls <- 0L
  mean_model <- pmse_model(
    "custom",
    name = "training mean",
    fit = function(x, y) {
      calls <<- calls + 1L
      mean(y)
    },
    predict = function(object, newx) rep(object, nrow(newx))
  )
  first <- pmse_learning_curve(
    mtcars,
    outcome = "mpg",
    predictors = "wt",
    models = mean_model,
    sample_sizes = 15,
    repeats = 2,
    seed = 7
  )
  expect_equal(calls, 2L)

  second <- pmse_learning_curve(
    mtcars,
    outcome = "mpg",
    predictors = "wt",
    models = mean_model,
    sample_sizes = 15,
    repeats = 2,
    seed = 7,
    previous = first
  )
  expect_equal(calls, 2L)
  expect_equal(second, first)
})

test_that("sample-size targets are read from tables without fitting", {
  curve <- data.frame(
    model = rep("ridge", 3),
    method = rep("ridge", 3),
    signature = rep("ridge:lambda=1", 3),
    n = c(50, 100, 150),
    mean_pmse = c(5, 3, 2),
    se_pmse = c(0.2, 0.1, 0.1)
  )

  grid_result <- pmse_table_samplesize(
    curve, target_pmse = 4, metric = "mean", monotone = FALSE
  )
  expect_equal(grid_result$required_n, 100)

  interpolated <- pmse_table_samplesize(
    curve, target_pmse = 4, metric = "mean", monotone = FALSE,
    interpolate = TRUE
  )
  expect_equal(interpolated$required_n, 75)
})
