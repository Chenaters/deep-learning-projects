# Tests for metrics_calculation.R
# Test performance metrics calculation

library(testthat)

# Source helpers if not already loaded
if (!exists("calculate_metrics")) {
  source("helpers/metrics_calculation.R")
}

test_that("calculate_metrics computes MSE correctly", {
  y_true <- c(1, 2, 3, 4, 5)
  y_pred <- c(1.1, 2.2, 2.9, 4.1, 5.0)
  
  results <- calculate_metrics(y_true, y_pred, metrics = c("mse"))
  
  expect_type(results, "list")
  expect_true("mse" %in% names(results))
  
  expected_mse <- mean((y_true - y_pred)^2)
  expect_equal(results$mse, expected_mse)
})

test_that("calculate_metrics computes RMSE correctly", {
  y_true <- c(1, 2, 3, 4, 5)
  y_pred <- c(1.1, 2.2, 2.9, 4.1, 5.0)
  
  results <- calculate_metrics(y_true, y_pred, metrics = c("rmse"))
  
  expected_rmse <- sqrt(mean((y_true - y_pred)^2))
  expect_equal(results$rmse, expected_rmse)
})

test_that("calculate_metrics computes MAE correctly", {
  y_true <- c(1, 2, 3, 4, 5)
  y_pred <- c(1.1, 2.2, 2.9, 4.1, 5.0)
  
  results <- calculate_metrics(y_true, y_pred, metrics = c("mae"))
  
  expected_mae <- mean(abs(y_true - y_pred))
  expect_equal(results$mae, expected_mae)
})

test_that("calculate_metrics computes R-squared correctly", {
  y_true <- c(1, 2, 3, 4, 5)
  y_pred <- c(1.1, 2.2, 2.9, 4.1, 5.0)
  
  results <- calculate_metrics(y_true, y_pred, metrics = c("r_squared"))
  
  ss_res <- sum((y_true - y_pred)^2)
  ss_tot <- sum((y_true - mean(y_true))^2)
  expected_r2 <- 1 - ss_res / ss_tot
  
  expect_equal(results$r_squared, expected_r2)
})

test_that("calculate_metrics handles perfect predictions", {
  y_true <- c(1, 2, 3, 4, 5)
  y_pred <- c(1, 2, 3, 4, 5)
  
  results <- calculate_metrics(y_true, y_pred, 
                               metrics = c("mse", "rmse", "mae", "r_squared"))
  
  expect_equal(results$mse, 0)
  expect_equal(results$rmse, 0)
  expect_equal(results$mae, 0)
  expect_equal(results$r_squared, 1)
})

test_that("calculate_metrics computes AUC for binary outcomes", {
  skip_if_not_installed("pROC")
  
  set.seed(123)
  y_true <- c(rep(0, 50), rep(1, 50))
  y_pred <- c(rnorm(50, 0.3, 0.2), rnorm(50, 0.7, 0.2))
  y_pred <- pmax(0, pmin(1, y_pred))  # Bound between 0 and 1
  
  results <- calculate_metrics(y_true, y_pred, metrics = c("auc"))
  
  expect_true("auc" %in% names(results))
  expect_gte(results$auc, 0)
  expect_lte(results$auc, 1)
})

test_that("calculate_auc_manual computes AUC correctly", {
  # Perfect separation
  y_true <- c(0, 0, 0, 1, 1, 1)
  y_pred <- c(0.1, 0.2, 0.3, 0.7, 0.8, 0.9)
  
  auc <- calculate_auc_manual(y_true, y_pred)
  expect_equal(auc, 1.0)
  
  # Random predictions
  set.seed(123)
  y_true <- sample(c(0, 1), 100, replace = TRUE)
  y_pred <- runif(100)
  
  auc <- calculate_auc_manual(y_true, y_pred)
  expect_gte(auc, 0)
  expect_lte(auc, 1)
})

test_that("calculate_metrics computes accuracy correctly", {
  y_true <- c(0, 0, 1, 1, 0, 1)
  y_pred <- c(0.2, 0.3, 0.7, 0.8, 0.4, 0.6)
  
  results <- calculate_metrics(y_true, y_pred, metrics = c("accuracy"))
  
  # Using threshold 0.5
  y_pred_class <- ifelse(y_pred > 0.5, 1, 0)
  expected_accuracy <- mean(y_true == y_pred_class)
  
  expect_equal(results$accuracy, expected_accuracy)
})

test_that("calculate_metrics computes sensitivity and specificity", {
  y_true <- c(0, 0, 0, 1, 1, 1)
  y_pred <- c(0.2, 0.3, 0.7, 0.7, 0.8, 0.9)
  
  results <- calculate_metrics(y_true, y_pred, 
                               metrics = c("sensitivity", "specificity"))
  
  expect_true("sensitivity" %in% names(results))
  expect_true("specificity" %in% names(results))
  expect_gte(results$sensitivity, 0)
  expect_lte(results$sensitivity, 1)
  expect_gte(results$specificity, 0)
  expect_lte(results$specificity, 1)
})

test_that("calculate_metrics computes F1 score", {
  y_true <- c(0, 0, 0, 1, 1, 1)
  y_pred <- c(0.2, 0.3, 0.4, 0.7, 0.8, 0.9)
  
  results <- calculate_metrics(y_true, y_pred, metrics = c("f1"))
  
  expect_true("f1" %in% names(results))
  expect_gte(results$f1, 0)
  expect_lte(results$f1, 1)
})

test_that("calculate_metrics handles NA values", {
  y_true <- c(1, 2, NA, 4, 5)
  y_pred <- c(1.1, 2.2, 2.9, NA, 5.0)
  
  results <- calculate_metrics(y_true, y_pred, metrics = c("mse", "r_squared"))
  
  # Should calculate on valid values only
  expect_false(is.na(results$mse))
  expect_false(is.na(results$r_squared))
})

test_that("calculate_metrics handles all NA values", {
  y_true <- c(NA, NA, NA)
  y_pred <- c(NA, NA, NA)
  
  expect_warning(
    results <- calculate_metrics(y_true, y_pred, metrics = c("mse")),
    "No valid observations"
  )
  
  expect_true(is.na(results$mse))
})

test_that("calculate_confusion_matrix computes correctly", {
  y_true <- c(0, 0, 0, 1, 1, 1)
  y_pred <- c(0.2, 0.3, 0.7, 0.7, 0.8, 0.9)
  
  cm <- calculate_confusion_matrix(y_true, y_pred, threshold = 0.5)
  
  expect_type(cm, "list")
  expect_true(all(c("tp", "tn", "fp", "fn") %in% names(cm)))
  expect_equal(cm$tp + cm$tn + cm$fp + cm$fn, length(y_true))
})

test_that("calculate_brier_score computes correctly", {
  y_true <- c(0, 0, 1, 1)
  y_pred <- c(0.2, 0.3, 0.7, 0.8)
  
  brier <- calculate_brier_score(y_true, y_pred)
  
  expected_brier <- mean((y_true - y_pred)^2)
  expect_equal(brier, expected_brier)
})

test_that("calculate_calibration creates bins correctly", {
  set.seed(123)
  y_true <- sample(c(0, 1), 100, replace = TRUE)
  y_pred <- runif(100)
  
  calibration <- calculate_calibration(y_true, y_pred, n_bins = 10)
  
  expect_s3_class(calibration, "data.frame")
  expect_true("bin" %in% names(calibration))
})

test_that("calculate_metrics handles multiple metrics simultaneously", {
  y_true <- c(1, 2, 3, 4, 5)
  y_pred <- c(1.1, 2.2, 2.9, 4.1, 5.0)
  
  results <- calculate_metrics(
    y_true, 
    y_pred, 
    metrics = c("mse", "rmse", "mae", "r_squared")
  )
  
  expect_length(results, 4)
  expect_true(all(c("mse", "rmse", "mae", "r_squared") %in% names(results)))
  expect_false(any(is.na(unlist(results))))
})

test_that("calculate_auc handles edge cases", {
  # All same class
  y_true <- c(1, 1, 1, 1)
  y_pred <- c(0.5, 0.6, 0.7, 0.8)
  
  auc <- calculate_auc_manual(y_true, y_pred)
  expect_true(is.na(auc))
  
  # Perfect separation
  y_true <- c(0, 0, 1, 1)
  y_pred <- c(0.1, 0.2, 0.9, 0.8)
  
  auc <- calculate_auc_manual(y_true, y_pred)
  expect_equal(auc, 1.0)
})
