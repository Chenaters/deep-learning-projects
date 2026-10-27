#' Define a predictive model for empirical PMSE calculations
#'
#' @description Creates an explicit model specification used by
#' [pmse_learning_curve()]. Built-in specifications are available for ordinary
#' least squares (`"linear"`), ridge regression (`"ridge"`), and
#' k-nearest-neighbor regression (`"knn"`). A `"custom"` specification adapts
#' any machine-learning implementation without making it a hard package
#' dependency.
#'
#' Each fitting function receives a numeric predictor matrix `x` and numeric
#' outcome vector `y`. Each prediction function receives the fitted object and
#' a numeric matrix `newx`, and must return one numeric prediction per row.
#'
#' @param method One of `"linear"`, `"ridge"`, `"knn"`, or `"custom"`.
#' @param name Optional unique display name. Custom models must have a name.
#' @param lambda Positive ridge penalty. Used only for `method = "ridge"`.
#' @param neighbors Positive integer number of neighbors. Used only for
#' `method = "knn"`.
#' @param fit For a custom model, a function with arguments `x` and `y`.
#' @param predict For a custom model, a function with arguments `object` and
#' `newx`.
#'
#' @return An object of class `pmse_model`.
#' @export
#'
#' @examples
#' linear <- pmse_model("linear")
#' ridge <- pmse_model("ridge", lambda = 0.5)
#' nearest <- pmse_model("knn", neighbors = 7)
pmse_model <- function(method = c("linear", "ridge", "knn", "custom"),
                       name = NULL,
                       lambda = 1,
                       neighbors = 5L,
                       fit = NULL,
                       predict = NULL) {
  method <- match.arg(method)

  if (method == "custom") {
    if (is.null(name) || length(name) != 1L || !nzchar(name)) {
      stop("A custom model requires a non-empty 'name'.", call. = FALSE)
    }
    if (!is.function(fit) || !is.function(predict)) {
      stop("A custom model requires both 'fit' and 'predict' functions.",
           call. = FALSE)
    }
    fit_function <- fit
    predict_function <- predict
    signature <- paste0("custom:", name)
    parameters <- list()
  } else if (method == "linear") {
    if (!is.null(fit) || !is.null(predict)) {
      stop("'fit' and 'predict' are only used by method = 'custom'.",
           call. = FALSE)
    }
    if (is.null(name)) name <- "linear"
    fit_function <- .fit_pmse_linear
    predict_function <- .predict_pmse_linear
    signature <- "linear"
    parameters <- list()
  } else if (method == "ridge") {
    if (!is.null(fit) || !is.null(predict)) {
      stop("'fit' and 'predict' are only used by method = 'custom'.",
           call. = FALSE)
    }
    .validate_positive(lambda, "lambda")
    if (is.null(name)) name <- sprintf("ridge(lambda=%s)", format(lambda))
    fit_function <- function(x, y) .fit_pmse_ridge(x, y, lambda = lambda)
    predict_function <- .predict_pmse_ridge
    signature <- sprintf("ridge:lambda=%s", format(lambda, digits = 17L))
    parameters <- list(lambda = lambda)
  } else {
    if (!is.null(fit) || !is.null(predict)) {
      stop("'fit' and 'predict' are only used by method = 'custom'.",
           call. = FALSE)
    }
    .validate_count(neighbors, "neighbors", minimum = 1L)
    if (is.null(name)) name <- sprintf("knn(k=%d)", as.integer(neighbors))
    fit_function <- function(x, y) {
      .fit_pmse_knn(x, y, neighbors = as.integer(neighbors))
    }
    predict_function <- .predict_pmse_knn
    signature <- sprintf("knn:k=%d", as.integer(neighbors))
    parameters <- list(neighbors = as.integer(neighbors))
  }

  if (length(name) != 1L || !is.character(name) || !nzchar(name)) {
    stop("'name' must be one non-empty character value.", call. = FALSE)
  }

  structure(
    list(
      name = name,
      method = method,
      signature = signature,
      parameters = parameters,
      fit = fit_function,
      predict = predict_function
    ),
    class = "pmse_model"
  )
}

.fit_pmse_linear <- function(x, y) {
  design <- cbind(`(Intercept)` = 1, x)
  fit <- stats::lm.fit(x = design, y = y)
  coefficients <- fit$coefficients
  coefficients[is.na(coefficients)] <- 0
  list(coefficients = coefficients)
}

.predict_pmse_linear <- function(object, newx) {
  as.numeric(cbind(`(Intercept)` = 1, newx) %*% object$coefficients)
}

.fit_pmse_ridge <- function(x, y, lambda) {
  center <- colMeans(x)
  scale <- apply(x, 2L, stats::sd)
  scale[!is.finite(scale) | scale == 0] <- 1
  standardized <- sweep(sweep(x, 2L, center, "-"), 2L, scale, "/")
  y_center <- mean(y)
  penalty <- diag(lambda, ncol(standardized))
  coefficients <- solve(
    crossprod(standardized) + penalty,
    crossprod(standardized, y - y_center)
  )
  list(
    coefficients = as.numeric(coefficients),
    center = center,
    scale = scale,
    y_center = y_center
  )
}

.predict_pmse_ridge <- function(object, newx) {
  standardized <- sweep(
    sweep(newx, 2L, object$center, "-"), 2L, object$scale, "/"
  )
  as.numeric(object$y_center + standardized %*% object$coefficients)
}

.fit_pmse_knn <- function(x, y, neighbors) {
  center <- colMeans(x)
  scale <- apply(x, 2L, stats::sd)
  scale[!is.finite(scale) | scale == 0] <- 1
  standardized <- sweep(sweep(x, 2L, center, "-"), 2L, scale, "/")
  list(
    x = standardized,
    y = y,
    neighbors = min(neighbors, nrow(x)),
    center = center,
    scale = scale
  )
}

.predict_pmse_knn <- function(object, newx) {
  newx <- sweep(sweep(newx, 2L, object$center, "-"), 2L, object$scale, "/")
  vapply(seq_len(nrow(newx)), function(i) {
    distances <- rowSums((object$x - matrix(
      newx[i, ], nrow = nrow(object$x), ncol = ncol(object$x), byrow = TRUE
    ))^2)
    nearest <- order(distances)[seq_len(object$neighbors)]
    mean(object$y[nearest])
  }, numeric(1L))
}

.normalize_pmse_models <- function(models) {
  if (inherits(models, "pmse_model")) models <- list(models)
  if (!is.list(models) || length(models) == 0L ||
      !all(vapply(models, inherits, logical(1L), what = "pmse_model"))) {
    stop("'models' must be a pmse_model object or a non-empty list of them.",
         call. = FALSE)
  }
  names <- vapply(models, `[[`, character(1L), "name")
  if (anyDuplicated(names)) {
    stop("Every model must have a unique name.", call. = FALSE)
  }
  models
}
