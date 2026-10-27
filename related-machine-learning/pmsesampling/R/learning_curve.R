#' Build a reusable PMSE learning-curve table
#'
#' @description Empirically evaluates one or more predictive models at selected
#' training sample sizes against one fixed holdout set. The returned data frame
#' is both a compact comparison of the models and a lookup table for
#' [pmse_table_samplesize()]. Pass an earlier table through `previous` to compute
#' only missing model/sample-size combinations.
#'
#' @details This function is intended for models that are not covered by the
#' OLS-only formula in [pmse_samplesize()]. All models use the same model matrix,
#' holdout rows, and repeated training subsets for a fair comparison. Character
#' and factor predictors are encoded once with `model.matrix()`.
#'
#' A previous table is safe to reuse only when the data and model definitions
#' have not changed. Metadata are checked, but the data values themselves are
#' not copied into or hashed inside the table.
#'
#' @param data A data frame containing the outcome and predictors.
#' @param outcome Name of one numeric outcome column.
#' @param predictors Character vector of predictor names. Defaults to every
#' column except `outcome`.
#' @param models A [pmse_model()] object or list of model specifications.
#' Defaults to OLS and ridge regression.
#' @param sample_sizes Integer training sample sizes to evaluate. By default, up
#' to eight approximately evenly spaced values are used.
#' @param repeats Number of repeated training subsets per sample size.
#' @param test_fraction Fraction of rows reserved once as a shared holdout set.
#' @param seed Integer seed used for the holdout and training subsets.
#' @param previous Optional table returned by an earlier compatible call. Rows
#' already represented by the same model signature and sample size are reused.
#'
#' @return A data frame of class `pmse_learning_curve` with one row per model and
#' training sample size. `mean_pmse`, `sd_pmse`, and `se_pmse` summarize the
#' held-out prediction errors; `failures` counts failed fits.
#' @export
#'
#' @examples
#' data(mtcars)
#' curve <- pmse_learning_curve(
#'   mtcars,
#'   outcome = "mpg",
#'   predictors = c("wt", "hp", "disp"),
#'   models = list(pmse_model("linear"), pmse_model("ridge")),
#'   sample_sizes = c(12, 18),
#'   repeats = 2,
#'   seed = 42
#' )
#' curve
pmse_learning_curve <- function(data,
                                outcome,
                                predictors = NULL,
                                models = list(
                                  pmse_model("linear"),
                                  pmse_model("ridge")
                                ),
                                sample_sizes = NULL,
                                repeats = 10L,
                                test_fraction = 0.2,
                                seed = 1L,
                                previous = NULL) {
  if (!is.data.frame(data) || nrow(data) < 4L) {
    stop("'data' must be a data frame with at least four rows.", call. = FALSE)
  }
  if (length(outcome) != 1L || !is.character(outcome) ||
      !outcome %in% names(data)) {
    stop("'outcome' must name one column in 'data'.", call. = FALSE)
  }
  if (!is.numeric(data[[outcome]])) {
    stop("The outcome column must be numeric for PMSE calculation.",
         call. = FALSE)
  }
  if (is.null(predictors)) predictors <- setdiff(names(data), outcome)
  if (!is.character(predictors) || length(predictors) == 0L ||
      anyDuplicated(predictors) || any(!predictors %in% names(data)) ||
      outcome %in% predictors) {
    stop("'predictors' must contain unique predictor columns from 'data'.",
         call. = FALSE)
  }
  if (any(!stats::complete.cases(data[c(outcome, predictors)]))) {
    stop("Outcome and predictor columns cannot contain missing values.",
         call. = FALSE)
  }
  .validate_count(repeats, "repeats", minimum = 1L)
  .validate_probability(test_fraction, "test_fraction")
  .validate_count(seed, "seed", minimum = 0L)
  models <- .normalize_pmse_models(models)

  formula <- stats::reformulate(predictors)
  x <- stats::model.matrix(formula, data = data)
  x <- x[, colnames(x) != "(Intercept)", drop = FALSE]
  if (ncol(x) == 0L || any(!is.finite(x))) {
    stop("Predictors must produce a finite, non-empty numeric model matrix.",
         call. = FALSE)
  }
  y <- as.numeric(data[[outcome]])
  if (any(!is.finite(y))) {
    stop("The outcome cannot contain infinite values.", call. = FALSE)
  }

  test_n <- max(1L, floor(nrow(data) * test_fraction))
  test_n <- min(test_n, nrow(data) - 2L)
  available_n <- nrow(data) - test_n
  if (is.null(sample_sizes)) {
    first_n <- min(available_n, max(5L, ncol(x) + 2L))
    sample_sizes <- unique(as.integer(round(seq(
      first_n, available_n, length.out = min(8L, available_n - first_n + 1L)
    ))))
  }
  if (!is.numeric(sample_sizes) || length(sample_sizes) == 0L ||
      any(!is.finite(sample_sizes)) || any(sample_sizes != floor(sample_sizes)) ||
      any(sample_sizes < 2L) || any(sample_sizes > available_n)) {
    stop(sprintf(
      "'sample_sizes' must be integers between 2 and %d for this holdout split.",
      available_n
    ), call. = FALSE)
  }
  sample_sizes <- sort(unique(as.integer(sample_sizes)))

  predictor_key <- paste(predictors, collapse = ", ")
  metadata <- list(
    outcome = outcome,
    predictors = predictor_key,
    test_fraction = test_fraction,
    seed = as.integer(seed),
    data_rows = nrow(data),
    test_n = test_n,
    repeats = as.integer(repeats)
  )
  previous <- .validate_previous_curve(previous, metadata)

  old_seed_exists <- exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  if (old_seed_exists) old_seed <- get(".Random.seed", envir = .GlobalEnv)
  on.exit({
    if (old_seed_exists) {
      assign(".Random.seed", old_seed, envir = .GlobalEnv)
    } else if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
      rm(".Random.seed", envir = .GlobalEnv)
    }
  }, add = TRUE)
  set.seed(as.integer(seed))

  test_rows <- sample.int(nrow(data), size = test_n)
  training_pool <- setdiff(seq_len(nrow(data)), test_rows)
  permutations <- replicate(
    as.integer(repeats), sample(training_pool), simplify = FALSE
  )
  training_sets <- lapply(sample_sizes, function(n) {
    lapply(permutations, function(index) index[seq_len(n)])
  })
  names(training_sets) <- as.character(sample_sizes)

  rows <- list()
  row_index <- 0L
  for (model in models) {
    for (n in sample_sizes) {
      cached <- !is.null(previous) && any(
        previous$model == model$name &
          previous$signature == model$signature &
          previous$n == n
      )
      if (cached) next

      errors <- rep(NA_real_, repeats)
      for (repeat_index in seq_len(repeats)) {
        train_rows <- training_sets[[as.character(n)]][[repeat_index]]
        set.seed(.pmse_fit_seed(seed, model$signature, n, repeat_index))
        errors[repeat_index] <- tryCatch({
          object <- model$fit(
            x[train_rows, , drop = FALSE],
            y[train_rows]
          )
          prediction <- model$predict(object, x[test_rows, , drop = FALSE])
          if (!is.numeric(prediction) || length(prediction) != test_n ||
              any(!is.finite(prediction))) {
            stop("model returned invalid predictions")
          }
          mean((prediction - y[test_rows])^2)
        }, error = function(error) NA_real_)
      }

      successful <- sum(is.finite(errors))
      mean_pmse <- if (successful > 0L) mean(errors, na.rm = TRUE) else NA_real_
      sd_pmse <- if (successful > 1L) stats::sd(errors, na.rm = TRUE) else 0
      se_pmse <- if (successful > 0L) sd_pmse / sqrt(successful) else NA_real_
      row_index <- row_index + 1L
      rows[[row_index]] <- data.frame(
        model = model$name,
        method = model$method,
        signature = model$signature,
        n = as.integer(n),
        mean_pmse = mean_pmse,
        sd_pmse = sd_pmse,
        se_pmse = se_pmse,
        repeats = as.integer(repeats),
        failures = as.integer(repeats - successful),
        test_n = as.integer(test_n),
        test_fraction = test_fraction,
        seed = as.integer(seed),
        data_rows = as.integer(nrow(data)),
        outcome = outcome,
        predictors = predictor_key,
        stringsAsFactors = FALSE
      )
    }
  }

  computed <- if (length(rows)) do.call(rbind, rows) else NULL
  result <- if (is.null(previous)) {
    computed
  } else if (is.null(computed)) {
    as.data.frame(previous)
  } else {
    rbind(as.data.frame(previous), computed)
  }
  ordering <- order(result$model, result$n)
  result <- result[ordering, , drop = FALSE]
  rownames(result) <- NULL
  class(result) <- c("pmse_learning_curve", "data.frame")
  result
}

.validate_previous_curve <- function(previous, metadata) {
  if (is.null(previous)) return(NULL)
  required <- c(
    "model", "method", "signature", "n", "mean_pmse", "sd_pmse",
    "se_pmse", "repeats", "failures", "test_n", "test_fraction", "seed",
    "data_rows", "outcome", "predictors"
  )
  if (!is.data.frame(previous) || !all(required %in% names(previous))) {
    stop("'previous' must be a table returned by pmse_learning_curve().",
         call. = FALSE)
  }
  cache_key <- paste(previous$model, previous$signature, previous$n, sep = "\r")
  if (anyNA(cache_key) || anyDuplicated(cache_key)) {
    stop("'previous' contains missing or duplicate model/sample-size rows.",
         call. = FALSE)
  }
  checks <- c(
    outcome = isTRUE(all(previous$outcome == metadata$outcome)),
    predictors = isTRUE(all(previous$predictors == metadata$predictors)),
    test_fraction = isTRUE(all(previous$test_fraction == metadata$test_fraction)),
    seed = isTRUE(all(previous$seed == metadata$seed)),
    data_rows = isTRUE(all(previous$data_rows == metadata$data_rows)),
    test_n = isTRUE(all(previous$test_n == metadata$test_n)),
    repeats = isTRUE(all(previous$repeats == metadata$repeats))
  )
  if (!all(checks)) {
    mismatch <- paste(names(checks)[!checks], collapse = ", ")
    stop(sprintf("'previous' is incompatible: %s changed.", mismatch),
         call. = FALSE)
  }
  previous
}

.pmse_fit_seed <- function(seed, signature, n, repeat_index) {
  codes <- utf8ToInt(signature)
  signature_value <- sum(as.double(codes) * seq_along(codes))
  value <- (as.double(seed) + 104729 * n + 1009 * repeat_index +
              signature_value) %% 2147483646
  as.integer(value + 1)
}

#' Read ML sample-size targets from a learning-curve table
#'
#' @description Estimates the smallest sample size at which each model reaches
#' a target held-out PMSE. This is a table-only operation: models are not fitted
#' again, so the same learning-curve table can answer many target scenarios
#' cheaply. Table rows with failed fits are excluded from the lookup.
#'
#' @param table A table returned by [pmse_learning_curve()].
#' @param target_pmse Positive target prediction mean-squared error.
#' @param metric Use the one-sided upper confidence bound (`"upper"`) or the
#' mean held-out PMSE (`"mean"`).
#' @param confidence Confidence level used by `metric = "upper"`.
#' @param monotone If `TRUE`, apply decreasing isotonic regression to stabilize
#' noisy empirical learning curves before lookup.
#' @param interpolate If `TRUE`, linearly interpolate between the last failing
#' and first passing table rows, then round up. If `FALSE`, return the first
#' evaluated sample size that passes.
#'
#' @return A data frame with one row per model and its estimated required sample
#' size. `NA` indicates that the target was not reached in the table.
#' @export
#'
#' @examples
#' curve <- data.frame(
#'   model = rep("ridge", 3),
#'   method = rep("ridge", 3),
#'   signature = rep("ridge:lambda=1", 3),
#'   n = c(50, 100, 150),
#'   mean_pmse = c(5, 3, 2),
#'   se_pmse = c(0.2, 0.1, 0.1)
#' )
#' pmse_table_samplesize(curve, target_pmse = 4, metric = "mean")
pmse_table_samplesize <- function(table,
                                  target_pmse,
                                  metric = c("upper", "mean"),
                                  confidence = 0.95,
                                  monotone = TRUE,
                                  interpolate = FALSE) {
  metric <- match.arg(metric)
  .validate_positive(target_pmse, "target_pmse")
  .validate_probability(confidence, "confidence")
  if (length(monotone) != 1L || !is.logical(monotone) || is.na(monotone) ||
      length(interpolate) != 1L || !is.logical(interpolate) ||
      is.na(interpolate)) {
    stop("'monotone' and 'interpolate' must each be TRUE or FALSE.",
         call. = FALSE)
  }
  required <- c("model", "method", "signature", "n", "mean_pmse", "se_pmse")
  if (!is.data.frame(table) || nrow(table) == 0L ||
      !all(required %in% names(table))) {
    stop("'table' must be a learning-curve table with PMSE summary columns.",
         call. = FALSE)
  }
  if (!all(vapply(table[c("n", "mean_pmse", "se_pmse")],
                  is.numeric, logical(1L)))) {
    stop("The table's 'n', 'mean_pmse', and 'se_pmse' columns must be numeric.",
         call. = FALSE)
  }
  keys <- unique(table[c("model", "method", "signature")])
  rows <- lapply(seq_len(nrow(keys)), function(i) {
    model_rows <- table[
      table$model == keys$model[i] & table$signature == keys$signature[i],
      , drop = FALSE
    ]
    model_rows <- model_rows[order(model_rows$n), , drop = FALSE]
    if (anyDuplicated(model_rows$n)) {
      stop(sprintf("Model '%s' has duplicate sample-size rows.", keys$model[i]),
           call. = FALSE)
    }
    criterion <- model_rows$mean_pmse
    if (metric == "upper") {
      criterion <- criterion + stats::qnorm(confidence) * model_rows$se_pmse
    }
    keep <- is.finite(criterion) & is.finite(model_rows$n)
    if ("failures" %in% names(model_rows)) {
      keep <- keep & is.finite(model_rows$failures) & model_rows$failures == 0L
    }
    criterion <- criterion[keep]
    n <- model_rows$n[keep]
    if (length(n) && monotone) {
      criterion <- -stats::isoreg(n, -criterion)$yf
    }

    passing <- which(criterion <= target_pmse)
    if (!length(passing)) {
      required_n <- NA_real_
      estimated_pmse <- if (length(criterion)) {
        criterion[length(criterion)]
      } else {
        NA_real_
      }
      status <- "target not reached"
    } else {
      first <- passing[1L]
      required_n <- n[first]
      estimated_pmse <- criterion[first]
      status <- "target reached"
      if (interpolate && first > 1L && criterion[first - 1L] != criterion[first]) {
        estimate <- n[first - 1L] +
          (target_pmse - criterion[first - 1L]) *
          (n[first] - n[first - 1L]) /
          (criterion[first] - criterion[first - 1L])
        required_n <- ceiling(estimate)
        estimated_pmse <- target_pmse
        status <- "target reached (interpolated)"
      }
    }
    data.frame(
      model = keys$model[i],
      method = keys$method[i],
      target_pmse = target_pmse,
      metric = metric,
      required_n = as.numeric(required_n),
      estimated_pmse = estimated_pmse,
      status = status,
      stringsAsFactors = FALSE
    )
  })
  result <- do.call(rbind, rows)
  rownames(result) <- NULL
  result
}
