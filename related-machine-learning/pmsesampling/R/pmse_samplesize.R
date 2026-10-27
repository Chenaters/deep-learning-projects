#' Analytic sample size for nested ordinary least-squares models
#'
#' @title Compute an OLS sample size under user-defined PMSE targets
#'
#' @description Computes the sample size for two nested ordinary least-squares
#' (OLS) models. The function implements the formulas in "Predictive Power and
#' Efficient Sample Size in Linear Regression Models" by Yifan Ma (2023).
#'
#' @details This analytic result is specific to nested OLS models under the
#' assumptions in Ma (2023); it is not an approximation for arbitrary machine
#' learning algorithms. For ridge, nearest-neighbor, or custom predictive
#' models, use [pmse_learning_curve()] and obtain sample sizes from its reusable
#' result table with [pmse_table_samplesize()].
#'
#' The response must be in the first row and column of `cov` or `corr`. The
#' next `p` variables define the reduced model and all `k` variables define the
#' full model. When residual variances are supplied directly, matrix inputs are
#' not needed. When only an incremental Cohen's \eqn{f^2} (or the two
#' \eqn{R^2} values used to derive it) is available, only the efficiency
#' criterion can be calculated.
#'
#'
#' @param k Integer. Total number of predictors in the full model.
#'
#' @param p Integer. Number of basic predictors in the reduced model.
#'
#' @param PMSE_val_k Numeric. Target PMSE value for the full model.
#'
#' @param PMSE_val_p Numeric. Target PMSE value for the reduced model.
#'
#' @param efficiency_level Numeric. Target efficiency level.
#' (default is 0.9, meaning 90% of asymptotic pPMSEr)
#'
#' @param sigma_k2 Numeric. Predictor error variance for full model. If 'NULL'
#' it is derived.
#'
#' @param sigma_p2 Numeric. Predictor error variance for basic model. If 'NULL'
#' it is derived.
#'
#' @param cov Optional covariance matrix. Must be `(k+1) x (k+1)` with the response
#' 1st row and column.
#'
#' @param corr Optional correlation matrix. (Same layout as `cov`).
#'
#' @param SD Optional numeric vector of standard deviation for the predictors when
#' a correlation matrix is supplied. Default `1`
#'
#' @param f2 Deprecated placeholder for Cohen's f2 for the full model. It is
#' retained for backward compatibility but is not used by the nested-model
#' efficiency formula.
#'
#' @param f2_2 Numeric. Cohen’s f2 for the effects of new predictors given
#' the basic model.
#'
#' @param R2_full Numeric. Coefficient of determination for full model.
#'
#' @param R2_basic Numeric. Coefficient of determination for basic model.
#'
#' @return A numeric scalar representing the required sample size. The value is
#' not rounded; use `ceiling()` when an integer recruitment target is needed.
#'
#' @references
#' Ma, Y. (2023). _Predictive Power and Efficient Sample Size in Linear
#' Regression Models_. Master’s Thesis, Worcester Polytechnic Institute.
#'
#' @examples
#' ## Example with a 5-predictor model (k = 5) and 2 basic predictors (p = 2)
#' pmse_samplesize(
#'   k = 5, p = 2,
#'   PMSE_val_k    = 1,
#'   PMSE_val_p    = 1,
#'   efficiency_level = 0.9,
#'   sigma_k2 = 0.50,
#'   sigma_p2 = 0.60
#' )
#'
#' @export
pmse_samplesize <- function(k, p,
                           PMSE_val_k = 1,
                           PMSE_val_p = 1,
                           efficiency_level = 0.9,
                           sigma_k2 = NULL,
                           sigma_p2 = NULL,
                           cov = NULL,
                           corr = NULL,
                           SD = 1,
                           f2 = NULL,
                           f2_2 = NULL,
                           R2_full = NULL,
                           R2_basic = NULL) {

  .validate_count(k, "k", minimum = 1L)
  .validate_count(p, "p", minimum = 0L)
  if (p >= k) {
    stop("'p' must be smaller than 'k' for nested reduced and full models.",
         call. = FALSE)
  }
  .validate_probability(efficiency_level, "efficiency_level")
  .validate_positive(PMSE_val_k, "PMSE_val_k")
  .validate_positive(PMSE_val_p, "PMSE_val_p")

  if (!is.null(cov) && !is.null(corr)) {
    stop("Supply only one of 'cov' and 'corr'.", call. = FALSE)
  }
  if (!is.null(corr)) {
    corr <- as.matrix(corr)
    .validate_covariance_input(corr, k, "corr")
    if (length(SD) == 1L) SD <- rep(SD, k + 1L)
    if (!is.numeric(SD) || length(SD) != k + 1L ||
        any(!is.finite(SD)) || any(SD <= 0)) {
      stop("'SD' must contain one positive finite value or k + 1 values.",
           call. = FALSE)
    }
    cov <- Corr_to_Cov(corr, SD)
  }
  if (!is.null(cov)) {
    cov <- as.matrix(cov)
    .validate_covariance_input(cov, k, "cov")
    if (is.null(sigma_k2)) sigma_k2 <- sigmak2(cov)
    if (is.null(sigma_p2)) sigma_p2 <- sigmap2(cov, p, k)
  }

  if (is.null(f2_2) && !is.null(R2_full) && !is.null(R2_basic)) {
    .validate_r_squared(R2_full, "R2_full")
    .validate_r_squared(R2_basic, "R2_basic")
    if (R2_basic > R2_full) {
      stop("'R2_basic' cannot exceed 'R2_full'.", call. = FALSE)
    }
    f2_2 <- f2_2_calcu(R2_full, R2_basic)
  }

  # With an incremental effect size alone, the absolute PMSE targets cannot be
  # evaluated. Scaling sigma_k2 to one leaves the efficiency calculation
  # unchanged and makes this limited return path explicit.
  if (!is.null(f2_2) && is.null(sigma_k2) && is.null(sigma_p2)) {
    .validate_positive(f2_2, "f2_2")
    return(gen_efficient_sampsize(
      sigma_k2 = 1,
      sigma_p2 = 1 + f2_2,
      k = k,
      p = p,
      alpha = 1 - efficiency_level
    ))
  }

  if (is.null(sigma_k2) || is.null(sigma_p2)) {
    stop("Cannot determine both OLS residual variances.", call. = FALSE)
  }
  .validate_positive(sigma_k2, "sigma_k2")
  .validate_positive(sigma_p2, "sigma_p2")
  if (sigma_k2 >= sigma_p2) {
    stop("'sigma_k2' must be smaller than 'sigma_p2' to define an efficiency gain.",
         call. = FALSE)
  }

  pmse_k <- gen_PMSE_sampsize(sigma_k2, k, PMSE = PMSE_val_k)
  pmse_p <- gen_PMSE_sampsize(sigma_p2, p, PMSE = PMSE_val_p)
  if (identical(pmse_k, -1)) {
    stop("PMSE target for the full model must exceed 'sigma_k2'.", call. = FALSE)
  }
  if (identical(pmse_p, -1)) {
    stop("PMSE target for the reduced model must exceed 'sigma_p2'.", call. = FALSE)
  }
  efficiency_n <- gen_efficient_sampsize(
    sigma_k2, sigma_p2, k, p, 1 - efficiency_level
  )

  max(pmse_k, pmse_p, efficiency_n)
}



