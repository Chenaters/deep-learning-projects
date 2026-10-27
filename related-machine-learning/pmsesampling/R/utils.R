#' Convert a correlation matrix to a covariance matrix
#'
#' Computes \eqn{\Sigma = D\,\mathrm{CORR}\,D} where
#' \eqn{D = \mathrm{diag}(\texttt{SD})}.
#'
#' @param CORR Square correlation matrix.
#' @param SD   Numeric vector of standard deviations (same length as `ncol(CORR)`).
#'
#' @return Covariance matrix of identical dimension to `CORR`.
#' @examples
#' C <- matrix(c(1, .3, .3, 1), 2)
#' Corr_to_Cov(C, SD = c(2, 1))
#' @keywords internal
#' @noRd
Corr_to_Cov <- function(CORR, SD) {
  sweep(sweep(CORR, 1, SD, "*"), 2, SD, "*")
}

.validate_count <- function(x, name, minimum = 0L) {
  if (length(x) != 1L || !is.numeric(x) || !is.finite(x) ||
      x != floor(x) || x < minimum || x > .Machine$integer.max) {
    stop(sprintf(
      "'%s' must be one integer between %d and .Machine$integer.max.",
      name, minimum
    ), call. = FALSE)
  }
  invisible(TRUE)
}

.validate_positive <- function(x, name) {
  if (length(x) != 1L || !is.numeric(x) || !is.finite(x) || x <= 0) {
    stop(sprintf("'%s' must be one positive finite number.", name),
         call. = FALSE)
  }
  invisible(TRUE)
}

.validate_probability <- function(x, name) {
  if (length(x) != 1L || !is.numeric(x) || !is.finite(x) || x <= 0 || x >= 1) {
    stop(sprintf("'%s' must be strictly between zero and one.", name),
         call. = FALSE)
  }
  invisible(TRUE)
}

.validate_r_squared <- function(x, name) {
  if (length(x) != 1L || !is.numeric(x) || !is.finite(x) || x < 0 || x >= 1) {
    stop(sprintf("'%s' must be in [0, 1).", name), call. = FALSE)
  }
  invisible(TRUE)
}

.validate_covariance_input <- function(x, k, name) {
  if (!is.matrix(x) || !is.numeric(x) ||
      !identical(dim(x), rep(as.integer(k + 1L), 2L))) {
    stop(sprintf("'%s' must be a numeric (k + 1) by (k + 1) matrix.", name),
         call. = FALSE)
  }
  if (any(!is.finite(x))) {
    stop(sprintf("'%s' cannot contain missing or infinite values.", name),
         call. = FALSE)
  }
  tolerance <- sqrt(.Machine$double.eps) * max(1, max(abs(x)))
  if (max(abs(x - t(x))) > tolerance) {
    stop(sprintf("'%s' must be symmetric.", name), call. = FALSE)
  }
  invisible(TRUE)
}

#' Error-term variance for the full regression model
#'
#' Calculates \eqn{\sigma_k^2 = \sigma_{00} -
#'   \boldsymbol{\sigma}^\top \Sigma^{-1}\boldsymbol{\sigma}}.
#'
#' @param COV Covariance matrix whose first row/column correspond to the
#'   response followed by the \eqn{k} predictors.
#'
#' @return Numeric scalar (`sigma_k2`).
#' @keywords internal
#' @noRd
sigmak2 <- function(COV){
  sigma_00 <- COV[1, 1]
  sig      <- as.matrix(COV[-1, 1])
  SIG      <- COV[-1, -1]
  as.numeric(sigma_00 - crossprod(sig, solve(SIG, sig)))
}

#' Error-term variance for the reduced regression model
#'
#' Computes \eqn{\sigma_p^2} when only the first \eqn{p} of \eqn{k}
#' predictors are retained.
#'
#' @param COV Covariance matrix of the response and all \eqn{k} predictors.
#' @param p   Number of predictors in the reduced model.
#' @param k   Total number of predictors in the full model (unused inside).
#'
#' @return Numeric scalar (`sigma_p2`).
#' @keywords internal
#' @noRd
sigmap2 <- function(COV, p, k){
  sigma_00 <- COV[1, 1]
  if (p == 0L) return(as.numeric(sigma_00))
  sig      <- as.matrix(COV[-1, 1])
  SIG      <- COV[-1, -1]
  sig_1    <- sig[1:p, , drop = FALSE]
  SIG_11   <- SIG[1:p, 1:p, drop = FALSE]
  as.numeric(sigma_00 - crossprod(sig_1, solve(SIG_11, sig_1)))
}

#' Cohen’s *f*² from two *R*² values
#'
#' Computes \eqn{f^2 = (R^2 - R_1^2)/(1 - R^2)}.
#'
#' @param R2   *R*-squared for the full model.
#' @param R2_1 *R*-squared for the reduced model.
#'
#' @return Numeric scalar (`f2`).
#' @keywords internal
#' @noRd
f2_2_calcu <- function(R2, R2_1){
  (R2 - R2_1) / (1 - R2)
}

#' Efficient sample size for a target pPMSEr fraction
#'
#' Finds the minimum \eqn{n} that achieves a fraction `alpha`
#' of the maximal predictive-MSE reduction.
#'
#' @param sigma_k2 Error variance of the full model.
#' @param sigma_p2 Error variance of the reduced model.
#' @param k        Number of predictors in the full model.
#' @param p        Number of predictors in the reduced model.
#' @param alpha    Fraction of the maximum pPMSEr.
#'
#' @return Numeric scalar (`n_star`).
#' @keywords internal
#' @noRd
gen_efficient_sampsize <- function(sigma_k2, sigma_p2, k, p, alpha = 0.1){
  EVR         <- sigma_k2 / sigma_p2
  if (!is.finite(EVR) || EVR >= 1) {
    stop("No efficiency gain (sigma_k2 >= sigma_p2).")
  }
  lambda_star <- 1 + alpha * (1 / EVR - 1)
  p2          <- k - p
  p + 2 + (p2 * lambda_star / (lambda_star - 1))
}

#' Sample size required to achieve a target PMSE
#'
#' Solves the PMSE equation directly.  Writing
#' \eqn{r = \mathrm{PMSE}/\sigma^2}, the feasible sample size is the larger
#' root of
#' \deqn{(r - 1)n^2 + \{1-r(k+2)\}n + 2 = 0.}
#' This avoids repeated numerical root searches.
#'
#' @param variance Error variance (\eqn{\sigma^2}).
#' @param k        Number of predictors in the model.
#' @param PMSE     Target PMSE.
#'
#' @return Numeric scalar containing the feasible sample size,
#'   or \code{-1} if the target is unattainable (`PMSE <= variance`).
#' @keywords internal
#' @noRd
gen_PMSE_sampsize <- function(variance, k, PMSE){
  variance <- as.numeric(variance)
  if (length(variance) != 1L || !is.finite(variance) || variance <= 0) {
    stop("'variance' must be one positive finite number.", call. = FALSE)
  }
  if (length(PMSE) != 1L || !is.finite(PMSE) || PMSE <= 0) {
    stop("'PMSE' must be one positive finite number.", call. = FALSE)
  }
  if (PMSE <= variance) return(-1)

  ratio <- PMSE / variance
  a <- ratio - 1
  b <- 1 - ratio * (k + 2)
  discriminant <- b^2 - 8 * a
  root <- (-b + sqrt(discriminant)) / (2 * a)

  if (!is.finite(root) || root <= k + 2) {
    stop("No feasible sample size for these inputs.", call. = FALSE)
  }
  as.numeric(root)
}
