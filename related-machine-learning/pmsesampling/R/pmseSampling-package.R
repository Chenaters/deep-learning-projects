#' pmsesampling: Sample Size Determination for Predictive Models
#'
#' Tools to estimate the minimum sample size required to achieve a target
#' prediction mean-squared error (PMSE). Nested ordinary least-squares models
#' use the analytic criteria described by Ma (2023). Ridge,
#' nearest-neighbor, and user-defined machine-learning models use empirical
#' learning curves that can be saved and reused as lookup tables.
#'
#' @section Core functions:
#' \describe{
#'   \item{\code{pmse_samplesize()}}{Analytic sample size for nested OLS models.}
#'   \item{\code{pmse_model()}}{Explicit built-in or custom model specification.}
#'   \item{\code{pmse_learning_curve()}}{Empirical PMSE table across models and
#'   training sizes.}
#'   \item{\code{pmse_table_samplesize()}}{Fast sample-size lookup from a saved
#'   learning-curve table.}
#' }
#'
#' @section Typical workflow:
#' 1. Obtain \eqn{\sigma_k^2} and \eqn{\sigma_p^2}
#' 2. Or import or build a predictor covariance matrix.\cr
#' 3. Or obtain Cohen's \eqn{f^2} and \eqn{R^2}.
#' 4. Use \code{pmse_samplesize()} for OLS or build a reusable empirical table
#' with \code{pmse_learning_curve()} for machine-learning models.\cr
#' @references
#' Ma Y. (2023) *Predictive Power and Efficient Sample Size in
#' Linear Regression Models*. Worcester Polytechnic Institute
#'
#' @docType package
#' @name pmsesampling-package
#'
"_PACKAGE"
