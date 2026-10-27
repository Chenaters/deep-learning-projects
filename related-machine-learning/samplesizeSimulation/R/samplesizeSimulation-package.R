#' Simulation Based Sample Size Planning
#'
#' Compare prediction models across simulated sample sizes. Recommendations
#' describe relative performance under the supplied assumptions, rather than
#' formal power or a guarantee of external predictive performance.
#'
#' @importFrom stats as.formula lm glm predict sd quantile median rnorm runif
#'   rbinom plogis rpois rgamma aggregate
#' @importFrom utils modifyList write.csv
#' @importFrom graphics par plot polygon axis mtext abline text legend
#' @importFrom grDevices rgb
#' @keywords internal
"_PACKAGE"
