# Installed-package tests inherit the package namespace. Standalone test_dir()
# loads the same canonical R sources into its test environment.
if (!exists("run_general_simulation", mode = "function")) {
  candidates <- unique(c(getwd(), dirname(getwd()), dirname(dirname(getwd()))))
  roots <- candidates[file.exists(file.path(candidates, "R", "simulation.R"))]
  if (!length(roots)) stop("Could not locate package R sources for tests.")
  files <- c("utilities.R", "storage_utils.R", "data_generation.R",
             "model_fitting.R", "metrics_calculation.R",
             "results_processing.R", "simulation.R")
  for (file in files) {
    sys.source(file.path(roots[1], "R", file), envir = environment())
  }
}
