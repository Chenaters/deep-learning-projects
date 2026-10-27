# Legacy source() entry point. Prefer library(samplesizeSimulation) after installation.
local({
  source_files <- lapply(sys.frames(), function(frame) frame$ofile)
  source_files <- Filter(Negate(is.null), source_files)
  source_file <- if (length(source_files)) tail(source_files, 1)[[1]] else NULL
  if (is.null(source_file)) stop("Load this file with source().")
  package_dir <- dirname(normalizePath(source_file))
  files <- c("utilities.R", "storage_utils.R", "data_generation.R",
             "model_fitting.R", "metrics_calculation.R",
             "results_processing.R", "simulation.R")
  for (file in files) {
    sys.source(file.path(package_dir, "R", file),
               envir = parent.env(environment()))
  }
})
