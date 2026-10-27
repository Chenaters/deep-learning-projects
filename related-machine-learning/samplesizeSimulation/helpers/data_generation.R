# Legacy source() entry point; implementation lives in R/.
local({
  source_files <- lapply(sys.frames(), function(frame) frame$ofile)
  source_files <- Filter(Negate(is.null), source_files)
  source_file <- if (length(source_files)) tail(source_files, 1)[[1]] else NULL
  if (is.null(source_file)) stop("Load this file with source().")
  implementation <- file.path(dirname(normalizePath(source_file)), "..", "R", "data_generation.R")
  sys.source(implementation, envir = parent.env(environment()))
})
