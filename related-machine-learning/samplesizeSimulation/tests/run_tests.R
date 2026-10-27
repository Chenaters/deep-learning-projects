# Test runner for the sample size simulation framework
# Run all tests with: testthat::test_dir("tests/testthat")

library(testthat)

# Set up test directory relative to this script
script_arg <- commandArgs(trailingOnly = FALSE)
script_file <- sub("^--file=", "", script_arg[grepl("^--file=", script_arg)])
script_file <- if (length(script_file) > 0) {
  normalizePath(script_file[1], mustWork = FALSE)
} else {
  ""
}
tests_dir <- if (nzchar(script_file)) {
  file.path(dirname(script_file), "testthat")
} else {
  file.path("tests", "testthat")
}
tests_dir <- normalizePath(tests_dir, mustWork = TRUE)

# Run tests; helper-load.R handles sourcing framework files
testthat::test_dir(tests_dir)
