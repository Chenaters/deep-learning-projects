# Run from the package root: Rscript tools/check-package.R
# Additional R CMD check arguments can be supplied, e.g. --no-manual.
root <- normalizePath(getwd(), mustWork = TRUE)
if (!file.exists(file.path(root, "DESCRIPTION"))) {
  stop("Run this script from the package source directory")
}
work <- file.path(root, ".cran-work")
dir.create(work, showWarnings = FALSE)
local_lib <- file.path(work, "library")
if (dir.exists(local_lib)) {
  .libPaths(c(local_lib, .libPaths()))
  Sys.setenv(R_LIBS_USER = local_lib)
}
pandoc_exe <- file.path(work, "python", "pypandoc", "files", "pandoc.exe")
if (file.exists(pandoc_exe)) {
  Sys.setenv(PATH = paste(dirname(pandoc_exe), Sys.getenv("PATH"),
                         sep = .Platform$path.sep))
}
r <- file.path(R.home("bin"), if (.Platform$OS.type == "windows") "R.exe" else "R")
run_check <- function() {
  old_dir <- setwd(work)
  on.exit(setwd(old_dir), add = TRUE)
  build_status <- system2(r, c("CMD", "build", shQuote(root)))
  if (build_status != 0) stop("Source build failed")
  metadata <- read.dcf(file.path(root, "DESCRIPTION"))
  archive <- paste0(metadata[1, "Package"], "_", metadata[1, "Version"], ".tar.gz")
  check_status <- system2(r, c("CMD", "check", "--as-cran",
                              commandArgs(trailingOnly = TRUE), shQuote(archive)))
  if (check_status != 0) stop("R CMD check reported errors or warnings; inspect .cran-work")
}
run_check()
