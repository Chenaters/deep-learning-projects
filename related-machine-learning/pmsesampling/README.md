# PMSE Sample Size Planning

Analytic nested-OLS sample-size planning and empirical learning curves for OLS, ridge, k-nearest-neighbor and custom models.

## Run locally

From this folder, with R installed:

```sh
Rscript -e "install.packages('testthat', repos='https://cloud.r-project.org')"
Rscript -e "testthat::test_local('.')"
R CMD build .
```

The project API and examples are in [the original README](docs/ORIGINAL_README.md).

## Publication and reproducibility

Existing authorship and license declarations are preserved. No new CRAN availability or approval is claimed. Only the standalone package is included; the surrounding research archive, correspondence, thesis copies, private history and generated check directories are excluded. Install optional dependencies from DESCRIPTION for optional-model tests.

This is a source-only publication copy prepared on October 26, 2026. Datasets,
trained weights, notebook outputs, local environment files, and Git history are
excluded. Preparation and validation details are in `PUBLICATION_NOTES.md`.
