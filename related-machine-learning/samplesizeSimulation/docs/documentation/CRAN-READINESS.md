# CRAN submission record

Prepared on 2026-10-13 for `samplesizeSimulation` 0.1.0, maintained by Louis
Chen (`chenaters@gmail.com`). License: GPL-3; Louis Chen is the sole author and
copyright holder as confirmed by the maintainer.

## Verified locally

| Check | Result |
| --- | --- |
| Source build, including executable HTML vignette | Passed |
| Windows 11 x64, R 4.6.1: `R CMD check --as-cran --no-manual` | 0 errors, 0 warnings, 1 note |
| The note | `New submission` in CRAN incoming feasibility |
| Full local test suite with `NOT_CRAN=true` | 321 passed assertions, 0 failures, 0 warnings, 0 skipped tests |
| Installed-package tests under CRAN checking | 313 passed assertions, 0 failures, 0 warnings, 2 skipped tests |
| Executable examples and vignette rebuild | Passed |
| Namespace, S3 registration, dependency declarations, help/usage consistency | Passed |
| README/NEWS validation with local Pandoc | Passed |
| Legacy `source("functionSamplesizeSimu.R")` smoke test | Loaded and simulated successfully |
| Case-insensitive name search | No matches in current CRAN, CRAN Archive, or the four queried Bioconductor release indexes |

The two skipped CRAN tests are existing `skip_on_cran()` file save/load tests.
They pass in the full local suite; an integration save test also runs during
CRAN checking. No optional learner was skipped in the full local run.

The final built source archive is
`../.cran-work/samplesizeSimulation_0.1.0.tar.gz` (54,003 bytes). Check evidence:

- `../.cran-work/samplesizeSimulation.Rcheck/00check.log`
- `../.cran-work/samplesizeSimulation.Rcheck/tests/testthat.Rout`
- `../.cran-work/tests.log`
- `../.cran-work/legacy-smoke.log`
- `../.cran-work/name-check.log`

The workspace-local dependency library, Pandoc installation, generated build
outputs, and logs are excluded by `.gitignore` and `.Rbuildignore`. The source
archive contains package code, documentation, examples in help files, tests,
and the vignette; study scripts and historical notes are excluded.

## Work completed

The source-script framework was converted into a standard package with
`DESCRIPTION`, `NAMESPACE`, `R/`, generated `man/` help, registered print/summary
methods, an executable self-contained HTML vignette, release notes, and a
reproducible check command. Existing entry points load the canonical code from
`R/`. The old vignette is retained as historical documentation.

Correctness repairs include input validation, sorted sample sizes, handling of
missing and deteriorating efficiency references, binary metric edge cases,
Brier scores, removal of the falsely labeled adjusted R-squared placeholder,
linear-memory rank-based AUC, caller RNG restoration, plot-setting restoration,
mixed-model group identifiers and response-scale prediction, ridge/lasso
prediction matrices, invalidation of failed CV repeats, and visible failed
simulation counts. Method changes invalidate old cache keys.

The method documentation explicitly distinguishes simulation percentiles from
confidence limits, finite reference runs from a known asymptote, and relative
efficiency from formal power or a sequential stopping test. Ordered validation
with empirical/synthetic resampling is blocked because resampled rows do not
establish independent chronological validation. Random empirical bootstrap
scores retain that method's plug-in-population interpretation and limitation.

## Required release verification still outstanding

1. Run the final source archive through `R CMD check --as-cran` on Linux or
   macOS, and a current R-devel environment. These checks were not performed.
   An existing Ubuntu WSL distribution was inspected but contains no R binary.
2. Run a check with PDF manual generation enabled on a machine with LaTeX.
   Local checks used `--no-manual` because LaTeX is unavailable. Rd syntax,
   cross-references, widths, usage and executable examples were checked.
3. Review the method description and exploratory sizing convention against the
   intended paper/reference. The code checks establish software consistency,
   not the statistical adequacy of the method for every deployment population.
4. Refresh the name check and `cran-comments.md` with the final external check
   evidence immediately before submission.

CRAN expects portable code and normally requires packages to run on at least
two major R platforms; examples and checks should be economical. See the
[CRAN Repository Policy](https://cran.r-project.org/web/packages/policies.html)
and [submission checklist](https://cran.r-project.org/web/packages/submission_checklist.html).
The local result is a submission candidate, not a guarantee of CRAN acceptance.
No package was submitted, no check service was emailed, and no remote code was
published during this work.

## Reproduce

From the package source directory, with declared dependencies available:

```sh
Rscript tools/check-package.R --no-manual
```

On a machine with LaTeX, omit `--no-manual`. The helper builds into `.cran-work`
and runs `R CMD check --as-cran`. It uses the workspace-local R library and
Pandoc if present. Documentation generation during development requires
`roxygen2`; generated help files and the namespace are already supplied, so
users do not need it to install the package.

To exercise the CRAN-skipped tests as well:

```r
Sys.setenv(NOT_CRAN = "true")
testthat::test_dir("tests/testthat")
```

Image and momentum applications are assessed separately in
[EXTENSIONS.md](EXTENSIONS.md). Neither has been validated on real task data.
