# -----------------------------------------------------------------------------
# Unit tests for the internal helper gen_PMSE_sampsize()
# -----------------------------------------------------------------------------
local({
  f <- pmsesampling:::gen_PMSE_sampsize   # shorthand
  k <- 4                                   # predictors in toy model
  var <- 1                                 # σ²

  test_that("returns -1 when PMSE target is not attainable", {
    expect_identical(f(var, k, PMSE = 0.9), -1)       # PMSE < σ²
    expect_identical(f(var, k, PMSE = var), -1)       # PMSE == σ²
  })

  test_that("returns a numeric vector of feasible n when PMSE > σ²", {
    out <- f(var, k, PMSE = 1.2)
    expect_type(out, "double")
    expect_true(all(out > k + 2))          # theoretical lower bound
  })

  test_that("monotonicity: tighter PMSE ⇒ larger n", {
    n_loose <- min(f(var, k, PMSE = 1.4))  # easier goal
    n_tight <- min(f(var, k, PMSE = 1.1))  # harder goal
    expect_lt(n_loose, n_tight)
  })

  test_that("vectorised σ² input is accepted", {
    vec_out <- f(rep(var, 1), k, PMSE = 1.3)
    expect_length(vec_out, 1)
  })

  test_that("upper-bound search stops before hard ceiling", {
    # choose an extreme but valid PMSE close to σ²
    expect_error(f(var, k, PMSE = 1.0001), NA)  # should *not* error
  })

  test_that("closed-form solution satisfies the PMSE equation", {
    target <- 1.2
    n <- f(var, k, PMSE = target)
    calculated <- var * (n + 1) * (n - 2) / (n * (n - k - 2))
    expect_equal(calculated, target, tolerance = 1e-10)
  })
})
