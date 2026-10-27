# Model catalog, machine-learning options, ordered validation, and table reuse

source("functionSamplesizeSimu.R")

# Inspect definitions before choosing models.
print(get_model_registry())
print(get_model_spec("nnet"))

base_arguments <- list(
  data_generation = list(
    method = "parametric",
    params = list(
      x1 = list(distribution = "normal", mean = 0, sd = 1),
      x2 = list(distribution = "uniform", min = -2, max = 2)
    )
  ),
  predictor_config = list(
    names = c("x1", "x2"),
    types = c("continuous", "continuous"),
    effects = list(fixed = c(0.8, -0.4), residual_sd = 1)
  ),
  reference_model = "lm",
  target_models = c("lm", "gam", "nnet"),
  model_options = list(
    gam = list(k = 6),
    nnet = list(size = 3, decay = 0.001, maxit = 100)
  ),
  n_simulations = 25,
  metrics = c("mse", "mae"),
  cv_folds = 5,
  cv_repeats = 1,
  infinite_n = 500,
  seed = 2025,
  verbose = TRUE
)

# First run computes all requested rows.
first <- do.call(
  run_general_simulation,
  c(base_arguments, list(sample_sizes = c(50, 100, 150)))
)

# The second run reuses 50/100/150 and computes only the new n = 200 rows.
second <- do.call(
  run_general_simulation,
  c(
    base_arguments,
    list(
      sample_sizes = c(50, 100, 150, 200),
      lookup_table = first$simulation_table
    )
  )
)

print(second$cache)
print(recommend_from_simulation_table(second$simulation_table, 0.95))

# Persist a growing lookup table if desired.
combined_table <- merge_simulation_tables(
  first$simulation_table,
  second$simulation_table
)
utils::write.csv(combined_table, "simulation_lookup.csv", row.names = FALSE)
