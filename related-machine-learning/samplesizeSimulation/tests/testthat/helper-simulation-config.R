base_joint_args <- function() {
  list(
    data_generation = list(
      method = "joint_mvn",
      params = list(
        correlation_matrix = matrix(c(1, 0.25, 0.25, 1), nrow = 2),
        mean_vector = c(0, 0)
      )
    ),
    predictor_config = list(
      names = c("x1", "x2"),
      types = c("continuous", "continuous"),
      effects = list(
        fixed = c(0.5, -0.3),
        residual_sd = 1.0
      )
    ),
    reference_model = "lm",
    target_models = c("lm"),
    sample_sizes = c(30, 40),
    n_simulations = 1,
    metrics = c("mse", "r_squared"),
    efficiency_threshold = 0.95,
    cv_folds = NULL,
    cv_repeats = 1,
    infinite_n = NULL,
    seed = 123,
    output_path = NULL,
    verbose = FALSE
  )
}

run_with_overrides <- function(overrides = list()) {
  do.call(run_general_simulation, modifyList(base_joint_args(), overrides, keep.null = TRUE))
}

