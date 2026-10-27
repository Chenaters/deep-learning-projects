# Sample Size Simulation Framework - Helper Functions

> Implementations now live in `../R/`. The `.R` files in this directory are
> compatibility loaders for existing study scripts. Use
> `library(samplesizeSimulation)` for the installed package interface.

This directory contains modular helper functions for the general sample size simulation framework.

## File Structure

```
helpers/
├── utilities.R              # General utility functions
├── storage_utils.R          # Results storage and file I/O
├── data_generation.R        # Data generation methods
├── model_fitting.R          # Model fitting and prediction
├── metrics_calculation.R    # Performance metrics
├── results_processing.R     # Post-processing and plotting
└── validation.R             # Validation and cross-validation
```

## Module Descriptions

### utilities.R
General utility functions used across the framework:
- `%||%`: Null-coalescing operator
- `check_packages()`: Verify required packages are installed
- `generate_timestamp()`: Create timestamp strings for file naming
- `derive_simulation_seed()`: Stable independent seeds for data and model streams
- `process_sample_sizes()`: Convert sample size specifications
- `validate_configuration()`: Validate input parameters
- `create_progress_bar()`: Create progress indicators

### storage_utils.R
Functions for managing simulation results storage:
- `initialize_results_storage()`: Create storage structure for all results
- `initialize_sim_storage()`: Storage for single sample size simulations
- `aggregate_results()`: Aggregate statistics across simulations
- `results_to_simulation_table()`: Flatten results into a reusable keyed table
- `restore_results_from_table()`: Skip exact model/sample-size cache hits
- `recommend_from_simulation_table()`: Read recommendations without refitting
- `merge_simulation_tables()`: Maintain a de-duplicated lookup table
- `save_simulation_results()`: Save results to CSV and RDS files
- `load_simulation_results()`: Load previously saved results

### data_generation.R
Data generation methods:
- `generate_data()`: Main dispatcher for different data generation methods
- `generate_mvn_data()`: Multivariate normal distribution
- `generate_synthetic_data()`: Synthpop-based generation
- `generate_parametric_data()`: Parametric distributions
- `resample_empirical_data()`: Bootstrap resampling
- `generate_outcome()`: Create outcome variables from reference models
- `get_effect_sizes()`: Extract or sample effect sizes

### model_fitting.R
Model fitting and prediction:
- `fit_and_evaluate_model()`: Main function for fitting and evaluation
- `get_model_registry()`: Canonical model catalog and dependency metadata
- `get_model_spec()`: Definition and defaults for one model
- `resolve_model_options()`: Validate explicit tuning overrides
- `fit_evaluate_split()`: Train/test split evaluation
- `fit_evaluate_cv()`: Cross-validation evaluation
- `fit_model()`: Fit various model types (lm, glm, rf, gbm, lmm, etc.)
- `predict_model()`: Generate predictions from fitted models
- `get_model_type()`: Identify model class

Supported models:
- Linear models: `lm`, `glm`
- Tree-based: `rf` (random forest), `gbm` (gradient boosting), `cart`
- Nonlinear smooth: `gam`
- Ordered outcome: `ordinal` (proportional odds)
- Neural network: `nnet` (one hidden layer)
- Mixed models: `lmm`, `glmm`
- Other: `svm`, `ridge`, `lasso`

### metrics_calculation.R
Performance metrics calculation:
- `calculate_metrics()`: Main dispatcher for metric calculation
- `calculate_auc()`: Area under ROC curve
- `calculate_auc_manual()`: Manual AUC calculation (fallback)
- `calculate_confusion_matrix()`: Classification metrics
- `calculate_brier_score()`: Brier score for probabilistic predictions
- `calculate_calibration()`: Calibration statistics

Supported metrics:
- Regression: `mse`, `rmse`, `mae`, `r_squared`, `adj_r_squared`
- Classification: `auc`, `c_stat`, `accuracy`, `sensitivity`, `specificity`, `precision`, `recall`, `f1`

### results_processing.R
Post-processing and visualization:
- `calculate_efficiency_metrics()`: Compute efficiency relative to max sample size
- `determine_minimum_sample_sizes()`: Find minimum n for target efficiency
- `generate_summary_plots()`: Create performance curves
- `plot_performance_curve()`: Plot individual model/metric performance
- `create_summary_table()`: Tabular summary of results
- `print.sample_size_simulation()`: Print method for results
- `summary.sample_size_simulation()`: Summary method for results

### validation.R
Validation and comparison functions:
- `validate_sample_size_recommendations()`: Test on external data
- `cross_validate_sample_size()`: K-fold cross-validation
- `compare_methods()`: Compare different simulation configurations
- `validate_against_benchmark()`: Compare with published benchmarks
- `bootstrap_sample_size()`: Bootstrap confidence intervals
- `sensitivity_analysis()`: Parameter sensitivity analysis

## Usage Example

```r
# Source the main function (which automatically loads all helpers)
source("codeFunction/functionSamplesizeSimu.R")

# Run simulation
results <- run_general_simulation(
  data_generation = list(
    method = "joint_mvn",
    params = list(
      correlation_matrix = my_corr_matrix
    )
  ),
  predictor_config = list(
    names = c("x1", "x2", "x3"),
    types = c("continuous", "continuous", "continuous"),
    effects = list(fixed = c(0.5, -0.3, 0.2))
  ),
  reference_model = "lm",
  target_models = c("lm", "rf"),
  sample_sizes = list(min = 50, max = 500, step = 50),
  n_simulations = 1000,
  metrics = c("mse", "r_squared"),
  output_path = "results"
)

# View results
print(results)
summary(results)

# Access specific components
results$min_sample_sizes
results$results$lm$mse
```

## Adding New Functionality

### Adding a New Model Type

Edit `model_fitting.R`:
1. Add case to `fit_model()` function
2. Add case to `predict_model()` function if needed
3. Update `get_model_type()` if new class type

### Adding a New Metric

Edit `metrics_calculation.R`:
1. Add calculation logic to `calculate_metrics()` function
2. Update `results_processing.R` if metric needs special efficiency calculation

### Adding a New Data Generation Method

Edit `data_generation.R`:
1. Add case to `generate_data()` function
2. Create new helper function for the method
3. Update documentation

## Dependencies

### Required Packages
- `MASS`: Multivariate normal generation
- `Matrix`: Matrix operations

### Optional Packages
- `synthpop`: Synthetic data generation
- `lme4`: Mixed effects models
- `randomForest`: Random forest models
- `gbm`: Gradient boosting
- `rpart`: CART trees
- `mgcv`: Generalized additive models
- `nnet`: Feed-forward neural networks
- `e1071`: SVM
- `glmnet`: Ridge/lasso regression
- `pROC`: ROC curve analysis
- `ggplot2`: Enhanced plotting
- `parallel`: Parallel processing

## Notes

- All functions use consistent naming conventions
- Helper functions are designed to be modular and reusable
- Error handling and validation built into most functions
- Progress messages controlled by `verbose` parameter in main function
