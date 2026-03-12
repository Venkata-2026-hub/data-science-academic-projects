#!/usr/bin/env Rscript

# This script trains an XGBoost model from a provided CSV dataset,
# evaluates it on a holdout set, and saves the trained model bundle
# (preprocessing recipe + model) to an RDS file.
# It can be executed from the command line with appropriate arguments.
# Usage example:
# ./train_xgb_model.R --data data/LCdata.csv --seed 1 --rounds 2400 --fraction 1.0
# For more options, run:
# ./train_xgb_model.R --help

source("R/00_common.R")

##' Train an XGBoost model from CSV and evaluate on the holdout
##'
##' High-level convenience wrapper that reads raw CSV data, prepares it,
##' splits into train/test partitions, fits the preprocessing recipe on the
##' training set, trains an XGBoost model, evaluates on the test set, and
##' saves a bundle containing the prepped recipe and trained model.
##'
##' @param data_path Path to input CSV file (semicolon-delimited)
##' @param train_frac Fraction of data to use for training (0-1] (default `TRAIN_FRACTION`)
##' @param seed Integer RNG seed for reproducibility (default `TRAIN_SEED`)
##' @param nrounds Number of XGBoost boosting rounds (default `TRAIN_ROUNDS`)
##' @param output_path Path to save the model bundle (RDS) (default `MODEL_BUNDLE_FILE_PATH`)
##' @return Invisibly returns `NULL`; side effects include writing `output_path`
train_xgb_model <- function(data_path,
                            train_frac = TRAIN_FRACTION,
                            seed = TRAIN_SEED,
                            nrounds = TRAIN_ROUNDS,
                            output_path = MODEL_BUNDLE_FILE_PATH) {
  # Source preprocessing helpers, training, and evaluation functions
  source("R/01_preprocessing.R")
  source("R/02_train.R")
  source("R/03_evaluation.R")

  set.seed(seed)

  data_in <- read_delim(
    data_path,
    delim = ";",
    escape_double = FALSE,
    trim_ws = TRUE,
    show_col_types = FALSE
  )

  prepared_data <- prepare_data(data_in)
  write_delim(prepared_data, TRAIN_PREPARED_FILE_PATH)
  cat("Prepared data saved to:", TRAIN_PREPARED_FILE_PATH, "\n")

  split <- split_data(prepared_data, train_frac = train_frac)

  train_data <- split$train

  recipe <- train_recipe(
    train_data,
    target_var = TARGET_VAR
  )

  train_matrix <- convert_to_xgb_matrix(train_data, recipe, target_var = TARGET_VAR)

  model <- train_model(
    train_matrix = train_matrix,
    nrounds      = nrounds
  )

  test_data <- split$test
  if (nrow(test_data) == 0) {
    cat("No test data available after split. Evaluating on given input.\n")
    test_data <- prepared_data
  }

  cat("Evaluating model...\n")
  test_matrix <- convert_to_xgb_matrix(test_data, recipe, target_var = TARGET_VAR)

  predictions <- predict(model, test_matrix)

  results <- evaluate_predictions(
      true_values = test_data[[TARGET_VAR]],
      predictions = predictions
  )

  cat("Evaluation Results on Test Set:\n")
  cat(sprintf("MSE: %.4f\n", results$MSE))
  cat(sprintf("RMSE: %.4f\n", results$RMSE))
  cat(sprintf("R-squared: %.4f\n", results$R2))

  save_bundle(recipe, model, bundle_path = output_path)
}

##' Command-line entrypoint for `train_xgb_model`
##'
##' Parses command-line options when the script is executed directly and
##' invokes `train_xgb_model()` with the parsed arguments.
train_xgb_model_main <- function() {
  install_packages(c("optparse"))

  # Define command line options
  option_list <- list(
      make_option(c("-d", "--data"), type = "character", default = NULL,
                              help = "Path to the data file", metavar = "character"),
      make_option(c("-f", "--fraction"), type = "double", default = TRAIN_FRACTION,
                              help = "Training data fraction [default = %default]", metavar = "double"),
      make_option(c("-s", "--seed"), type = "integer", default = TRAIN_SEED,
                              help = "Random seed for reproducibility [default = %default]", metavar = "integer"),
      make_option(c("-n", "--rounds"), type = "integer", default = TRAIN_ROUNDS,
                              help = "Number of boosting rounds [default = %default]", metavar = "integer"),
      make_option(c("-o", "--output"), type = "character", default = MODEL_BUNDLE_FILE_PATH,
                  help = "Path to save the trained model bundle [default = %default]", metavar = "character")
    )

  # Parse arguments
  opt_parser <- OptionParser(option_list = option_list)
  opt <- parse_args(opt_parser)

  # Handle help
  if (opt$help) {
    print_help(opt_parser)
    quit(save = "no", status = 0)
  }

  # Validate required arguments
  if (is.null(opt$data)) {
    print_help(opt_parser)
    stop("--data argument is required", call. = FALSE)
  }

  train_xgb_model(
    data_path   = opt$data,
    train_frac  = opt$fraction,
    seed        = opt$seed,
    nrounds     = opt$rounds,
    output_path = opt$output
  )
}

# Execute main function if script is run directly
if (sys.nframe() == 0) {
  train_xgb_model_main()
}