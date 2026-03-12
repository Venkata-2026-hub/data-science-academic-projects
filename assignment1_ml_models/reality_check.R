#!/usr/bin/env Rscript

# This script performs a reality check (evaluation) of a saved XGBoost model
# on a provided dataset. It loads the model bundle (recipe + model), processes
# the input data, makes predictions, and computes regression metrics.
# It can be executed from the command line with appropriate arguments.
# Usage example:
# ./reality_check.R --data path/to/data.csv --seed 1
# For more options, run:
# ./reality_check.R --help

source("R/00_common.R")

##' Run model on a dataset for a reality check (evaluation)
##'
##' Loads a saved model bundle (recipe + xgboost model) and evaluates it on
##' the provided CSV dataset. Prints common regression metrics to stdout.
##'
##' @param data_path Path to input CSV file (semicolon-delimited)
##' @param model_path Path to saved model bundle RDS (created by `save_bundle`)
##' @param seed RNG seed used for any nondeterministic steps (optional, default 1)
##' @return Invisibly returns `NULL`; prints evaluation metrics as side effects.
reality_check_xgb <- function(data_path, model_path = MODEL_BUNDLE_FILE_PATH, seed = 1) {
  # Source preprocessing and evaluation functions
  source("R/01_preprocessing.R")
  source("R/03_evaluation.R")

  if (file.exists(data_path)) {
    cat("Loading data from:", data_path, "\n")
  } else {
    stop("Data file does not exist:", data_path)
  }

  set.seed(seed)

  data_in <- read_delim(
    data_path,
    delim = ";",
    escape_double = FALSE,
    trim_ws = TRUE,
    show_col_types = FALSE
  )

  if (file.exists(model_path)) {
    cat("Loading model from:", model_path, "\n")
  } else {
    stop("Model file does not exist:", model_path)
  }

  bundle <- readRDS(model_path)

  xgb_matrix <- raw_data_to_xgb_matrix(data_in,
                                       bundle$recipe,
                                       target_var = TARGET_VAR)

  predictions <- predict(bundle$model, xgb_matrix)

  results <- evaluate_predictions(
    true_values = data_in[[TARGET_VAR]],
    predictions = predictions
  )

  cat("Evaluation Results:\n")
  cat(sprintf("MSE: %.4f\n", results$MSE))
  cat(sprintf("RMSE: %.4f\n", results$RMSE))
  cat(sprintf("R-squared: %.4f\n", results$R2))
}

##' Command-line entrypoint for `reality_check_xgb`
##'
##' Parses command-line options when the script is executed directly and
##' invokes `reality_check_xgb()` with the parsed arguments.
reality_check_main <- function() {
  install_packages(c("optparse"))

  # Define command line options
  option_list <- list(
    make_option(c("-d", "--data"), type = "character", default = NULL,
                            help = "Path to the data file", metavar = "character"),
    make_option(c("-m", "--model"), type = "character", default = MODEL_BUNDLE_FILE_PATH,
                            help = "Path to the trained model bundle file [default = %default]", metavar = "character"),
    make_option(c("-s", "--seed"), type = "integer", default = 1337,
                help = "Random seed for reproducibility [default = %default]", metavar = "integer")
  )

  # Parse arguments
  opt_parser <- OptionParser(option_list = option_list)
  opt <- parse_args(opt_parser)

  # Handle help
  if (opt$help) {
    print_help(opt_parser)
    quit(save = "no", status = 0)
  }

  if (is.null(opt$data)) {
    print_help(opt_parser)
    stop("--data argument is required", call. = FALSE)
  }

  reality_check_xgb(
    data_path  = opt$data,
    model_path = opt$model,
    seed       = opt$seed
  )
}

# Execute main function if script is run directly
if (sys.nframe() == 0) {
  reality_check_main()
}