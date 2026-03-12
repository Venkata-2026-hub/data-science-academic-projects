source("R/00_common.R")
source("R/01_preprocessing.R")

##' Train a `recipes` preprocessing pipeline
##'
##' Builds and preps a `recipes` pipeline using `prepared_data`. The function
##' returns a prepped recipe object (retain = TRUE) ready to be `bake()`d.
##'
##' @param prepared_data A data.frame/tibble produced by `prepare_data()`.
##' @param target_var Name of the target variable (default `TARGET_VAR`).
##' @return A prepped recipe object (result of `prep()` with `retain = TRUE`).
##' @examples
##' rec <- train_recipe(train_df, target_var = "int_rate")
train_recipe <- function(prepared_data, target_var = TARGET_VAR) {
  cat("Training preprocessing recipe...")
  # Create and prep recipe
  rec <- recipe(as.formula(paste(target_var, " ~ .")), data = prepared_data) %>%
    step_unknown(all_nominal_predictors()) %>%
    step_novel(all_nominal_predictors()) %>%
    step_impute_mode(all_nominal_predictors()) %>%
    step_impute_median(all_numeric_predictors()) %>%
    step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
    step_zv(all_predictors())

  prep_rec <- prep(rec, training = prepared_data, retain = TRUE)
  cat(" done.\n")
  
  return(prep_rec)
}

# XGBoost params
params_all <- list(
  objective        = "reg:squarederror",
  eval_metric      = "rmse",
  eta              = 0.05,   # slow learning rate, slow learning but better results
  max_depth        = 4,      # more conservative than 6 
  subsample        = 0.8,
  colsample_bytree = 0.8,
  min_child_weight = 10,     # more conservative
  gamma            = 0.0,
  lambda           = 1,
  alpha            = 0,
  tree_method      = "hist",
  seed             = 42
)

##' Split data into train and test sets reproducibly
##'
##' Performs a random split according to `train_frac`.
##'
##' @param data A data.frame/tibble to be split.
##' @param train_frac Fraction of rows assigned to the training set (0-1].
##' @return A list with `train` and `test` data.frames.
##' @examples
##' s <- split_data(df, train_frac = 0.8)
split_data <- function(data, train_frac = TRAIN_FRACTION) {
  if (train_frac <= 0 || train_frac > 1) {
    stop("train_frac must be in (0, 1]")
  }

  if (train_frac == 1.0) {
    return(list(train = data, test = data[0, ]))
  }

  n <- nrow(data)
  train_indices <- sample(1:n, size = floor(train_frac * n))
  
  train_data <- data[train_indices, ]
  test_data  <- data[-train_indices, ]

  print(sprintf("Data split: %d training rows, %d test rows", nrow(train_data), nrow(test_data)))
  
  return(list(train = train_data, test = test_data))
}

##' Train an XGBoost model
##'
##' Trains an XGBoost model using `xgb.train()` and the global `params_all`
##' parameter list. Expects a prepared `xgb.DMatrix` (labels included).
##'
##' @param train_matrix An `xgb.DMatrix` containing training data and labels.
##' @param nrounds Number of boosting rounds (default `TRAIN_ROUNDS`).
##' @return An `xgb.Booster` object (trained model).
##' @examples
##' model <- train_model(train_dmat, nrounds = 100)
train_model <- function(train_matrix, nrounds = TRAIN_ROUNDS) {
  cat("Training XGBoost model for", nrounds, "rounds...")
  # Train XGBoost model on ALL features
  xgb_model <- xgb.train(
    params    = params_all,
    data      = train_matrix,
    nrounds   = nrounds,
    verbose   = 1
  )
  cat("done.\n")

  return(xgb_model)
}

##' Save a model bundle (recipe + model)
##'
##' Persists a list containing the trained `recipe` and `model` to disk using
##' `saveRDS()`. This bundle is read by `run_model.R` for evaluation/inference.
##'
##' @param recipe A prepped recipe object.
##' @param model An `xgb.Booster` trained model.
##' @param bundle_path Path where to save the RDS bundle.
##' @return Invisibly returns the path (invisible `NULL`).
##' @examples
##' save_bundle(rec, model, "data/trained_bundle.rds")
save_bundle <- function(recipe, model, bundle_path = MODEL_BUNDLE_FILE_PATH) {
  bundle <- list(
    recipe = recipe,
    model  = model
  )
  
  saveRDS(bundle, bundle_path)
  cat("Model bundle saved to:", bundle_path, "\n")

  return(bundle)
}
