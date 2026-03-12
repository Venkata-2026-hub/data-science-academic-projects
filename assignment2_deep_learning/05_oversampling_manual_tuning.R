# R/05_manual_tuning.R
# Manual hyperparameter tuning with 10-fold CV
# Includes oversampling of minority classes in the training folds.
# Test set remains untouched and is NOT used here.

source("R/00_setup.R")
source("R/01_load_data_and_basic_clean.R")
source("R/02_preprocess_v1.R")
source("R/03_make_splits.R")
source("R/04_build_model.R")

library(rsample)
library(yardstick)
library(dplyr)

# -------------------------------------------------------------------
# 1) Helper: oversample training data in a fold
# -------------------------------------------------------------------

balance_train_data <- function(train_df) {
  # train_df: data.frame with first column 'status'
  max_n <- train_df %>%
    count(status) %>%
    pull(n) %>%
    max()
  
  train_df_bal <- train_df %>%
    group_by(status) %>%
    sample_n(max_n, replace = TRUE) %>%
    ungroup()
  
  train_df_bal
}

# -------------------------------------------------------------------
# 2) Data preparation (once)
# -------------------------------------------------------------------

set.seed(123)  # for reproducible split

CCdata_raw <- load_cc_data()
prep <- preprocess_v1(CCdata_raw)

X <- prep$X
y <- as.integer(prep$y) - 1   # classes 0..7

# One global split: train (with folds) + test (unused here)
splits <- make_splits(X, y, test_prop = 0.2, k_folds = 10)

train_data <- splits$train_data
# test_data  <- splits$test_data   # not used in this script
folds      <- splits$folds

input_dim <- ncol(train_data) - 1  # minus 'status' column

cat("Training samples:", nrow(train_data), "\n")
cat("Class distribution in training data:\n")
print(table(train_data$status))

# -------------------------------------------------------------------
# 3) Hyperparameter grid (you can adjust)
# -------------------------------------------------------------------

param_grid <- expand.grid(
  units1        = c(64, 128),
  units2        = c(32, 64),
  dropout1      = c(0.2, 0.3),
  dropout2      = c(0.2, 0.3),
  learning_rate = c(0.001, 0.0005),
  epochs        = c(20),
  batch_size    = c(256)
)

# If this is too slow, you can restrict to fewer rows, e.g.:
# param_grid <- param_grid[1:8, ]

results_list <- vector("list", nrow(param_grid))

# -------------------------------------------------------------------
# 4) Manual tuning loop with 10-fold CV and oversampling per fold
# -------------------------------------------------------------------

for (r in seq_len(nrow(param_grid))) {
  
  pars <- param_grid[r, ]
  cat("\n===============================\n")
  cat("Combination", r, "of", nrow(param_grid), "\n")
  print(pars)
  
  k <- length(folds$splits)
  
  val_acc      <- numeric(k)
  val_bal_acc  <- numeric(k)
  val_macro_f1 <- numeric(k)
  
  for (i in seq_len(k)) {
    cat("  Fold", i, "of", k, "\n")
    
    fold <- folds$splits[[i]]
    
    # analysis: training part of this fold
    # assessment: validation part (untouched)
    analysis_df   <- analysis(fold)
    assessment_df <- assessment(fold)
    
    # ---- BALANCE training data for this fold ----
    analysis_bal <- balance_train_data(analysis_df)
    
    # Matrices / vectors for keras
    X_train <- as.matrix(analysis_bal[, -1])
    y_train <- analysis_bal$status
    
    X_val <- as.matrix(assessment_df[, -1])
    y_val <- assessment_df$status
    
    # Build model for this combination
    model <- build_ffnn(
      input_dim    = input_dim,
      units1       = pars$units1,
      units2       = pars$units2,
      dropout1     = pars$dropout1,
      dropout2     = pars$dropout2,
      learning_rate = pars$learning_rate
    )
    
    history <- model |>
      fit(
        x = X_train,
        y = y_train,
        validation_data = list(X_val, y_val),
        epochs      = pars$epochs,
        batch_size  = pars$batch_size,
        verbose     = 0
      )
    
    # Keras final validation accuracy
    val_acc[i] <- tail(history$metrics$val_accuracy, 1)
    
    # Predictions on validation fold
    probs <- model |> predict(X_val)
    preds <- max.col(probs) - 1   # back to 0..7
    
    eval_df <- data.frame(
      truth       = factor(y_val,   levels = 0:7),
      .pred_class = factor(preds,   levels = 0:7)
    )
    
    # Balanced accuracy
    bal <- bal_accuracy(
      eval_df,
      truth   = truth,
      estimate = .pred_class
    )$.estimate
    
    # Macro F1
    macro <- f_meas(
      eval_df,
      truth    = truth,
      estimate = .pred_class,
      estimator = "macro"
    )$.estimate
    
    val_bal_acc[i]  <- bal
    val_macro_f1[i] <- macro
  }
  
  # Aggregate across folds
  results_list[[r]] <- data.frame(
    units1        = pars$units1,
    units2        = pars$units2,
    dropout1      = pars$dropout1,
    dropout2      = pars$dropout2,
    learning_rate = pars$learning_rate,
    epochs        = pars$epochs,
    batch_size    = pars$batch_size,
    mean_val_accuracy     = mean(val_acc),
    mean_val_bal_accuracy = mean(val_bal_acc),
    mean_val_macro_f1     = mean(val_macro_f1)
  )
  
  cat("  -> mean val accuracy    :", mean(val_acc), "\n")
  cat("  -> mean val bal acc     :", mean(val_bal_acc), "\n")
  cat("  -> mean val macro F1    :", mean(val_macro_f1), "\n")
}

# -------------------------------------------------------------------
# 5) Collect and save results
# -------------------------------------------------------------------

results <- bind_rows(results_list)

# Order by macro F1 (main metric for imbalance)
results_sorted <- results[order(-results$mean_val_macro_f1), ]

print(results_sorted)

write.csv(results_sorted, "tuning_results_manual_oversampled.csv", row.names = FALSE)

cat("\nTuning finished. Results saved to 'tuning_results_manual_oversampled.csv'.\n")
