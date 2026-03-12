# Manual hyperparameter tuning with 10-fold CV, no tfruns.

source("R/00_setup.R")
source("R/01_load_data_and_basic_clean.R")
source("R/02_preprocess_v1.R")
source("R/03_make_splits.R")
source("R/04_build_model.R")

library(rsample)
library(yardstick)
library(dplyr)

# ---- 1. Data prep (once) --------------------------------------------

CCdata_raw <- load_cc_data()
prep <- preprocess_v1(CCdata_raw)

X <- prep$X
y <- as.integer(prep$y) - 1   # 0..7

splits <- make_splits(X, y, test_prop = 0.2, k_folds = 10)
train_data <- splits$train_data
test_data  <- splits$test_data
folds      <- splits$folds

input_dim <- ncol(train_data) - 1

# ---- 2. Define a SMALL hyperparameter grid --------------------------

param_grid <- expand.grid(
  units1        = c(64, 128),
  units2        = c(32, 64),
  dropout1      = c(0.2, 0.3),
  dropout2      = c(0.2, 0.3),
  learning_rate = c(0.001, 0.0005),
  epochs        = c(20),
  batch_size    = c(256)
)

# If this is too slow, you can restrict:
# param_grid <- param_grid[1:6, ]  # only first 6 combinations

results_list <- vector("list", nrow(param_grid))

# ---- 3. Loop over hyperparameters -----------------------------------

for (r in seq_len(nrow(param_grid))) {
  
  pars <- param_grid[r, ]
  cat("\n=== Combination", r, "of", nrow(param_grid), "===\n")
  print(pars)
  
  k <- length(folds$splits)
  
  val_acc      <- numeric(k)
  val_bal_acc  <- numeric(k)
  val_macro_f1 <- numeric(k)
  
  for (i in seq_len(k)) {
    cat("  Fold", i, "of", k, "\n")
    
    fold <- folds$splits[[i]]
    analysis   <- analysis(fold)
    assessment <- assessment(fold)
    
    X_train <- as.matrix(analysis[, -1])
    y_train <- analysis$status
    
    X_val   <- as.matrix(assessment[, -1])
    y_val   <- assessment$status
    
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
    
    # last epoch validation accuracy from keras
    val_acc[i] <- tail(history$metrics$val_accuracy, 1)
    
    # predictions on validation fold
    probs <- model |> predict(X_val)
    preds <- max.col(probs) - 1
    
    eval_df <- data.frame(
      truth       = factor(y_val, levels = 0:7),
      .pred_class = factor(preds, levels = 0:7)
    )
    
    bal <- bal_accuracy(
      eval_df,
      truth   = truth,
      estimate = .pred_class
    )$.estimate
    
    macro <- f_meas(
      eval_df,
      truth    = truth,
      estimate = .pred_class,
      estimator = "macro"
    )$.estimate
    
    val_bal_acc[i]  <- bal
    val_macro_f1[i] <- macro
  }
  
  # aggregate metrics for this hyperparameter combo
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
}

results <- bind_rows(results_list)

# Save and print
write.csv(results, "tuning_results_manual.csv", row.names = FALSE)
print(results[order(-results$mean_val_macro_f1), ])
