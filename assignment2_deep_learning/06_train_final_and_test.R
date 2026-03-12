source("R/00_setup.R")
source("R/01_load_data_and_basic_clean.R")
source("R/02_preprocess_v1.R")
source("R/03_make_splits.R")
source("R/04_build_model.R")

CCdata_raw <- load_cc_data()
prep <- preprocess_v1(CCdata_raw)

X <- prep$X
y <- as.integer(prep$y) - 1

splits <- make_splits(X, y, test_prop = 0.2, k_folds = 10)
train_data <- splits$train_data
test_data  <- splits$test_data

input_dim <- ncol(train_data) - 1

# <-- insert the BEST hyperparameters found with tfruns here -->
best_params <- list(
  units1        = 128,
  units2        = 64,
  dropout1      = 0.3,
  dropout2      = 0.3,
  learning_rate = 0.001,
  batch_size    = 256,
  epochs        = 60
)

model <- build_ffnn(
  input_dim   = input_dim,
  units1      = best_params$units1,
  units2      = best_params$units2,
  dropout1    = best_params$dropout1,
  dropout2    = best_params$dropout2,
  learning_rate = best_params$learning_rate
)

history <- model |>
  fit(
    x = as.matrix(train_data[, -1]),
    y = train_data$status,
    epochs = best_params$epochs,
    batch_size = best_params$batch_size,
    validation_split = 0.1,
    verbose = 1
  )

# Accuracy on hold-out test set
scores <- model |>
  evaluate(
    x = as.matrix(test_data[, -1]),
    y = test_data$status,
    verbose = 0
  )
cat("Test accuracy:", scores["accuracy"], "\n")

# Accuracy on full historic data (as requested in assignment)
full_scores <- model |>
  evaluate(
    x = X,
    y = y,
    verbose = 0
  )
cat("Accuracy on full historic data:", full_scores["accuracy"], "\n")

# Save final data sets and model
write.csv(train_data, "data/train_data_final.csv", row.names = FALSE)
write.csv(test_data,  "data/test_data_final.csv",  row.names = FALSE)

save_model(model, "models/best_model.keras")
saveRDS(prep$recipe, file = "models/preprocessing_recipe.rds")
