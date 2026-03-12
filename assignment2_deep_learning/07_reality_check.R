args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0) {
  stop("Please provide the path to the secret data CSV as first argument.")
}
secret_path <- args[1]

source("R/00_setup.R")
source("R/01_load_data_and_basic_clean.R")
source("R/02_preprocess_v1.R")

# 1. Load secret data (same structure as Dataset-part-2.csv)
secret_raw <- load_cc_data(secret_path)

# 2. Load preprocessing recipe and apply to secret data
rec <- readRDS("models/preprocessing_recipe.rds")
secret_baked <- bake(rec, new_data = secret_raw)

y_secret <- as.integer(secret_baked$status) - 1
X_secret <- secret_baked %>%
  select(-status) %>%
  as.matrix()

# 3. Load best model
model <- load_model("models/best_model.keras")

# 4. Evaluate
scores <- model |>
  evaluate(X_secret, y_secret, verbose = 0)
cat("Accuracy on secret data:", scores["accuracy"], "\n")
