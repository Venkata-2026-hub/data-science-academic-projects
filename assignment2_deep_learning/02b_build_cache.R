source("R/00_setup.R")
source("R/01_load_data_and_basic_clean.R")
source("R/02_preprocess_v3.R")
source("R/03_make_splits_v3.R")

library(recipes)
library(rsample)
library(dplyr)

set.seed(1)

dir.create("cache", showWarnings = FALSE)

# 1) Load + recipe
CCdata_raw <- load_cc_data()
prep_out   <- preprocess_v3(CCdata_raw)
df         <- prep_out$df
rec        <- prep_out$rec

# 2) Train/Test split (keep fixed)
splits <- make_splits(df = df, rec = rec, test_prop = 0.2, k_folds = 3)  # <- 3 folds now
train_raw <- splits$train_data_raw
test_raw  <- splits$test_data_raw
folds     <- splits$folds

class_levels <- levels(df$status)
K <- length(class_levels)

# 3) Build fold-matrices WITHOUT leakage (prep on analysis only)
fold_mats <- vector("list", length(folds$splits))

for (i in seq_along(folds$splits)) {
  fold <- folds$splits[[i]]
  analysis_raw   <- analysis(fold)
  assessment_raw <- assessment(fold)
  
  rec_fold <- prep(rec, training = analysis_raw, verbose = FALSE)
  
  ana_df <- bake(rec_fold, new_data = analysis_raw)
  ass_df <- bake(rec_fold, new_data = assessment_raw)
  
  X_train <- ana_df %>% select(-status) %>% as.matrix()
  y_train <- as.integer(ana_df$status) - 1L
  
  X_val <- ass_df %>% select(-status) %>% as.matrix()
  y_val <- as.integer(ass_df$status) - 1L
  
  fold_mats[[i]] <- list(
    X_train = X_train,
    y_train = y_train,
    X_val   = X_val,
    y_val   = y_val
  )
}

# 4) Also create "final" matrices for later (train_full + test) with recipe prepped on full training only
rec_train_full <- prep(rec, training = train_raw, verbose = FALSE)
train_full_df  <- bake(rec_train_full, new_data = train_raw)
test_df        <- bake(rec_train_full, new_data = test_raw)

X_train_full <- train_full_df %>% select(-status) %>% as.matrix()
y_train_full <- as.integer(train_full_df$status) - 1L

X_test <- test_df %>% select(-status) %>% as.matrix()
y_test <- as.integer(test_df$status) - 1L

cache <- list(
  fold_mats = fold_mats,
  X_train_full = X_train_full,
  y_train_full = y_train_full,
  X_test = X_test,
  y_test = y_test,
  K = K,
  class_levels = class_levels,
  input_dim = ncol(X_train_full)
)

saveRDS(cache, file = "cache/cc_cv3_cache.rds")
cat("Cache saved to cache/cc_cv3_cache.rds\n")
