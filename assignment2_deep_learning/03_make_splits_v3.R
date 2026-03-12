library(rsample)
library(dplyr)

make_splits <- function(df, rec, test_prop = 0.2, k_folds = 10) {
  
  set.seed(1)
  init <- initial_split(df, prop = 1 - test_prop, strata = status)
  train_data <- training(init)
  test_data  <- testing(init)
  
  rec_prep <- prep(rec, training = train_data)
  
  train_baked <- bake(rec_prep, new_data = train_data)
  test_baked  <- bake(rec_prep, new_data = test_data)
  
  y_train <- train_baked$status
  X_train <- train_baked %>%
    select(-status) %>%
    as.matrix()
  
  y_test <- test_baked$status
  X_test <- test_baked %>%
    select(-status) %>%
    as.matrix()
  
  folds <- vfold_cv(train_data, v = k_folds, strata = status)
  
  list(
    train_data_raw = train_data,  # for CV; still has original columns
    test_data_raw  = test_data,
    folds          = folds,
    recipe_prep    = rec_prep,
    X_train        = X_train,
    y_train        = y_train,
    X_test         = X_test,
    y_test         = y_test
  )
}