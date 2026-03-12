make_splits <- function(X, y, test_prop = 0.2, k_folds = 10) {
  
  df <- data.frame(status = y, X)
  set.seed(1)
  
  # Stratified initial split (by status)
  init <- initial_split(df, prop = 1 - test_prop, strata = status)
  train_data <- training(init)
  test_data  <- testing(init)
  
  # 10-fold cross-validation on training data (again stratified)
  folds <- vfold_cv(train_data, v = k_folds, strata = status)
  
  list(
    train_data = train_data,
    test_data  = test_data,
    folds      = folds
  )
}
