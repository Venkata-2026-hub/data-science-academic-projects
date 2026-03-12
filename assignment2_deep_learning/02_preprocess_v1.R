preprocess_v1 <- function(CCdata) {
  
  # Drop ID and original day variables (we use AGE_YEARS, EMPLOYED_YEARS)
  df <- CCdata %>%
    select(
      -ID,
      -DAYS_BIRTH,
      -DAYS_EMPLOYED,
      -FLAG_MOBIL
    )
  
  # Handle OCCUPATION_TYPE NAs as explicit "Unknown"
  df <- df %>%
    mutate(
      OCCUPATION_TYPE = ifelse(
        is.na(OCCUPATION_TYPE), "Unknown", OCCUPATION_TYPE
      ),
      OCCUPATION_TYPE = as.factor(OCCUPATION_TYPE)
    )
  
  # Make all character variables factors
  df <- df %>%
    mutate(across(where(is.character), as.factor))
  
  # Target as factor with fixed level order
  df$status <- factor(df$status)
  
  # Use 'recipes' to define preprocessing
  rec <- recipe(status ~ ., data = df) %>%
    # one-hot encode all factors except status
    step_dummy(all_nominal_predictors()) %>%
    # replace remaining NAs (e.g. EMPLOYED_YEARS) with median
    step_impute_median(all_numeric_predictors()) %>%
    # scale to [0,1]
    step_range(all_numeric_predictors(), min = 0, max = 1)
  
  rec_prep <- prep(rec)
  baked <- bake(rec_prep, new_data = NULL)
  
  # Split into X (matrix) and y (factor)
  y <- baked$status
  X <- baked %>%
    select(-status) %>%
    as.matrix()
  
  list(
    X = X,
    y = y,
    recipe = rec_prep    # we need this later in reality_check
  )
}
