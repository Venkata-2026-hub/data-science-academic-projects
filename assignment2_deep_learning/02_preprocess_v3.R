preprocess_v3 <- function(CCdata) {
  
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
  df$status <- factor(df$status, levels = sort(unique(df$status)))
  
  # Use 'recipes' to define preprocessing
  rec <- recipe(status ~ ., data = df) %>%
    # one-hot encode all factors except status
    step_dummy(all_nominal_predictors()) %>%
    # replace remaining NAs (e.g. EMPLOYED_YEARS) with median
    step_impute_median(all_numeric_predictors()) %>%
    # remove zero-variance columns
    step_zv(all_predictors()) %>%        
    # scale to [0,1]
    step_range(all_numeric_predictors(), min = 0, max = 1)
  
  list(
    df   = df,
    rec  = rec
  )
}
