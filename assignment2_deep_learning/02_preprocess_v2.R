
# Structural preprocessing ONLY (no recipes, no scaling)

preprocess_v2 <- function(CCdata) {
  
  CCdata <- CCdata %>%
    mutate(
      # target as factor with fixed order
      status = factor(
        status,
        levels = c("0", "1", "2", "3", "4", "5", "C", "X")
      ),
      # OCCUPATION_TYPE: NA -> "Unknown"
      OCCUPATION_TYPE = ifelse(is.na(OCCUPATION_TYPE), "Unknown", OCCUPATION_TYPE)
    ) %>%
    mutate(
      OCCUPATION_TYPE = factor(OCCUPATION_TYPE),
      # all remaining characters to factors
      across(where(is.character), as.factor)
    ) %>%
    # drop ID and raw day columns; AGE_YEARS / EMPLOYED_YEARS come from 01_load_data
    select(
      -ID,
      -DAYS_BIRTH,
      -DAYS_EMPLOYED,
      -FLAG_MOBIL
    )
  
  CCdata
}
