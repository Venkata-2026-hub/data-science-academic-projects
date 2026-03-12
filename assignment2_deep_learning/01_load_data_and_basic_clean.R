load_cc_data <- function(path = "data/Dataset-part-2.csv") {
  CCdata <- read.table(path, sep = ",", header = TRUE)
  
  CCdata <- CCdata %>%
    mutate(
      DAYS_EMPLOYED = ifelse(DAYS_EMPLOYED == 365243, NA, DAYS_EMPLOYED),
      AGE_YEARS = abs(DAYS_BIRTH) / 365,
      EMPLOYED_YEARS = abs(DAYS_EMPLOYED) / 365
    )
  
  CCdata
}
