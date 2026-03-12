if (!exists(".COMMON_LOADED")) {
  .COMMON_LOADED <- TRUE
  # Constants
  TRAIN_FILE_PATH <- "data/LCdata.csv"
  TRAIN_PREPARED_FILE_PATH <- "data/prepared_data.csv"
  TRAIN_FRACTION <- 0.7
  TRAIN_SEED <- 1
  TRAIN_ROUNDS <- 2400
  TARGET_VAR <- "int_rate"
  MODEL_BUNDLE_FILE_PATH <- "data/trained_bundle.rds"

  # Load required packages

  #' Install and load missing packages
  #'
  #' Ensures a CRAN mirror is configured, installs any packages from
  #' `packages` that are missing, and then loads them with `require()`.
  #'
  #' @param packages Character vector of package names to install/load.
  #' @return Invisibly returns the result of `lapply(..., require)`.
  #' @examples
  #' install_packages(c("readr", "xgboost"))
  install_packages <- function(packages) {
    # Ensure a CRAN mirror is set so install.packages() runs non-interactively
    if (is.null(getOption("repos")) || getOption("repos")["CRAN"] == "@CRAN@") {
      options(repos = c(CRAN = "https://cloud.r-project.org"))
    }
    missing <- packages[!sapply(packages, require, character.only = TRUE, quietly = TRUE)]
    if (length(missing)) {
      install.packages(missing, repos = getOption("repos"))
    }
    lapply(packages, require, character.only = TRUE)
  }

  install_packages(c("readr", "xgboost", "recipes", "tidyverse"))
}
