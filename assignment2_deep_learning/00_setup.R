rm(list = ls())

set.seed(1)

required_pkgs <- c(
  "tidyverse",
  "keras3",      
  "tfruns",
  "recipes",
  "rsample",
  "yardstick",
  "ggplot2",
  "gridExtra",
  "dplyr"
)

for (p in required_pkgs) {
  if (!require(p, character.only = TRUE)) {
    install.packages(p)
    library(p, character.only = TRUE)
  }
}

options(digits = 4)
