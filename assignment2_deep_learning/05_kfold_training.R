source("R/00_setup.R")
library(tfruns)

my_flags <- list(
  units1 = c(64, 128),
  units2 = c(32, 64),
  dropout1 = c(0.2, 0.3),
  dropout2 = c(0.2, 0.3),
  learning_rate = c(0.001, 0.0005),
  batch_size = c(256),
  epochs = c(30, 60)
)

tuning_run(
  "experiments/nn_experiment.R",
  runs_dir = "tuning_runs",
  flags = my_flags
)

# Afterwards, inspect the best run:
best <- ls_runs(
  runs_dir = "tuning_runs",
  order = metric_mean_val_accuracy,
  decreasing = TRUE
)
head(best)
