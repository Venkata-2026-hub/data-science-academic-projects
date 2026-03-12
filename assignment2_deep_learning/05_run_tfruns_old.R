# Small tfruns hyperparameter search to check everything works.


library(tfruns)

tuning_run(
  "experiments/nn_experiment.R",
  runs_dir = "tuning_runs",
  flags = list(
    units1        = c(64, 128),
    units2        = c(32, 64),
    dropout1      = c(0.2, 0.3),
    dropout2      = c(0.2, 0.3),
    learning_rate = c(1e-3, 5e-4),
    epochs        = c(20),
    batch_size    = c(256)
  ),
  sample = 0.1   # only 4 random combinations to keep it small at first
)

