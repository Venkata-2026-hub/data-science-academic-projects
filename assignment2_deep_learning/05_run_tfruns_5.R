library(tfruns)

tuning_run(
  "experiments/nn_experiment.R",
  runs_dir = "06_tuning_runs_activator_leaky_relu_v4",
  flags = list(
    units1           = c(64, 128, 256),
    units2           = c(32, 64),
    units3           = c(16, 32),
    use_second_layer = c(TRUE),
    use_third_layer  = c(TRUE, FALSE),
    optimizer_name   = c("adam"),
    learning_rate    = c(1e-3, 5e-4, 2e-4),
    epochs           = c(15, 20, 25),
    batch_size       = c(128)
  ),
  sample = 0.4
)