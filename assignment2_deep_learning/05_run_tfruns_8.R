library(tfruns)

tuning_run(
  "experiments/nn_experiment_pp.R",
  runs_dir = "09_new_preprocessing_v3",
  flags = list(
    units1           = c(64, 128, 256),
    units2           = c(32, 64),
    units3           = c(16, 32),
    units4           = c(8, 16),
    units4           = c(4, 8),
    use_second_layer = c(TRUE),
    use_third_layer  = c(TRUE, FALSE),
    use_fourth_layer = c(TRUE, FALSE),
    use_fifth_layer = c(TRUE, FALSE),
    optimizer_name   = c("adam"),
    learning_rate    = c(1e-3, 5e-4),
    epochs           = c(15, 20, 25),
    batch_size       = c(128)
  ),
  sample = 0.001
)