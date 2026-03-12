library(tfruns)

tuning_run(
  "experiments/nn_experiment.R",
  runs_dir = "02_tuning_runs_arch_lrbatch_epochs",
  flags = list(
    units1           = c(32, 64, 128),
    units2           = c(16, 32, 64),
    units3           = c(8, 16, 32),
    use_second_layer = c(TRUE, FALSE),
    use_third_layer  = c(TRUE, FALSE),
    optimizer_name   = c("adam", "rmsprop"),
    learning_rate    = c(1e-2, 5e-3, 1e-3, 5e-4, 1e-4),
    epochs        = c(15, 25, 35), 
    batch_size    = c(128, 256, 512)
  ),
  sample = 0.1     
)