library(tfruns)

tuning_run(
  "experiments/nn_experiment.R",
  runs_dir = "MOdels/tuning_runs_arch_mix",
  flags = list(
    units1           = c(32, 64, 128),
    units2           = c(16, 32, 64),
    units3           = c(8, 16, 32),
    use_second_layer = c(TRUE, FALSE),
    use_third_layer  = c(TRUE, FALSE),
    optimizer_name   = c("adam", "rmsprop", "sgd"),
    learning_rate    = c(1e-3),
    epochs        = c(20), # LR fix; Fokus Architektur
    batch_size    = c(256)
  ),
  sample = 0.3     # sonst >1000 Kombinationen!
)
