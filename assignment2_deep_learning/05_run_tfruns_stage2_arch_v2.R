library(tfruns)
set.seed(1)


tuning_run(
  "experiments/nn_experiments_stage2_arch.R",
  runs_dir = "Models/13_stage2_arch_cv3_conf",
  flags = list(
    # fixed
    optimizer_name = "adam",
    learning_rate = 1e-3,
    batch_size = 64,
    epochs = 60,
    early_patience = 6,
    lr_patience = 3,
    use_batch_norm = TRUE,
    n_hidden_layers = 3,
    
    # confirmation variables
    activation = c("relu", "leaky_relu"),
    
    # explicit architectures
    units1 = c(64, 128, 256),
    units2 = c(32, 64, 128),
    units3 = c(32, 32, 64)
  )
)
