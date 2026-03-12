library(tfruns)

set.seed(1)

runs <- tuning_run(
  "experiments/nn_experiments_v3.R",
  runs_dir = "10_activations +  rest_v2",
  flags = list(
    # start small and meaningful:
    activation = c("relu", "elu", "leaky_relu", "tanh"),
    dropout_rate = c(0.0, 0.1, 0.2),
    l2_lambda = c(0.0, 1e-5, 1e-4),
    use_batch_norm = c(FALSE, TRUE),
    
    optimizer_name = c("adam", "rmsprop", "nadam"),
    learning_rate = c(1e-3, 5e-4, 1e-4),
    batch_size = c(64, 128, 256),
    
    # keep architecture stable first (you can expand later)
    use_second_layer = TRUE,
    use_third_layer  = c(FALSE, TRUE),
    use_fourth_layer = FALSE,
    use_fifth_layer  = FALSE,
    units1 = c(64, 128),
    units2 = c(32, 64),
    units3 = c(16, 32),
    
    epochs = 80,
    use_early_stopping = TRUE,
    early_patience = 8,
    use_reduce_lr = TRUE,
    lr_patience = 4,
    lr_factor = 0.5,
    min_lr = 1e-6,
    
    # optional imbalance handling (try later if needed)
    use_class_weights = c(FALSE, TRUE)
  ),
  sample = 0.001
)

