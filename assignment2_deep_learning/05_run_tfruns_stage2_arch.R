library(tfruns)
set.seed(1)

tuning_run(
  "experiments/nn_experiments_stage2_arch.R",
  runs_dir = "12_stage2_arch_cv3_v4",
  flags = list(
    # fixed from Stage 1:
    optimizer_name = "adam",
    learning_rate = 1e-3,
    
    # architecture:
    n_hidden_layers = c(2, 3, 4),
    activation = c("relu", "elu", "tanh", "leaky_relu"),
    use_batch_norm = c(TRUE, FALSE),
    
    # neurons (patterns):
    # (u1 is the "main width", u2/u3/u4 are decreasing widths)
    units1 = c(64, 128, 256),
    units2 = c(32, 64, 128),
    units3 = c(16, 32, 64),
    units4 = c(8, 16, 32),
    
    # training:
    batch_size = c(64, 128),
    epochs = 60,
    early_patience = 6,
    lr_patience = 3
  ),
  sample = 0.03  # IMPORTANT: random sample of grid to keep it fast
)
