library(tfruns)
set.seed(1)

drop_grid <- c(0.0, 0.1, 0.2, 0.3, 0.4, 0.5)

tuning_run(
  "experiments/nn_experiments_stage3.R",
  runs_dir = "Models/14_stage3_dropout_cv3",
  flags = list(
    dropout_rate = drop_grid,
    
    # oversampling present but not tested yet
    use_oversampling = FALSE,
    oversample_seed = 1,
    
    # training loop controls (optional)
    epochs = 60,
    use_early_stopping = TRUE,
    early_patience = 6,
    use_reduce_lr = TRUE,
    lr_patience = 3,
    lr_factor = 0.5,
    min_lr = 1e-6
  )
)
