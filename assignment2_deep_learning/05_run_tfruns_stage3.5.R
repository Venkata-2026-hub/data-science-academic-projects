library(tfruns)
set.seed(1)

tuning_run(
  "experiments/nn_experiments_stage3.5.R",
  runs_dir = "Models/15_stage3_5_oversampling_cv3",
  flags = list(
    use_oversampling = c(FALSE, TRUE),
    oversample_seed = 1,
    
    epochs = 60,
    use_early_stopping = TRUE,
    early_patience = 6,
    use_reduce_lr = TRUE,
    lr_patience = 3,
    lr_factor = 0.5,
    min_lr = 1e-6
  )
)
