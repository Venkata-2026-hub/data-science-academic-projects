library(tfruns)
set.seed(1)

tuning_run(
  "experiments/nn_experiments_stage1_lr_opt.R",
  runs_dir = "11_stage1_lr_opt_cv3",
  flags = list(
    learning_rate = c(1e-3, 5e-4, 2e-4, 1e-4),
    optimizer_name = c("adam", "nadam", "rmsprop"),
    batch_size = c(64, 128),
    epochs = 40,
    early_patience = 5,
    lr_patience = 3
  )
)