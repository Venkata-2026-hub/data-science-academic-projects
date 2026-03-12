library(tfruns)

tuning_run(
  "experiments/nn_experiment_v2.R",
  runs_dir = "04_tuning_runs_balanced_architecture_v2",
  flags = list(
    # architecture
    units1           = c(128, 256),
    units2           = c(32, 64),
    units3           = c(16, 32),
    use_second_layer = c(TRUE),           # always at least 2 layers
    use_third_layer  = c(TRUE, FALSE),    # with / without 3rd layer
    
    # optimization
    optimizer_name   = c("adam"),         # fixed
    learning_rate    = c(1e-2, 5e-3),     # see plot_lr_effect("03_tuning_runs_arch_lrbatch_epochs_no1layer")
    
    # training schedule 
    epochs           = c(15, 25),        # max 25 - see plot_lr_effect("03_tuning_runs_arch_lrbatch_epochs_no1layer")
    batch_size       = c(128)            # fixed - see plot_lr_effect("03_tuning_runs_arch_lrbatch_epochs_no1layer")
  ),
  sample = 0.2   # run 50% of all combinations (you can change to 0.2 if you want more)
)