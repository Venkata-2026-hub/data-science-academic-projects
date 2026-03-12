# ============================================================
# 06_view_tuning_results.R
# Works with BOTH naming styles:
#   - old:  eval_mean_val_accuracy / eval_...
#   - new:  metric_mean_val_accuracy / metric_...
# Also supports Stage2: flag_n_hidden_layers
# ============================================================

library(tfruns)
library(dplyr)
library(ggplot2)

# ------------------------------------------------------------
# Helper: safely get an existing column (first match wins)
# ------------------------------------------------------------
get_col_safe <- function(df, candidates, default = NA_real_) {
  for (nm in candidates) {
    if (nm %in% names(df)) return(df[[nm]])
  }
  rep(default, nrow(df))
}

# ------------------------------------------------------------
# Helper: pretty LR labels
# ------------------------------------------------------------
fmt_lr <- function(x) {
  if (all(is.na(x))) return(x)
  # keep scientific notation short
  format(x, scientific = TRUE, digits = 1)
}

# ------------------------------------------------------------
# Build unified columns so the rest of the script is stable
# ------------------------------------------------------------
add_unified_columns <- function(df) {
  
  # --- unified main metrics ---
  df$mean_val_accuracy <- suppressWarnings(as.numeric(
    get_col_safe(df, c("eval_mean_val_accuracy", "metric_mean_val_accuracy"))
  ))
  
  df$mean_val_bal_accuracy <- suppressWarnings(as.numeric(
    get_col_safe(df, c("eval_mean_val_bal_accuracy", "metric_mean_val_bal_accuracy"))
  ))
  
  df$mean_val_macro_f1 <- suppressWarnings(as.numeric(
    get_col_safe(df, c("eval_mean_val_macro_f1", "metric_mean_val_macro_f1"))
  ))
  
  # --- unified hyperparams ---
  df$optimizer_name <- as.character(get_col_safe(df, c(
    "optimizer_name",
    "flag_optimizer_name", "flag_optimizer",
    "metric_optimizer_name", "eval_optimizer_name"
  ), default = NA_character_))
  
  df$lr <- suppressWarnings(as.numeric(get_col_safe(df, c(
    "lr", "learning_rate",
    "flag_learning_rate", "flag_lr",
    "metric_learning_rate", "eval_learning_rate"
  ))))
  
  df$batch_size <- suppressWarnings(as.integer(get_col_safe(df, c(
    "batch_size",
    "flag_batch_size",
    "metric_batch_size", "eval_batch_size"
  ))))
  
  df$epochs <- suppressWarnings(as.integer(get_col_safe(df, c(
    "epochs",
    "flag_epochs",
    "metric_epochs", "eval_epochs"
  ))))
  
  df$activation <- as.character(get_col_safe(df, c(
    "activation",
    "flag_activation",
    "metric_activation", "eval_activation"
  ), default = NA_character_))
  
  df$dropout_rate <- suppressWarnings(as.numeric(get_col_safe(df, c(
    "dropout_rate",
    "flag_dropout_rate",
    "metric_dropout_rate", "eval_dropout_rate"
  ))))
  
  df$l2_lambda <- suppressWarnings(as.numeric(get_col_safe(df, c(
    "l2_lambda", "l2",
    "flag_l2_lambda", "flag_l2",
    "metric_l2_lambda", "eval_l2_lambda"
  ))))
  
  df$use_batch_norm <- suppressWarnings(as.logical(get_col_safe(df, c(
    "use_batch_norm",
    "flag_use_batch_norm",
    "metric_use_batch_norm", "eval_use_batch_norm"
  ), default = NA)))
  
  # --- units (may or may not exist) ---
  df$units1 <- suppressWarnings(as.integer(get_col_safe(df, c("units1", "flag_units1"))))
  df$units2 <- suppressWarnings(as.integer(get_col_safe(df, c("units2", "flag_units2"))))
  df$units3 <- suppressWarnings(as.integer(get_col_safe(df, c("units3", "flag_units3"))))
  df$units4 <- suppressWarnings(as.integer(get_col_safe(df, c("units4", "flag_units4"))))
  df$units5 <- suppressWarnings(as.integer(get_col_safe(df, c("units5", "flag_units5"))))
  
  # --- number of hidden layers ---
  # Stage 2: directly logged as flag_n_hidden_layers
  nh <- suppressWarnings(as.integer(get_col_safe(df, c(
    "num_hidden_layers",
    "flag_n_hidden_layers",
    "flag_num_hidden_layers",
    "metric_layers", "eval_layers",
    "flag_layers"
  ), default = NA_real_)))
  
  df$num_hidden_layers <- nh
  
  # If still NA, try to compute from boolean flags (older experiments)
  if (all(is.na(df$num_hidden_layers))) {
    # handle both plain and flag_ prefixed names
    f2 <- get_col_safe(df, c("use_second_layer", "flag_use_second_layer"), default = 0)
    f3 <- get_col_safe(df, c("use_third_layer",  "flag_use_third_layer"),  default = 0)
    f4 <- get_col_safe(df, c("use_fourth_layer", "flag_use_fourth_layer"), default = 0)
    f5 <- get_col_safe(df, c("use_fifth_layer",  "flag_use_fifth_layer"),  default = 0)
    
    f2 <- as.integer(replace(f2, is.na(f2), 0))
    f3 <- as.integer(replace(f3, is.na(f3), 0))
    f4 <- as.integer(replace(f4, is.na(f4), 0))
    f5 <- as.integer(replace(f5, is.na(f5), 0))
    
    df$num_hidden_layers <- 1L + f2 + f3 + f4 + f5
  }
  
  # --- architecture label (nice for tables/plots) ---
  df$arch <- NA_character_
  
  if (any(df$num_hidden_layers == 2, na.rm = TRUE)) {
    idx <- df$num_hidden_layers == 2 & !is.na(df$units2)
    df$arch[idx] <- paste0(df$units1[idx], "-", df$units2[idx])
  }
  
  if (any(df$num_hidden_layers == 3, na.rm = TRUE)) {
    idx <- df$num_hidden_layers == 3 & !is.na(df$units3)
    df$arch[idx] <- paste0(df$units1[idx], "-", df$units2[idx], "-", df$units3[idx])
  }
  
  if (any(df$num_hidden_layers == 4, na.rm = TRUE)) {
    idx <- df$num_hidden_layers == 4 & !is.na(df$units4)
    df$arch[idx] <- paste0(
      df$units1[idx], "-", df$units2[idx], "-", df$units3[idx], "-", df$units4[idx]
    )
  }
  
  df
}

# ------------------------------------------------------------
# Load + prepare runs
# ------------------------------------------------------------
load_tuning_runs <- function(runs_dir) {
  
  # allow passing "12_stage2_arch_cv3" or "tfruns/12_stage2_arch_cv3"
  if (!dir.exists(runs_dir)) {
    alt <- file.path("tfruns", runs_dir)
    if (dir.exists(alt)) runs_dir <- alt
  }
  
  runs <- tfruns::ls_runs(runs_dir = runs_dir)
  
  # remove noisy columns (optional) – robust, falls Spalten fehlen
  runs <- runs %>%
    dplyr::select(-dplyr::any_of(c(
      "output",
      "source_code",
      "error_traceback",
      "metrics",
      "script"
    )))
  
  
  runs <- add_unified_columns(runs)
  
  # keep only runs where we have the metric available
  good <- subset(runs, !is.na(mean_val_accuracy))
  
  list(runs = runs, good = good, runs_dir = runs_dir)
}

# ------------------------------------------------------------
# Main: show results (table + some quick plots)
# ------------------------------------------------------------
show_tuning_results <- function(runs_dir, top_n = 20) {
  out <- load_tuning_runs(runs_dir)
  good <- out$good
  
  if (nrow(good) == 0) {
    message("No runs with mean_val_accuracy found in: ", out$runs_dir)
    return(invisible(NULL))
  }
  
  good <- good[order(-good$mean_val_accuracy), ]
  
  cat("\nDirectory:", out$runs_dir, "\n")
  cat("Runs:", nrow(out$runs), " | Valid runs:", nrow(good), "\n\n")
  
  wanted <- c(
    "run_dir",
    "mean_val_accuracy",
    "mean_val_bal_accuracy",
    "mean_val_macro_f1",
    "num_hidden_layers",
    "arch",
    "activation",
    "optimizer_name",
    "lr",
    "batch_size",
    "epochs",
    "dropout_rate",
    "l2_lambda",
    "use_batch_norm"
  )
  
  wanted <- wanted[wanted %in% names(good)]
  #print(head(good[, wanted, drop = FALSE], top_n))
  
  # Plots (only if columns exist)
  plot_lr_effect(good)
  plot_batch_effect(good)
  plot_epochs_effect(good)
  plot_sumary(good)
  
  View(good)
  return(good)
}

# ------------------------------------------------------------
# Plots
# ------------------------------------------------------------
plot_lr_effect <- function(df) {
  if (!("lr" %in% names(df)) || all(is.na(df$lr))) return(invisible(NULL))
  
  p <- ggplot(df, aes(x = factor(fmt_lr(lr)), y = mean_val_accuracy)) +
    geom_boxplot() +
    labs(title = "Effect of learning rate", x = "learning rate", y = "mean val accuracy") +
    theme_minimal()
  
  print(p)
}

plot_batch_effect <- function(df) {
  if (!("batch_size" %in% names(df)) || all(is.na(df$batch_size))) return(invisible(NULL))
  
  p <- ggplot(df, aes(x = factor(batch_size), y = mean_val_accuracy)) +
    geom_boxplot() +
    labs(title = "Effect of batch size", x = "batch size", y = "mean val accuracy") +
    theme_minimal()
  
  print(p)
}

plot_epochs_effect <- function(df) {
  if (!("epochs" %in% names(df)) || all(is.na(df$epochs))) return(invisible(NULL))
  
  p <- ggplot(df, aes(x = epochs, y = mean_val_accuracy)) +
    geom_point(alpha = 0.6) +
    geom_smooth(se = FALSE) +
    labs(title = "Effect of epochs (as configured)", x = "epochs", y = "mean val accuracy") +
    theme_minimal()
  
  print(p)
}

plot_sumary <- function(df) {
  # architecture vs accuracy
  if (!("arch" %in% names(df)) || all(is.na(df$arch))) return(invisible(NULL))
  
  # keep top architectures only (avoid unreadable plot)
  top_arch <- df %>%
    group_by(arch) %>%
    summarise(m = mean(mean_val_accuracy, na.rm = TRUE), .groups = "drop") %>%
    arrange(desc(m)) %>%
    slice_head(n = 12) %>%
    pull(arch)
  
  df2 <- df %>% filter(arch %in% top_arch)
  
  p <- ggplot(df2, aes(x = reorder(arch, mean_val_accuracy, FUN = mean), y = mean_val_accuracy)) +
    geom_boxplot() +
    coord_flip() +
    labs(title = "Top architectures (by mean val accuracy)", x = "architecture (units)", y = "mean val accuracy") +
    theme_minimal()
  
  print(p)
}

# ------------------------------------------------------------
# Optional: quick look at a single run's curves (requires run dir)
# ------------------------------------------------------------
plot_run_curves <- function(run_dir) {
  # Uses tfruns built-in viewer; if you need a plot from CSV logs we can add later.
  tfruns::view_run(run_dir)
}

plot_best_run_curves <- function(runs_dir) {
  out <- load_tuning_runs(runs_dir)
  good <- out$good
  if (nrow(good) == 0) return(invisible(NULL))
  best <- good[order(-good$mean_val_accuracy), ][1, ]
  tfruns::view_run(best$run_dir)
}
