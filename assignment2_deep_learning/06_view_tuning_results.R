# ------------------------------------------------------------
# 06_view_tuning_results.R
# Auswertung der tfruns-Experimente
# ------------------------------------------------------------

library(tfruns)
library(dplyr)
library(ggplot2)

# Hilfsfunktion: ersten existierenden Spaltennamen nehmen
get_col_safe <- function(df, candidates, default = NA_real_) {
  for (nm in candidates) {
    if (nm %in% names(df)) {
      return(df[[nm]])
    }
  }
  return(rep(default, nrow(df)))
}

# ------------------------------------------------------------
# 1) Runs laden und saubere Spalten bauen
# ------------------------------------------------------------

load_tuning_runs <- function(runs_dir = "tuning_runs") {
  runs <- tfruns::ls_runs(runs_dir = runs_dir)
  good <- subset(runs, !is.na(eval_mean_val_accuracy))
  
    # Extract hyperparameters from evaluation-metadata (batch_size!)
  if ("eval_batch_size" %in% colnames(good)) good$batch_size <- good$eval_batch_size
  if ("eval_learning_rate" %in% colnames(good)) good$learning_rate <- good$eval_learning_rate
  if ("eval_epochs" %in% colnames(good)) good$epochs <- good$eval_epochs
  if ("eval_units1" %in% colnames(good)) good$units1 <- good$eval_units1
  if ("eval_units2" %in% colnames(good)) good$units2 <- good$eval_units2
  if ("eval_units3" %in% colnames(good)) good$units3 <- good$eval_units3
  if ("eval_units4" %in% colnames(good)) good$units4 <- good$eval_units4
  if ("eval_units5" %in% colnames(good)) good$units5 <- good$eval_units5
  if ("eval_hidden_layers" %in% colnames(good)) good$hidden_layers <- good$eval_hidden_layers
  
  # konsistente Spalten für Hyperparameter
  good$lr <- get_col_safe(
    good,
    c("lr", "learning_rate", "eval_learning_rate", "flag_learning_rate")
  )
  
  good$batch_size <- get_col_safe(
    good,
    c("batch_size", "eval_batch_size", "flag_batch_size")
  )
  
  good$epochs <- get_col_safe(
    good,
    c("epochs", "eval_epochs", "flag_epochs")
  )
  
  good$optimizer_name <- get_col_safe(
    good,
    c("optimizer_name", "flag_optimizer_name")
  )
  
  good$units1 <- get_col_safe(
    good,
    c("units1", "eval_units1", "flag_units1")
  )
  
  good$units2 <- get_col_safe(
    good,
    c("units2", "eval_units2", "flag_units2")
  )
  
  good$units3 <- get_col_safe(
    good,
    c("units3", "eval_units3", "flag_units3")
  )
  
  good$units4 <- get_col_safe(
    good,
    c("units4", "eval_units4", "flag_units4")
  )
  
  good$units5 <- get_col_safe(
    good,
    c("units5", "eval_units5", "flag_units5")
  )
  
  # Anzahl Hidden-Layer berechnen
  if (all(c("use_second_layer", "use_third_layer", "use_fourth_layer", "use_fifth_layer") %in% colnames(good))) {
    good$num_hidden_layers <- 1L +
      rowSums(
        as.data.frame(lapply(
          good[, c("use_second_layer", "use_third_layer", "use_fourth_layer", "use_fifth_layer")],
          function(x) as.integer(replace(x, is.na(x), 0))
        ))
      )
    
  } else if (all(c("flag_use_second_layer", "flag_use_third_layer", "flag_use_fourth_layer", "flag_use_fifth_layer") %in% colnames(good))) {
    good$num_hidden_layers <- 1L +
      rowSums(
        as.data.frame(lapply(
          good[, c("flag_use_second_layer", "flag_use_third_layer", "flag_use_fourth_layer", "flag_use_fifth_layer")],
          function(x) as.integer(replace(x, is.na(x), 0))
        ))
      )
    
  } else if ("eval_layers" %in% colnames(good)) {
    good$num_hidden_layers <- good$eval_layers
    
  } else {
    good$num_hidden_layers <- NA_integer_
  }
  
  
  # Run-Index für Plots
  good$run_index <- seq_len(nrow(good))
  
  good
}

# ------------------------------------------------------------
# 2) Hauptfunktion: Tabelle + Überblicksplot
# ------------------------------------------------------------

show_tuning_results <- function(runs_dir = "tuning_runs") {
  good <- load_tuning_runs(runs_dir)
  
  # nach Accuracy sortieren (beste oben)
  good <- good[order(-good$eval_mean_val_accuracy), ]
  
  # Spalten für die Tabelle
  wanted <- c(
    "run_dir",
    "eval_mean_val_accuracy",
    "eval_mean_val_bal_accuracy",
    "eval_mean_val_macro_f1",
    #"units1", "units2", "units3",
    "num_hidden_layers",
    "optimizer_name",
    "lr", "batch_size", "epochs",
    "units1", "units2", "units3", "units4", "units5"
  )
  cols <- intersect(wanted, colnames(good))
  
  View(good[, cols, drop = FALSE])
  
  ggplot(
    good,
    aes(
      x     = run_index,
      y     = eval_mean_val_accuracy,
      color = factor(num_hidden_layers)
    )
  ) +
    geom_line(alpha = 0.4) +
    geom_point() +
    labs(
      x     = "Run #",
      y     = "Mean validation accuracy",
      color = "# Hidden layers",
      title = paste("Hyperparameter tuning results (", runs_dir, ")", sep = "")
    ) +
    theme_minimal()
  
  # Einfacher Überblicksplot: wie vorher, nur mit Layers
  p <- ggplot(
    good,
    aes(x = run_index,
        y = eval_mean_val_accuracy,
        color = factor(num_hidden_layers))
  ) +
    geom_line(alpha = 0.4) +
    geom_point() +
    labs(
      x = "Run #",
      y = "Mean validation accuracy",
      color = "# Hidden layers",
      title = paste(
        "Hyperparameter tuning results (",
        runs_dir, ")",
        sep = ""
      )
    ) +
    theme_minimal()
  
  print(p)
  
  invisible(good)
}

# ------------------------------------------------------------
# 3) Zusätzliche Auswertungsplots
# ------------------------------------------------------------

# a) Effekt der Learning Rate
plot_lr_effect <- function(runs_dir = "tuning_runs") {
  library(scales)
  library(dplyr)
  library(ggplot2)
  
  good <- load_tuning_runs(runs_dir)
  
  # Learning Rate robust bestimmen (je nach Spaltennamen)
  base_lr <- dplyr::coalesce(
    good$lr,
    good$learning_rate,
    good$eval_learning_rate,
    good$flag_learning_rate
  )
  
  # Falls wirklich alles NA ist: freundlich abbrechen
  if (all(is.na(base_lr))) {
    stop("Keine Lernraten in den Runs gefunden (alle NA). Prüfe, ob learning_rate in metrics geschrieben wird.")
  }
  
  # Nur gültige LR-Werte formatieren
  sci_fmt <- scientific_format(digits = 1)
  lr_char <- sci_fmt(base_lr)
  
  # neue Spalte mit hübscher Darstellung
  good$lr_s <- factor(lr_char)
  
  ggplot(
    good,
    aes(
      x     = lr_s,
      y     = eval_mean_val_accuracy,
      color = factor(num_hidden_layers)
    )
  ) +
    geom_boxplot(outlier.alpha = 0.3) +
    geom_jitter(width = 0.15, alpha = 0.5) +
    theme_minimal() +
    labs(
      title = paste("Validation accuracy vs learning rate (", runs_dir, ")", sep = ""),
      x     = "Learning rate",
      y     = "Mean validation accuracy",
      color = "# Hidden layers"
    )
}

# b) Effekt der Batch Size
plot_batch_effect <- function(runs_dir = "tuning_runs") {
  good <- load_tuning_runs(runs_dir)
  
  ggplot(
    good,
    aes(
      x = factor(batch_size),
      y = eval_mean_val_accuracy,
      fill = factor(num_hidden_layers)
    )
  ) +
    geom_boxplot(alpha = 0.6) +
    theme_minimal() +
    labs(
      title = paste("Validation accuracy vs batch size (", runs_dir, ")", sep = ""),
      x = "Batch size",
      y = "Mean validation accuracy",
      fill = "# Hidden layers"
    )
}

# c) Effekt der Epochenzahl
plot_epochs_effect <- function(runs_dir = "tuning_runs") {
  good <- load_tuning_runs(runs_dir)
  
  ggplot(
    good,
    aes(
      x = factor(epochs),
      y = eval_mean_val_accuracy,
      color = factor(num_hidden_layers)
    )
  ) +
    geom_boxplot() +
    geom_jitter(width = 0.1, alpha = 0.4) +
    theme_minimal() +
    labs(
      title = paste("Effect of epochs on validation accuracy (", runs_dir, ")", sep = ""),
      x = "Epochs",
      y = "Mean validation accuracy",
      color = "# Hidden layers"
    )
}

plot_sumary <- function(runs_dir = "tuning_runs") {
  good <- load_tuning_runs(runs_dir)
  
  ggplot(
    good,
    aes(x     = run_index,
        y     = eval_mean_val_accuracy,
        color = factor(num_hidden_layers),   # 1 / 2 / 3 Hidden-Layer
        shape = batch_size_f,                # Batch Size
        size  = epochs_num)                  # mehr Epochen = größere Punkte
  ) +
    geom_line(alpha = 0.3) +
    geom_point() +
    facet_wrap(~ learning_rate_f, nrow = 1) +  # Panel pro Learning Rate
    labs(
      x     = "Run # (innerhalb Learning-Rate-Panel)",
      y     = "Mean validation accuracy",
      color = "# Hidden layers",
      shape = "Batch size",
      size  = "Epochs",
      title = paste(
        "Hyperparameter tuning results (",
        runs_dir, ")",
        sep = ""
      )
    ) +
    theme_minimal()
  
  print(p)
}
   

# ----------------------------------------------------------------
# 2) Lernkurven (Accuracy & Loss) für einen konkreten Run plotten
# ----------------------------------------------------------------
plot_run_curves <- function(run_dir) {
  
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(jsonlite)
  library(gridExtra)   # für die Anordnung der Plots
  
  # irgendeine Metrics-Datei finden (csv oder json)
  files <- list.files(run_dir,
                      pattern = "metrics\\.(csv|json)$",
                      recursive = TRUE,
                      full.names = TRUE)
  
  if (length(files) == 0) {
    stop("Keine Metrics-Datei im Run-Ordner gefunden: ", run_dir)
  }
  
  metrics_path <- files[1]
  message("Verwende Metrics-Datei: ", metrics_path)
  
  # CSV oder JSON einlesen
  if (grepl("\\.json$", metrics_path)) {
    m <- jsonlite::fromJSON(metrics_path)
    if (!is.data.frame(m)) m <- as.data.frame(m)
  } else {
    m <- read.csv(metrics_path)
  }
  
  m$epoch <- seq_len(nrow(m))
  
  p_acc  <- NULL
  p_loss <- NULL
  
  ## ----- Accuracy (train + val) -----
  acc_cols_candidates     <- intersect(c("accuracy", "acc"), names(m))
  val_acc_cols_candidates <- intersect(c("val_accuracy", "val_acc"), names(m))
  
  if (length(acc_cols_candidates) >= 1 && length(val_acc_cols_candidates) >= 1) {
    
    acc_col     <- acc_cols_candidates[1]
    val_acc_col <- val_acc_cols_candidates[1]
    
    acc_df <- m %>%
      select(epoch,
             train = !!sym(acc_col),
             val   = !!sym(val_acc_col)) %>%
      pivot_longer(cols = c("train", "val"),
                   names_to = "dataset",
                   values_to = "value")
    
    p_acc <- ggplot(acc_df,
                    aes(x = epoch, y = value, color = dataset)) +
      geom_line() +
      geom_point() +
      labs(title = "Accuracy per epoch",
           x = "Epoch", y = "Accuracy") +
      theme_minimal()
  } else {
    message("Keine passenden Accuracy-Spalten gefunden. Verfügbare Spalten: ",
            paste(names(m), collapse = ", "))
  }
  
  ## ----- Loss (train + val) -----
  if (all(c("loss", "val_loss") %in% names(m))) {
    
    loss_df <- m %>%
      select(epoch,
             train = loss,
             val   = val_loss) %>%
      pivot_longer(cols = c("train", "val"),
                   names_to = "dataset",
                   values_to = "value")
    
    p_loss <- ggplot(loss_df,
                     aes(x = epoch, y = value, color = dataset)) +
      geom_line() +
      geom_point() +
      labs(title = "Loss per epoch",
           x = "Epoch", y = "Loss") +
      theme_minimal()
  } else {
    message("Keine passenden Loss-Spalten gefunden. Verfügbare Spalten: ",
            paste(names(m), collapse = ", "))
  }
  
  # Beide Plots gemeinsam anzeigen (Accuracy oben, Loss unten)
  plots <- list()
  if (!is.null(p_acc))  plots <- c(plots, list(p_acc))
  if (!is.null(p_loss)) plots <- c(plots, list(p_loss))
  
  if (length(plots) == 0) {
    message("Keine Plots zu zeichnen.")
  } else if (length(plots) == 1) {
    print(plots[[1]])
  } else {
    gridExtra::grid.arrange(grobs = plots, ncol = 1)
  }
}


# ----------------------------------------------------------------
# 3) Komfortfunktion: besten Run suchen und seine Kurven plotten
# ----------------------------------------------------------------
plot_best_run_curves <- function(runs_dir = "tuning_runs") {
  
  runs <- tfruns::ls_runs(runs_dir = runs_dir)
  good <- subset(runs, !is.na(eval_mean_val_accuracy))
  
  if (nrow(good) == 0) {
    message("Keine Runs mit 'eval_mean_val_accuracy' gefunden.")
    return(invisible(NULL))
  }
  
  good <- good[order(-good$eval_mean_val_accuracy), ]
  
  best_run_dir  <- good$run_dir[1]
  best_accuracy <- good$eval_mean_val_accuracy[1]
  
  message("Bester Run: ", best_run_dir,
          "  (eval_mean_val_accuracy = ",
          round(best_accuracy, 4), ")")
  
  plot_run_curves(best_run_dir)
  
  invisible(best_run_dir)
}


# Aufrufen: source("R/06_view_tuning_results.R")
# Übersicht aller Runs + Plots: good <- show_tuning_results()
# Lern- und Validierungskurven für den besten Run: plot_best_run_curves()
# Lernkurven für einen konkreten Run (wenn du den Pfad kennst) plot_run_curves("tuning_runs/2025-11-28T12-50-03Z")



#show_tuning_results("tuning_runs_arch_mix")
#plot_lr_effect("02_tuning_runs_arch_lrbatch_epochs")
#plot_batch_effect("02_tuning_runs_arch_lrbatch_epochs")
#plot_epochs_effect("02_tuning_runs_arch_lrbatch_epochs")
#plot_best_run_curves("tuning_runs_arch_mix")