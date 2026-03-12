# R/06_view_tuning_results.R
# Kleine Auswertungs- und Visualisierungs-Tools für tfruns-Ergebnisse

library(tfruns)
library(ggplot2)
library(dplyr)
library(tidyr)
library(jsonlite) 

# ------------------------------------------------------------
# 1) Alle Runs anzeigen + ggplot der Mean Validation Accuracy
# ------------------------------------------------------------
show_tuning_results <- function(runs_dir = "tuning_runs") {
  
  runs <- tfruns::ls_runs(runs_dir = runs_dir)
  
  # Nur Runs mit eval_mean_val_accuracy behalten
  good <- subset(runs, !is.na(eval_mean_val_accuracy))
  if (nrow(good) == 0) {
    message("Keine Runs mit 'eval_mean_val_accuracy' gefunden.")
    return(invisible(NULL))
  }
  
  # Nach bester Accuracy sortieren (beste Zeile zuerst)
  good <- good[order(-good$eval_mean_val_accuracy), ]
  
  # Spalten, die wir gern in der Tabelle sehen möchten
  wanted <- c(
    "run_dir",
    "eval_mean_val_accuracy",
    "eval_mean_val_bal_accuracy",
    "eval_mean_val_macro_f1",
    
    # Architektur: Neuronen + Layer-Flags + Optimizer
    "units1", "units2", "units3",
    "use_second_layer", "use_third_layer",
    "optimizer_name",
    "dropout1", "dropout2",
    "learning_rate", "batch_size", "epochs",
    
    # Falls tfruns sie als flag_* gespeichert hat:
    "flag_units1", "flag_units2", "flag_units3",
    "flag_use_second_layer", "flag_use_third_layer",
    "flag_dropout1", "flag_dropout2",
    "flag_optimizer_name",
    "flag_learning_rate", "flag_batch_size"
  )

  # Nur Spalten verwenden, die es wirklich gibt
  cols <- intersect(wanted, colnames(good))
  
  # Anzahl Layer berechnen, wenn Flags vorhanden sind
  if (all(c("use_second_layer", "use_third_layer") %in% colnames(good))) {
    good$num_hidden_layers <- 1 +
      as.integer(good$use_second_layer) +
      as.integer(good$use_third_layer)
  } else if (all(c("flag_use_second_layer", "flag_use_third_layer") %in% colnames(good))) {
    good$num_hidden_layers <- 1 +
      as.integer(good$flag_use_second_layer) +
      as.integer(good$flag_use_third_layer)
  } else {
    good$num_hidden_layers <- NA_integer_
  }
  
  # 1) Tabelle im Viewer öffnen
  View(good[, cols, drop = FALSE])
  
  # 2) ggplot: Mean Validation Accuracy über alle Runs
  good$run_index <- seq_len(nrow(good))
  good$learning_rate_f  <- factor(good$learning_rate)
  good$batch_size_f     <- factor(good$batch_size)
  good$epochs_num       <- as.numeric(good$epochs)
  

  p <- ggplot(good,
              aes(x = run_index,
                  y = eval_mean_val_accuracy,
                  color = factor(num_hidden_layers),
                  shape = if ("optimizer_name" %in% colnames(good))
                    optimizer_name
                  else NULL)) +
    geom_line(alpha = 0.5) +
    geom_point(size = 2) +
    labs(x = "Run #",
         y = "Mean validation accuracy",
         color = "# Hidden layers",
         shape = "Optimizer",
         title = paste("Hyperparameter tuning results (", runs_dir, ")", sep = "")) +
    theme_minimal()
  
  print(p)
  
  # Dataframe zurückgeben, falls man weiterarbeiten möchte
  invisible(good)
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
#plot_best_run_curves("tuning_runs_arch_mix")