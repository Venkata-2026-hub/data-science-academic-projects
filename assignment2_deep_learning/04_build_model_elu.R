build_ffnn <- function(input_dim,
                       units1 = 64,
                       units2 = 32,
                       units3 = 16,
                       units4 = 8,
                       use_second_layer = TRUE,
                       use_third_layer = FALSE,
                       use_fourth_layer = FALSE,
                       learning_rate = 1e-3,
                       optimizer_name = "adam") {
  
  model <- keras_model_sequential()
  
  # 1. Hidden Layer
  model %>%
    layer_dense(units = units1,
                activation = "elu",      # <– hier ELU
                input_shape = input_dim)
  
  # 2. Hidden Layer (optional)
  if (use_second_layer) {
    model %>%
      layer_dense(units = units2,
                  activation = "elu")    # <– ELU
  }
  
  # 3. Hidden Layer (optional)
  if (use_third_layer) {
    model %>%
      layer_dense(units = units3,
                  activation = "elu")    # <– ELU
  }
  
  # 4. Hidden Layer (optional)
  if (use_fourth_layer) {
    model %>%
      layer_dense(units = units4,
                  activation = "elu")    # <– ELU
  }
  
  # Output-Layer (8 Klassen)
  model %>%
    layer_dense(units = 8,
                activation = "softmax")
  
  # Optimizer
  opt <- switch(
    optimizer_name,
    "adam"    = optimizer_adam(learning_rate = learning_rate),
    "rmsprop" = optimizer_rmsprop(learning_rate = learning_rate),
    "sgd"     = optimizer_sgd(learning_rate = learning_rate),
    optimizer_adam(learning_rate = learning_rate)
  )
  
  model %>% compile(
    loss      = "sparse_categorical_crossentropy",
    optimizer = opt,
    metrics   = "accuracy"
  )
  
  model
}