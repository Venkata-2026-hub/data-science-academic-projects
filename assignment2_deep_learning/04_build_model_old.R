build_ffnn <- function(input_dim,
                       units1 = 64,
                       units2 = 32,
                       dropout1 = 0.3,
                       dropout2 = 0.3,
                       learning_rate = 0.001) {
  
  model <- keras_model_sequential() |>
    layer_dense(units = units1, activation = "relu",
                input_shape = input_dim) |>
    layer_dropout(rate = dropout1) |>
    layer_dense(units = units2, activation = "relu") |>
    layer_dropout(rate = dropout2) |>
    layer_dense(units = 8, activation = "softmax")  # 8 status classes
  
  model |>
    compile(
      loss = "sparse_categorical_crossentropy",
      optimizer = optimizer_adam(learning_rate = learning_rate),
      metrics = "accuracy"
    )
}
