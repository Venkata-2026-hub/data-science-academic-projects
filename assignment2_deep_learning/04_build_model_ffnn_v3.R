library(keras3)

add_dense_block <- function(x, units,
                            activation = "relu",
                            leaky_alpha = 0.1,
                            dropout_rate = 0,
                            l2_lambda = 0,
                            use_batch_norm = FALSE) {
  reg <- if (l2_lambda > 0) regularizer_l2(l2_lambda) else NULL
  
  x <- x %>%
    layer_dense(units = units, activation = "linear", kernel_regularizer = reg)
  
  if (use_batch_norm) x <- x %>% layer_batch_normalization()
  
  if (activation == "leaky_relu") {
    x <- x %>% layer_activation_leaky_relu(negative_slope = leaky_alpha)
  } else {
    x <- x %>% layer_activation(activation = activation)
  }
  
  if (dropout_rate > 0) x <- x %>% layer_dropout(rate = dropout_rate)
  
  x
}

build_ffnn_v3 <- function(input_dim,
                          num_classes,
                          units = c(128, 32, 32),
                          activation = "tanh",
                          leaky_alpha = 0.1,
                          dropout_rate = 0,
                          l2_lambda = 0,
                          use_batch_norm = TRUE,
                          learning_rate = 5e-4,
                          optimizer_name = "nadam") {
  
  inputs <- layer_input(shape = input_dim)
  x <- inputs
  
  for (u in units) {
    x <- add_dense_block(
      x, units = u,
      activation = activation,
      leaky_alpha = leaky_alpha,
      dropout_rate = dropout_rate,
      l2_lambda = l2_lambda,
      use_batch_norm = use_batch_norm
    )
  }
  
  outputs <- x %>% layer_dense(units = num_classes, activation = "softmax")
  model <- keras_model(inputs, outputs)
  
  opt <- switch(
    optimizer_name,
    "adam"    = optimizer_adam(learning_rate = learning_rate),
    "rmsprop" = optimizer_rmsprop(learning_rate = learning_rate),
    "nadam"   = optimizer_nadam(learning_rate = learning_rate),
    optimizer_adam(learning_rate = learning_rate)
  )
  
  model %>% compile(
    optimizer = opt,
    loss = "sparse_categorical_crossentropy",
    metrics = c("accuracy")
  )
  model
}
