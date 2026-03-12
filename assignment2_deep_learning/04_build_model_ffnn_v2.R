# R/04_build_model_ffnn_v2.R
library(keras3)

add_dense_block <- function(x,
                            units,
                            activation = "relu",
                            leaky_alpha = 0.1,
                            dropout_rate = 0.0,
                            l2_lambda = 0.0,
                            use_batch_norm = FALSE) {
  
  reg <- if (l2_lambda > 0) regularizer_l2(l2_lambda) else NULL
  
  # Dense without activation first (so we can support leaky_relu properly)
  x <- x %>%
    layer_dense(
      units = units,
      activation = "linear",
      kernel_regularizer = reg
    )
  
  if (use_batch_norm) {
    x <- x %>% layer_batch_normalization()
  }
  
  # Activation
  if (activation == "leaky_relu") {
    x <- x %>% layer_activation_leaky_relu(negative_slope = leaky_alpha)
  } else {
    x <- x %>% layer_activation(activation = activation)
  }
  
  if (dropout_rate > 0) {
    x <- x %>% layer_dropout(rate = dropout_rate)
  }
  
  x
}

build_ffnn_v2 <- function(input_dim,
                          units1 = 64,
                          units2 = 32,
                          units3 = 16,
                          units4 = 8,
                          units5 = 4,
                          use_second_layer = TRUE,
                          use_third_layer  = FALSE,
                          use_fourth_layer = FALSE,
                          use_fifth_layer  = FALSE,
                          activation = "relu",        # relu | elu | tanh | leaky_relu
                          leaky_alpha = 0.1,
                          dropout_rate = 0.0,
                          l2_lambda = 0.0,
                          use_batch_norm = FALSE,
                          learning_rate = 1e-3,
                          optimizer_name = "adam",    # adam | rmsprop | nadam
                          num_classes) {
  
  inputs <- layer_input(shape = input_dim)
  
  x <- add_dense_block(
    inputs,
    units = units1,
    activation = activation,
    leaky_alpha = leaky_alpha,
    dropout_rate = dropout_rate,
    l2_lambda = l2_lambda,
    use_batch_norm = use_batch_norm
  )
  
  if (use_second_layer) {
    x <- add_dense_block(x, units2, activation, leaky_alpha, dropout_rate, l2_lambda, use_batch_norm)
  }
  if (use_third_layer) {
    x <- add_dense_block(x, units3, activation, leaky_alpha, dropout_rate, l2_lambda, use_batch_norm)
  }
  if (use_fourth_layer) {
    x <- add_dense_block(x, units4, activation, leaky_alpha, dropout_rate, l2_lambda, use_batch_norm)
  }
  if (use_fifth_layer) {
    x <- add_dense_block(x, units5, activation, leaky_alpha, dropout_rate, l2_lambda, use_batch_norm)
  }
  
  outputs <- x %>%
    layer_dense(units = num_classes, activation = "softmax")
  
  model <- keras_model(inputs = inputs, outputs = outputs)
  
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
