##' Evaluate predictions with common regression metrics
##'
##' Compute Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and
##' R-squared for numeric `true_values` and `predictions` vectors.
##'
##' @param true_values Numeric vector of ground truth target values.
##' @param predictions Numeric vector of model predictions (same length).
##' @return A named list with elements `MSE`, `RMSE`, and `R2`.
##' @examples
##' res <- evaluate_predictions(c(1.0, 2.0, 3.0), c(0.9, 2.1, 2.8))
evaluate_predictions <- function(true_values, predictions) {
  # Calculate MSE
  mse <- mean((true_values - predictions)^2)
  # Calculate RMSE
  rmse <- sqrt(mse)
  # Calculate R-squared
  ss_total <- sum((true_values - mean(true_values))^2)
  ss_residual <- sum((true_values - predictions)^2)
  r_squared <- 1 - (ss_residual / ss_total)
  
  return(list(MSE = mse, RMSE = rmse, R2 = r_squared))
}
