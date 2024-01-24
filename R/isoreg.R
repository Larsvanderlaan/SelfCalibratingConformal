#' Perform Univariate Isotonic Regression using XGBoost
#'
#' This function uses the XGBoost algorithm to perform isotonic regression
#' on univariate data.
#'
#' @param x A numeric vector or one-column matrix containing the predictor values.
#' @param y A numeric vector containing the response variable.
#' @param max_depth The maximum depth of the trees. Default is 15.
#' @param min_child_weight Minimum sum of instance weight (hessian) needed in a child.
#'                         Default is 10.
#'                         For equal weights, this is the minimal number of observations in each leaf node.
#'
#' @return A function that takes a vector or one-column matrix of predictor values
#'         and returns a function that maps input to the predictions fit using isotonic regression.
#' @examples
#' # Generate some example data
#' x <- seq(1, 100, by = 1)
#' y <- sort(runif(100))
#'
#' # Fit the isotonic model
#' isotonic_function <- isoreg_with_xgboost(x, y)
#'
#' # Use the fitted model to make predictions
#' predicted_values <- isotonic_function(x)
#'
#' @seealso \code{\link[xgboost::xgb.train]{xgb.train}}
#' @import xgboost
#' @export
isoreg_with_xgboost <- function(x, y, max_depth = 12, min_child_weight = 20) {
  # Prepare data for XGBoost
  data <- xgboost::xgb.DMatrix(data = as.matrix(x), label = as.vector(y))

  # Train isotonic regression model with XGBoost
  iso_fit <- xgboost::xgb.train(params = list(max_depth = max_depth,
                                              min_child_weight = min_child_weight,
                                              monotone_constraints = 1,
                                              eta = 1, gamma = 0,
                                              lambda = 0),
                                data = data, nrounds = 1)

  # Create function for prediction
  fun <- function(x) {
    data_pred <- xgboost::xgb.DMatrix(data = as.matrix(x))
    pred <- predict(iso_fit, data_pred)
    return(pred)
  }

  return(fun)
}
