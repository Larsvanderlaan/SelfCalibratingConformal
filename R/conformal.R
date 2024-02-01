

#' @export
iso_calibrator <- function(f, Y, max_depth = 12, min_child_weight = 5) {
  isoreg_with_xgboost(f, Y, max_depth = max_depth, min_child_weight = min_child_weight)
}


#' @export
binning_calibrator <- function(f, Y, nbin = 10) {
  num_bins <- nbin

  qs <- unique(quantile(f, seq(0, 1, length = num_bins + 1)))
  f_discrete <- qs[findInterval(f, qs, all.inside = TRUE)]
  bin_ids <- findInterval(f, qs, all.inside = TRUE)
  bin_ids_uniq <- sort(unique(bin_ids))
  binned_labels <- sapply(bin_ids_uniq, function(id) {
    mean(Y[bin_ids == id])
  })
  #f_cal <- binned_labels[match(bin_ids, bin_ids_uniq)]
  return(stepfun(qs[-length(qs)], c(binned_labels[length(binned_labels)], binned_labels )))
}




#' @param f_train Predictions on training set.
#' @param Y_train Outcome labels for training set.
#' @param f_test Predictions on validation set.
#' @param alpha 1 - coverage probability for prediction regions.
#' @param iso_max_depth Maximum tree depth for isotonic regression. Used internally for calibration.
#' @param iso_min_child_weight Minimum number of observation in each tree node of isotonic regression tree. Used internally for calibration.
#' @param num_bins_Y Number of bins for discretization of outcomes to generate approximate prediction regions.
#' @import data.table
#' @export
conformal_calibrator <- function(f_train, Y_train, f_test = f_train, calibrator = iso_calibrator, alpha = 0.1, num_bins_Y = 500, num_bins_f = num_bins_Y, ...) {
  library(data.table)

  f_test_uniq <- sort(unique(f_test)) # store unique labels of test predictions. Why?

    # discretize outcome
  grid_Y <- unique(quantile(Y_train, seq(0, 1, length = num_bins_Y), type = 1))
  #grid_Y <- sort(union(grid_Y, seq(min(Y_train), max(Y_train), length = 30)))
  mesh_error <- max(diff(grid_Y))
  print(mesh_error)
  # iterate for test points for which to predict outcomes
  grid_f <- unique(quantile(c(f_train, f_test), seq(0, 1, length = num_bins_f + 1), type = 1))
  f_discretizer <- function(f) {
    grid_f[findInterval(f, grid_f, all.inside = TRUE)]
  }
  f_train <- f_discretizer(f_train)
  f_test <- f_discretizer(f_test)

  output_data <- rbindlist(lapply(grid_f, function(f_uncal) {
    # construct augmented training data
    f_augment <- c(f_train, f_uncal)
    out <- rbindlist(lapply(grid_Y, function(label_Y) {
      Y_augment <- c(Y_train, label_Y)
      # Venn-Abers isotonic calibration
      calibrator_VA <- calibrator(f_augment, Y_augment, ...)
      # get calibrated prediction for test point and augmented dataset
      f_cal <- calibrator_VA(f_uncal)
      f_augment_cal <- calibrator_VA(f_augment)
      # compute conformity scores for level set
      conformity_scores_level_set <- abs(Y_augment[f_augment_cal == f_cal] - f_cal)

      radius_alpha <- quantile(conformity_scores_level_set, 1 - alpha, type = 1) #+ mesh_error

      data.table(label_Y = label_Y, f_cal = f_cal,
                 radius_alpha = radius_alpha,
                 in_region = 1*(abs(label_Y - f_cal) <= radius_alpha))

    }))
    # Use linear extrapolation
    out$f_test <- f_uncal

    ymin <- min(out$label_Y[out$in_region==1])
    ymax <- max(out$label_Y[out$in_region==1])
    f_cal_fun <- approxfun(out$label_Y, out$f_cal, rule = 2)
    radius_alpha_fun <-  approxfun(out$label_Y, out$radius_alpha, rule = 2)
    search_grid <- seq(ymin -  mesh_error,
                       ymax +  mesh_error,
                       length = 100
    )
    keep <- abs(search_grid - f_cal_fun(search_grid)) <= radius_alpha_fun(search_grid)

    interval <- range(search_grid[keep])

    out$lower <- min(ymin, interval[1])
    out$upper <- max(ymax, interval[2])
    return(out)
  }))






  prediction_point <- function(f, return_median = TRUE, return_score = FALSE, Y = NULL) {
    if(return_score && is.null(Y)) {
      stop("If return_score = TRUE then Y must not be null.")
    }

    f <- f_discretizer(f)
    # this isnt necessary but just in case
    f_nearest <- output_data$f_test[which.min(abs(f - output_data$f_test))]
    data <- output_data[output_data$f_test == f_nearest]
    f_cal <- unique(sort(data$f_cal))

    if(return_score) {
      scores <- abs(Y - f_cal)
      return(scores)
    }

    if(return_median) {
      f_cal <- quantile(f_cal, 0.5, type = 1)
    }
    return(f_cal)
  }
  prediction_point <- Vectorize(prediction_point)




  interval_data <- output_data[ , .(lower = min(lower), upper = max(upper)), by = f_test]
  # do one lower to handle mesh error
  #interval_data$lower <- grid_Y[pmax(findInterval(interval_data$lower, grid_Y)-1, 1)]
  #interval_data$upper <- grid_Y[pmin(findInterval(interval_data$upper, grid_Y)+1, length(grid_Y))]

  # move labels up and down one
  interval_data$f_cal <- prediction_point(interval_data$f_test, return_median = TRUE)

  prediction_region <- function(f){
    f_discr <- f_discretizer(f)
    new_data <- interval_data[match(f_discr, interval_data$f_test), ]
    new_data$f_test <- f_discr
    return(new_data)
  }

  #prediction_region <- Vectorize(prediction_region)

  output <- list(prediction_point = prediction_point, prediction_region = prediction_region, output_data = output_data, interval_data = interval_data, f_discretizer = f_discretizer)

  return(output)
}

#' @export
conformal_predict <- function(output, f) {
  interval_data <- output$prediction_region(f)[, c("f_test", "f_cal", "lower", "upper")]
  return(interval_data)
}




