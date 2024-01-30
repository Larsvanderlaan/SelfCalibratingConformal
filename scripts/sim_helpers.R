
library(data.table)
library(sl3)
library(mvtnorm)
library(caret)

generate_data <- function(n, d = 5, shape = 3, b = 0.5, distr_shift = FALSE, a = 0,  ...) {
  # covariates
  X <- as.matrix(replicate(d, runif(n, 0, 1)))
  if(distr_shift) {
    X <- as.matrix(replicate(d, rbeta(n, 1, shape)))
  }
  colnames(X) <- paste0("X", 1:d)
  # biomarker mean and variance

  mu <-   rowMeans(X + sin(4*X))
  #sigma_range <- c(0.05, 0.4)^2
  g <- rowMeans(X + cos(4*X))
  sigma <-  0.035 + b * ( abs(mu)^6 / 20 - 0.02) + a * ( abs(g)^6 / 20 - 0.02)
  #plot(X, sqrt(sigma))
  #plot(mu, sqrt(sigma))f
  Y <- rnorm(n, mu, sigma)

   #Y <- qs[findInterval(Y, qs, all.inside = TRUE)]
  #median(Y)
  #plot(mu, Y)
  #plot(mu, sqrt(sigma))

  # potential outcomes being indicator of death
  thresh_upper <- quantile(Y, 0.8)
  thresh_lower <- quantile(Y, 0.2)
  eta <-  -1 + 2*rowMeans(sqrt(X) + cos(4*sqrt(X)))
  # treated, treating helps so long as blood-pressure is above threshold.


  Z  <- rbinom(n, 1, plogis(eta))
  # treat if extreme
  # two treatments based on extreme or nonextreme biomarker.
  # assigning the wrong treatment imp;ies death

  Z0 <- ifelse(Y >= thresh_upper | Y <= thresh_lower, 1,
               Z)
  #
  Z1 <- ifelse(Y >= thresh_upper | Y <= thresh_lower, Z,
               1)


  #Z1 <- rbinom(n, 1, plogis(eta + Y + (Y + 3)*(Y <= thresh)))
  # untreated, high blood pressure and not treating is bad
  #Z0 <- rbinom(n, 1, plogis(2.5+eta + Y  ))




  data <- list(X=X, Y = Y, Z1 = Z1, Z0 = Z0, mu = mu, sigma = sigma)
  return(data)
  #c(mean(Z1[Y <= thresh]), mean(Z1[Y > thresh]))
  #c(mean(Z0[Y <= thresh]), mean(Z0[Y > thresh]))
  #mean(Z1- Z0)

}

generate_data_splits <- function(n_train, n_cal, n_test, distr_shift = FALSE, shape = 1, b = 0.5, ...) {

  data_train <- generate_data(n_train, distr_shift = distr_shift, shape = shape, b= b, ...)
  data_cal <- generate_data(n_cal, shape = 1, b = b, ...)
  data_test <-generate_data(n_test, shape = 1, b = b, ...)

  return(list(data_train = data_train, data_cal = data_cal, data_test = data_test))
}

train_predictor <- function(X, Y, lrnr) {
  data_train <- data.table(X,Y)
  name_Y <- names(data_train)[ncol(data_train)]
  covariates <- names(data_train)[-ncol(data_train)]

  task <- sl3_Task$new(data_train, covariates = covariates, outcome = name_Y)
  lrnr_trained <- lrnr$train(task)

  task <- sl3_Task$new(as.data.table(X), covariates = covariates, outcome = c())
  preds <- lrnr_trained$predict(task)
  qs <- quantile(preds, seq(0,1, length = 100), type = 1)


  predictor <- function(x) {
    dat <- as.data.table(x)
    names(dat) <- covariates
    task <- sl3_Task$new(dat, covariates = covariates, outcome = c())
    preds <- lrnr_trained$predict(task)
    preds <- qs[findInterval(preds, qs, all.inside = TRUE)]
    return(as.matrix(preds))
  }


  return(predictor)
}



do_conformal_calibration <- function(X_cal, Y_cal, X_test, predictor, alpha, calibrator, ...) {
  f_cal <- predictor(X_cal)
  f_test <- predictor(X_test)
  out <- conformal_calibrator(f_train = f_cal, Y_train = Y_cal, f_test = f_test, calibrator = calibrator, alpha = alpha, ...)
  cf_preds <- conformal_predict(out, f_test)
  cf_preds$width <- cf_preds$upper - cf_preds$lower
  cf_preds$f <- cf_preds$f_cal
  cf_preds <- cf_preds[, c("f", "lower", "upper"), with = FALSE]
  cf_preds$width <- cf_preds$upper - cf_preds$lower
  return(cf_preds)
}

do_conformal_conditional <- function(X_cal, Y_cal, X_test, predictor, alpha, lambd = -1, ...) {
  library(reticulate)
  source_python(paste0(dir_path, "/conformal/scripts/condconf.py"))
  source_python(paste0(dir_path, "/conformal/scripts/crossval.py"))
  source_python(paste0(dir_path, "/conformal/scripts/sim_helpers.py"))
  out <- run_conditional_kernel(as.matrix(X_cal), as.matrix(Y_cal),  x_test = as.matrix(X_test), predictor = predictor, alpha = alpha, lambd = lambd)
  cf_preds <- as.data.table(do.call(cbind, out))
  names(cf_preds) <- c("lower", "upper", "f")
  cf_preds <- cf_preds[, c("f", "lower", "upper"), with = FALSE]
  cf_preds$width <- cf_preds$upper - cf_preds$lower
  return(cf_preds)
}

do_conformal_marginal <- function(X_cal, Y_cal, X_test, predictor, alpha, ...) {
  if(!require(conformalInference)) {
    devtools::install_github(repo="ryantibs/conformal", subdir="conformalInference")
  }
  X_cal_tmp <- rbind(X_cal[1, drop = FALSE], X_cal)
  Y_cal_tmp <- c(Y_cal[1], Y_cal)
  out <- conformalInference::conformal.pred.split(
    x = X_cal_tmp,
    y = Y_cal_tmp,
    x0 = as.matrix(X_test),
    rho = 1/nrow(X_cal_tmp),
    split = 1,
    alpha = alpha,
    train.fun = function(x, y) {
      return(NULL)
    },
    predict.fun = function(out, newx){
      predictor(newx)
    }
  )
  cf_preds <- data.table(f = as.vector(out$pred), lower = as.vector(out$lo), upper = as.vector(out$up))
  cf_preds$upper <- pmin(cf_preds$upper, max(Y_cal))
  cf_preds$lower <- pmax(cf_preds$lower, min(Y_cal))
  cf_preds$width = cf_preds$upper - cf_preds$lower
  return(cf_preds)
}


do_conformal_mondrian <- function(X_cal, Y_cal, X_test, predictor, alpha, nbin = 10, ...) {
  library(reticulate)
  source_python(paste0(dir_path, "/conformal/scripts/condconf.py"))
  source_python(paste0(dir_path, "/conformal/scripts/crossval.py"))
  source_python(paste0(dir_path, "/conformal/scripts/sim_helpers.py"))
  f_cal <- predictor(X_cal)
  f_test <- predictor(X_test)
  grid <- sort(unique(quantile(f_cal, seq(0, 1, length = nbin  + 1))))
  bin_id_cal <- factor(findInterval(f_cal, grid, all.inside = TRUE), levels = 1:nbin)
  bin_id_test <- factor(findInterval(f_test, grid, all.inside = TRUE), levels = 1:nbin)

  # apply split-conformal within bins of predictions
  cf_preds <- rbindlist(lapply(unique(bin_id_test), function(level) {
    index_cal <- which(bin_id_cal == level)
    index_test <- which(bin_id_test == level)
    X_cal_sub <- X_cal[index_cal, ,drop = FALSE]
    X_test_sub <- X_test[index_test, ,drop = FALSE]
    Y_cal_sub <- Y_cal[index_cal]


    cf_preds_sub <- do_conformal_marginal(X_cal_sub, Y_cal_sub, X_test_sub, predictor, alpha)
    cf_preds_sub$index <- index_test
    return(cf_preds_sub)
    }))
  # reorder so its back in original order
  cf_preds <- cf_preds[order(cf_preds$index),]
  # remove index column
  cf_preds$index <- NULL
  return(cf_preds)
}


do_conformal_oracle <- function(X_cal, Y_cal, X_test, predictor, alpha, data_test, ...) {

  f <-  as.vector(predictor(X_test))
  #sigma_range <- c(0.05, 0.4)^2
  mu <- data_test$mu
  sigma <- data_test$sigma



  radius <- sqrt(qchisq(1-alpha, df = 1, ncp = (f - mu)^2/sigma^2) * sigma^2)
  #lower_q <-  qnorm(alpha/ 2, mu , sigma)
  #upper_q <-   qnorm(1 - alpha/2, mu , sigma)
  #radius <- pmax(f - lower_q, upper_q - f)
  lower <- f - radius
  upper <- f + radius



  cf_preds <- data.table(f = f,
                         lower = lower,
                         upper = upper,
                         width = upper - lower)
  return(cf_preds)
}

#
#
# # parameters
# d <- 5 # covariate dimension
# alpha <- 0.05  # coverage probability
# lrnr <- Lrnr_gam$new()
#
# lrnr <- Lrnr_sl$new(Stack$new(
#   Lrnr_xgboost$new(max_depth = 3),
#   Lrnr_xgboost$new(max_depth = 4),
#   Lrnr_xgboost$new(max_depth = 5),
#   Lrnr_xgboost$new(max_depth = 6),
#   Lrnr_xgboost$new(max_depth = 7),
#   Lrnr_xgboost$new(max_depth = 8),
#   Lrnr_xgboost$new(max_depth = 9)
# ),
# Lrnr_cv_selector$new(loss_squared_error))
# lrnr <- Lrnr_gam$new()
# # parameters
# ## max-depth: 1, 3, 5, 7, 9
# ## n_cal: 100, 300, 500, 750, 1000
# #
# # get data splits
# data_list <- generate_data_splits(1000, 1000, 1000, d = d)
# data_train <- data_list$data_train; data_cal <- data_list$data_cal; data_test <- data_list$data_test
# X_train <- data_train$X; X_cal <- data_cal$X; X_test <- data_test$X
# Y_train <- data_train$Y; Y_cal <- data_cal$Y; Y_test <- data_test$Y
#
# # get predictor using learning algorithm specified by lrnr
# predictor <- train_predictor(X_train, Y_train, lrnr)
#
# plot(predictor(X_train), data_train$mu)
# #
# preds_bin <- do_conformal_calibration(X_cal, Y_cal, X_test, predictor, alpha = alpha, calibrator = binning_calibrator, nbin = 10)
# #preds_bin2 <- do_conformal_calibration(X_cal, Y_cal, X_test, predictor, alpha = alpha, calibrator = binning_calibrator, nbin = 50)
# preds_iso <- do_conformal_calibration(X_cal, Y_cal, X_test, predictor, alpha = alpha, calibrator = iso_calibrator)
# #preds_cond <- do_conformal_conditional(X_cal, Y_cal, X_test, predictor, alpha = alpha)
# preds_marg <- do_conformal_marginal(X_cal, Y_cal, X_test, predictor, alpha = alpha)
#
# preds_bin$method <- "binning"
# #preds_bin2$method <- "binning2"
# preds_iso$method <- "isotonic"
# #preds_cond$method <- "conditional"
# preds_marg$method <- "marginal"
#
#
# all_preds <- rbindlist(list(preds_bin, preds_iso, preds_marg))
# nmethod <- nrow(all_preds) / nrow(preds_bin)
# all_preds$Y <- rep(Y_test, nmethod)
# all_preds$Z1 <- rep(data_test$Z1, nmethod)
# all_preds$Z0 <- rep(data_test$Z0, nmethod)
# # Extract bins for differences in the conditional variance.
# sigma <- data_test$sigma
# bins_hetero <- findInterval(sigma, quantile(sigma, seq(0, 1 , length = 6)), all.inside = TRUE)
# all_preds$bin <- rep(bins_hetero, nmethod)
# setkey(all_preds, method, bin)
# all_preds[, .(mean(Y >= lower & Y <= upper), mean(width)), by = c("method","bin")]
#
# all_preds[, .(mean(Y >= lower & Y <= upper), mean(width)), by = c("method")]
#
#
# out <- rbindlist(lapply(quantile(data_train$Y, seq(0.7,0.9, length = 10)), function(threshold) {
#   treatment_rule <- function(f, lower, upper){
#     # high risk, so only treat if above threshold.
#     treatment <- 1*(upper <= threshold)
#     return(treatment)
#   }
#
#   all_preds$A <- treatment_rule(all_preds$f, all_preds$lower, all_preds$upper)
#   out <- all_preds[, mean(Z1 * A + Z0 * (1-A)), by = method]
#   print(out)
#   out$threshold <- threshold
#   return(out)
# }))
#
# plot(out$threshold[out$method=="marginal"], out$V1[out$method=="marginal"], type = "l", col = "blue")
#
# lines(out$threshold[out$method=="isotonic"], out$V1[out$method=="isotonic"], col = "red")
#
#
#
# threshold_upper <- quantile(data_train$Y, 0.8)
# threshold_lower <- quantile(data_train$Y, 0.2)
#
# treatment_rule <- function(f, lower, upper){
#   # high risk, so only treat if above threshold.
#   treatment <- 1- 1*(lower >= threshold_lower) * 1 * (upper <= threshold_upper)
#   return(treatment)
# }
# Ytrue <- data_test$Y
#
# all_preds$A <- treatment_rule(all_preds$f, all_preds$lower, all_preds$upper)
# #all_preds$Atrue <- rep(1*(data_test$Y <= threshold), 3)
# all_preds$Y <- rep(data_test$Y , 3)
#
#
#
# out <- all_preds[, mean((Z1 * A + Z0 * (1-A))), by = method]
# print(out)
#
