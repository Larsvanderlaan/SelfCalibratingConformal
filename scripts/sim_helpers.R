
library(data.table)
library(sl3)
library(mvtnorm)

generate_data <- function(n, d = 5, shape = 3, distr_shift = FALSE, cond_var_type = 1, ...) {
  # covariates
  X <- as.matrix(replicate(d, runif(n, 0, 1)))
  if(distr_shift) {
    X <- as.matrix(replicate(d, rbeta(n, 1, shape)))
  }
  colnames(X) <- paste0("X", 1:d)
  # biomarker mean and variance
  mu <-    rowMeans(X + sin(4*X))
  sigma_range <- c(0.05, 0.4)^2
  if (cond_var_type == 1) {
    sigma2 <-  abs(mu)^6
  } else if (cond_var_type == 2) {
    sigma2 <-  abs(rowMeans(X))^6
  }

  sigma2 <-  sigma_range[1] +  sigma_range[2] * (sigma2 - min(sigma2))/(diff(range(sigma2)))
  # biomarker
  Y <- rnorm(n, mu, sigma2)
   #Y <- qs[findInterval(Y, qs, all.inside = TRUE)]
  median(Y)
  plot(mu, Y)
  plot(mu, sqrt(sigma2))

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




  data <- list(X=X, Y = Y, Z1 = Z1, Z0 = Z0, mu = mu, sigma2 = sigma2)
  return(data)
  #c(mean(Z1[Y <= thresh]), mean(Z1[Y > thresh]))
  #c(mean(Z0[Y <= thresh]), mean(Z0[Y > thresh]))
  #mean(Z1- Z0)

}

generate_data_splits <- function(n_train, n_cal, n_test, distr_shift = FALSE, shape = 1, ...) {

  data_train <- generate_data(n_train, distr_shift = distr_shift, shape = shape, ...)
  data_cal <- generate_data(n_cal, ...)
  data_test <-generate_data(n_test, ...)

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

do_conformal_conditional <- function(X_cal, Y_cal, X_test, predictor, alpha, ...) {
  library(reticulate)
  source_python(paste0(dir_path, "/conformal/scripts/condconf.py"))
  source_python(paste0(dir_path, "/conformal/scripts/crossval.py"))
  out <- run_fun(as.matrix(X_cal), as.matrix(Y_cal),  x_test = as.matrix(X_test), predictor = predictor, alpha = alpha)
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
  cf_preds$width = cf_preds$upper - cf_preds$lower
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
# sigma2 <- data_test$sigma2
# bins_hetero <- findInterval(sigma2, quantile(sigma2, seq(0, 1 , length = 6)), all.inside = TRUE)
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
