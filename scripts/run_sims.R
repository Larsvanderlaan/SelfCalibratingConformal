

# Tuning parameters:
## Max_depth
##
library(conformal)
library(conformalInference)
library(data.table)

dir_path <- "~/repositories"
dir_path <- "~"
source(paste0(dir_path, "/conformal/scripts/sim_helpers.R"))



lrnr_xg <- Lrnr_sl$new(Stack$new(
  Lrnr_xgboost$new(max_depth = 3),
  Lrnr_xgboost$new(max_depth = 4),
  Lrnr_xgboost$new(max_depth = 5),
  Lrnr_xgboost$new(max_depth = 6),
  Lrnr_xgboost$new(max_depth = 7),
  Lrnr_xgboost$new(max_depth = 8),
  Lrnr_xgboost$new(max_depth = 9)
),
Lrnr_cv_selector$new(loss_squared_error))
lrnr_gam <- Lrnr_gam$new()
# parameters
## max-depth: 1, 3, 5, 7, 9
## n_cal: 100, 300, 500, 750, 1000
#
# get data splits



compute_calibration_error <- function(f, Y) {
  #data <- generate_data_splits(10000, 2, 2, d = d)$data_train
  #X <- data$X
  #Y <- data$Y
  f <- as.vector(f)
  Y <- as.vector(Y)
  lrnr <-  make_learner(Pipeline, Lrnr_cv$new(Stack$new(
    Lrnr_xgboost$new(max_depth = 3),
    Lrnr_xgboost$new(max_depth = 4),
    Lrnr_xgboost$new(max_depth = 5),
    Lrnr_xgboost$new(max_depth = 6),
    Lrnr_xgboost$new(max_depth = 7),
    Lrnr_xgboost$new(max_depth = 8),
    Lrnr_xgboost$new(max_depth = 9)
  )), Lrnr_cv_selector$new(loss_squared_error))

  task <- sl3_Task$new(data.table(f,Y), covariates = "f", outcome = "Y")
  f_cal <- lrnr$train(task)$predict(task)
  # debiased calibration error estimator of https://proceedings.mlr.press/v151/xu22c/xu22c.pdf
  calibration_error <- sqrt(max(mean((f_cal - f)*(Y - f)), 0))
  return(calibration_error)
}

run_sim_once <- function(n_train, lrnr, d, alpha, shape, n_test = n_train, b = 0.5) {
#Lrnr_xgboost$new(max_depth = max_depth)
  #lrnr <- Lrnr_gam$new()
  data_list <- generate_data_splits(n_train, n_train, n_test, d = d, distr_shift = TRUE, shape = shape, b = b)
  data_train <- data_list$data_train; data_cal <- data_list$data_cal; data_test <- data_list$data_test
  X_train <- data_train$X; X_cal <- data_cal$X; X_test <- data_test$X
  Y_train <- data_train$Y; Y_cal <- data_cal$Y; Y_test <- data_test$Y

  # get predictor using learning algorithm specified by lrnr
  predictor <- train_predictor(X_train, Y_train, lrnr)

  preds_oracle <- do_conformal_oracle(X_cal, Y_cal, X_test, predictor, alpha = alpha, data_test = data_test)
  print("histogram binning")
  preds_bin_10 <- do_conformal_calibration(X_cal, Y_cal, X_test, predictor, alpha = alpha, calibrator = binning_calibrator, nbin = 10)
  preds_bin_5 <- do_conformal_calibration(X_cal, Y_cal, X_test, predictor, alpha = alpha, calibrator = binning_calibrator, nbin = 5)
  print("mondrian")
  preds_mondrian_10 <- do_conformal_mondrian(X_cal, Y_cal, X_test, predictor, alpha = alpha, nbin = 10)
  preds_mondrian_5 <- do_conformal_mondrian(X_cal, Y_cal, X_test, predictor, alpha = alpha, nbin = 5)
  print("iso")
  preds_iso <- do_conformal_calibration(X_cal, Y_cal, X_test, predictor, alpha = alpha, calibrator = iso_calibrator)
  print("conditional")
  preds_cond <- do_conformal_conditional(X_cal, Y_cal, X_test, predictor, alpha = alpha)
  preds_marg <- do_conformal_marginal(X_cal, Y_cal, X_test, predictor, alpha = alpha)

  preds_oracle$method <- "oracle"
  preds_bin_10$method <- "cal_binning_10"
  preds_bin_5$method <- "cal_binning_5"
  preds_iso$method <- "isotonic"
  preds_cond$method <- "conditional"
  preds_marg$method <- "marginal"
  preds_mondrian_5$method <- "mondrian_10"
  preds_mondrian_10$method <- "mondrian_5"


  all_preds <- rbindlist(list(preds_oracle, preds_bin_10, preds_bin_5, preds_mondrian_10, preds_mondrian_5, preds_iso, preds_cond, preds_marg))
  nmethod <- nrow(all_preds) / nrow(preds_bin_10)
  all_preds$mu <- rep(data_test$mu, nmethod)
  all_preds$Y <- rep(Y_test, nmethod)
  all_preds$Z1 <- rep(data_test$Z1, nmethod)
  all_preds$Z0 <- rep(data_test$Z0, nmethod)
  threshold_upper <- quantile(data_train$Y, 0.8)
  threshold_lower <- quantile(data_train$Y, 0.2)
  treatment_rule <- function(f, lower, upper){
    # high risk, so only treat if above threshold.
    treatment <- 1- 1*(lower >= threshold_lower) * 1 * (upper <= threshold_upper)
    return(treatment)
  }
  all_preds$A <- treatment_rule(all_preds$f, all_preds$lower, all_preds$upper)
  all_preds$Y <- rep(data_test$Y , nmethod)


  # Extract bins for differences in the conditional variance.
  sigma2 <- data_test$sigma2
  bins_hetero <- findInterval(sigma2, quantile(sigma2, seq(0, 1 , length = 6)), all.inside = TRUE)
  all_preds$bin <- rep(bins_hetero, nmethod)
  setkey(all_preds, method, bin)
  print(dim(all_preds))
  all_preds <- all_preds[!is.na(all_preds$lower) & !is.na(all_preds$upper),]
  print(dim(all_preds))
  results_by_hetero <- all_preds[, .(coverage = mean(Y >= lower & Y <= upper), width = mean(width),
                                 risk = mean((Z1 * A + Z0 * (1-A))),
                                 rmse = sqrt(mean((f - mu)^2)),
                                 cal_error = compute_calibration_error(f, Y)
                                 )
                                 , by = c("method","bin")]
  results_marginal <- all_preds[, .(coverage = mean(Y >= lower & Y <= upper), width = mean(width),
                                risk = mean((Z1 * A + Z0 * (1-A))),
                                rmse = sqrt(mean((f - mu)^2)),
                                cal_error = compute_calibration_error(f, Y)
                                ),
                                , by = c("method")]

  results_marginal$bin <- "marginal"
  results_marginal <- results_marginal[, names(results_by_hetero), with = FALSE]
  results <- rbindlist(list(results_marginal, results_by_hetero))
  setkey(results, bin, method)

  return(results)
}









#
# if(lrnr_name == "gam") {
#   lrnr <- lrnr_gam
# } else if(lrnr_name == "xg") {
#   lrnr <- lrnr_xg
# } else if(lrnr_name == "ref") {
#   lrnr <- Lrnr_ranger$new()
# }
# d <- as.numeric(d)
# shape <- as.numeric(shape)
#
# n_train <- 1000
# alpha <- 0.05
# out <- run_sim_once(n_train = n_train, lrnr = lrnr, d = d, alpha = alpha, shape = shape)
# fwrite(out, file = paste0("sims_", n_train, "_", lrnr$name, "_", d, "_", alpha, "_", shape, ".csv"))
#
#
#
# d_list <- c(1, 2, 5, 10)
# lrnr_list <- list(lrnr_gam, lrnr_xg, Lrnr_ranger$new())
# shape_list <- c(1, 2 , 3 ,4)
# out_list <- list()
# for(d in d_list) {
#   for(lrnr in lrnr_list) {
#     for(shape in shape_list) {
#       n_train <- 1000
#       alpha <- 0.05
#       out <- run_sim_once(n_train = n_train, lrnr = lrnr, d = d, alpha = alpha, shape = shape, n_test = 10)
#       fwrite(out, file = paste0("sims_", n_train, "_", lrnr$name, "_", d, "_", alpha, "_", shape, ".csv"))
#       out_list <- c(out_list, list(out))
#     }
#   }
# }
