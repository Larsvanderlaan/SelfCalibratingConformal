


# Tuning parameters:
## Max_depth
##
library(conformal)
library(conformalInference)
library(data.table)


dir_path <- "~"
source(paste0(dir_path, "/conformal/scripts/sim_helpers.R"))


library(ggplot2)
#set.seed(12345)
d <- 5
n_train <- 1000
n_test <- 1000
b = 0.6
alpha <- 0.1
library(sl3)
set.seed(12345)
shape <- 3


get_widths_cal <- function(n_cal) {
  lrnr <- Lrnr_ranger$new()
  lrnr_name <- "random forests"
  # generate data splits
  data_list <- generate_data_splits(n_train, n_cal, n_test, d = d, distr_shift = TRUE, shape = shape, b = b)
  data_train <- data_list$data_train; data_cal <- data_list$data_cal; data_test <- data_list$data_test
  X_train <- data_train$X; X_cal <- data_cal$X; X_test <- data_test$X
  Y_train <- data_train$Y; Y_cal <- data_cal$Y; Y_test <- data_test$Y

  # train predictor
  predictor <- train_predictor(X_train, Y_train, lrnr)
  # get predictions
  f_cal <- predictor(X_cal)

  f_test <- predictor(X_test)

  # Do self-consistent conformal prediction
  out <- conformal_calibrator(f_train = f_cal, Y_train = Y_cal, f_test = f_test, calibrator = iso_calibrator, alpha = alpha, num_bins_Y = 100)

  # extract calibrated scores.
  scores_calibrated <- t(as.matrix(out$prediction_point(f_cal, return_score = TRUE, Y = Y_cal)))
  scores_calibrated <- apply(scores_calibrated, 1, max)
  scores_uncalibrated <- as.vector(abs(Y_cal - f_cal))
  width_cal <- quantile(scores_calibrated, ceiling(c(0.8, 0.9, 0.95) * (n_cal + 1))/n_cal)
  width_uncal <- quantile(scores_uncalibrated,ceiling(c(0.8, 0.9, 0.95) * (n_cal + 1))/n_cal)

  # get calibration error of initial predictor
  data_bench <- generate_data_splits(n_train = 2, n_cal = 10000, 2, d = d, distr_shift = TRUE, shape = shape, b = b)$data_cal
  X_bench <- data_bench$X
  Y_bench <- data_bench$Y
  f_bench <- predictor(X_bench)

  cal_error <- compute_calibration_error(f_bench, Y_bench)

  out <- data.table(n = n_cal, shape = shape, width = c(width_cal, width_uncal), alpha = rep(c(0.2, 0.1, 0.05), 2), Status = rep(c("Calibrated", "Uncalibrated"), each = 3), cal_error = cal_error)
  return(out)
}

results_all <- rbindlist(lapply(1:1000, function(iter) {
  try({
    print(iter)
    results <- rbindlist(lapply(c(20, 30, 40, 50, 75, 100, 200, 300, 500, 700, 1000), get_widths_cal) )
    results$iter <- iter
  })
  return(results)
}))

fwrite(results_all, "calerror_2.csv")
fwrite(results_all, paste0(dir_path, "/conformal/results/calerror_2.csv"))

#, 30, 40, 50, 75, 100, 200, 300, 500
results <- results_all[, .(width = mean(width)), by = c("n", "alpha", "Status")]


ggplot(results , aes(x = n, y = width, color = as.factor(alpha), linetype = Status)) + geom_line() + scale_x_log10()

library(latex2exp)
tmp <- results[, .(relative = width[Status=="Calibrated"] / width[Status=="Uncalibrated"]) , by = c("n", "alpha")]
ggplot(tmp , aes(x = n, y = relative, color = as.factor(alpha), linetype = as.factor(alpha))) + geom_line() +
  theme_bw() + labs(y = TeX("Relative Width (cal/uncal)"), x = TeX("$n_{cal}$"), color = TeX("$\\alpha$"), linetype = TeX("$\\alpha$")) + scale_x_log10() +
  geom_hline(yintercept = 1, color = "black", linetype = "dashed" , alpha = 0.5)




#fwrite(results, paste0(dir_path, "/conformal/results/calerror_2.csv"))

