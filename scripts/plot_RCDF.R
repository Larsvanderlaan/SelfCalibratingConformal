
###########
# plots of RCDF of conformity scores
###########
library(ggplot2)
set.seed(12345)
d <- 5
n_train <- n_test <-  500
shape <- 4
b = 0.5
alpha <- 0.1
library(sl3)


get_scores_rcdf <- function(lrnr, lrnr_name) {
  data_list <- generate_data_splits(n_train, n_train, n_test, d = d, distr_shift = TRUE, shape = shape, b = b)
  data_train <- data_list$data_train; data_cal <- data_list$data_cal; data_test <- data_list$data_test
  X_train <- data_train$X; X_cal <- data_cal$X; X_test <- data_test$X
  Y_train <- data_train$Y; Y_cal <- data_cal$Y; Y_test <- data_test$Y
  # get predictor using learning algorithm specified by lrnr
  predictor <- train_predictor(X_train, Y_train, lrnr)

  f_cal <- predictor(X_cal)
  f_test <- predictor(X_test)
  out <- conformal_calibrator(f_train = f_cal, Y_train = Y_cal, f_test = f_test, calibrator = iso_calibrator, alpha = alpha, num_bins_Y = 100)


  data_bench <- generate_data_splits(n_train = 2, n_cal = 50000, 2, d = d, distr_shift = TRUE, shape = shape, b = b)$data_cal
  X_bench <- data_bench$X
  Y_bench <- data_bench$Y
  f_bench <- predictor(X_bench)

  f_bench_calibrated <- as.vector(unlist(out$prediction_point(f_bench, return_median = TRUE)))

  rmse_uncal <- sqrt(mean((Y_bench - f_bench)^2))
  rmse_cal <- sqrt(mean((Y_bench - f_bench_calibrated)^2))

  calerror_uncal <- compute_calibration_error(f_bench, Y_bench)
  calerror_cal <- compute_calibration_error(f_bench_calibrated, Y_bench)


  scores_calibrated <- as.vector(unlist(out$prediction_point(f_cal, return_score = TRUE, Y = Y_cal)))
  scores_uncalibrated <- as.vector(abs(Y_cal - f_cal))

  library(ggplot2)

  all_scores <- union(scores_calibrated, scores_uncalibrated)
  scores_grid <- seq(min(all_scores), max(all_scores), length = 100)
  df_cal <- data.table(scores = scores_grid)
  df_cal$cdf <- 1 - ecdf(scores_calibrated)(scores_grid)

  df_uncal <- data.table(scores = scores_grid)
  df_uncal$cdf <- 1 - ecdf(scores_uncalibrated)(scores_grid)

  df_cal$Status <- "Calibrated"
  df_uncal$Status <- "Uncalibrated"
  df_cal$calerror <- calerror_cal
  df_uncal$calerror <- calerror_uncal
  df_cal$rmse <- rmse_cal
  df_uncal$rmse <- rmse_uncal

  df <- rbind(df_cal, df_uncal)
  df$learner <-lrnr_name
  return(df)
}

scores_gam <- get_scores_rcdf(Lrnr_gam$new(), lrnr_name = "GAM")
scores_ranger <- get_scores_rcdf(Lrnr_ranger$new(), lrnr_name = "random forests")
scores_xgboost <- get_scores_rcdf(lrnr_xg, lrnr_name = "xgboost")

df <- rbind(scores_gam, scores_ranger, scores_xgboost)
# Plotting
plt <- ggplot(df, aes(x = scores, y = cdf, color = learner, linetype = Status)) +
  geom_line() +
  scale_color_manual(values = c("GAM" = "#1f77b4",  # A brighter, more vivid blue
                                "random forests" = "#d62728",  # A richer, more eye-catching red
                                "xgboost" = "#2ca02c")) +  # A more vibrant shade of green
labs(title = "Reverse Cumulative Distribution Function of Conformity Scores",
       x = "Conformity score",
       y = "RCDF",
       color = "",
       linetype = "") + theme_bw() +  theme(legend.position="bottom")

ggsave("plot_RCDF_scores.pdf")





