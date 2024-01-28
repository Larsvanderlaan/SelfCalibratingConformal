



plot_curves <- function(n, b = 0.5) {
  library(ggplot2)
  alpha <- 0.1
  lrnr <- Lrnr_gam$new()
  n_test <- 1000
  data_list <- generate_data_splits(1000,  n, n_test = n_test, d = 1, distr_shift = TRUE, shape = 1, b = b)
  data_train <- data_list$data_train; data_cal <- data_list$data_cal; data_test <- data_list$data_test
  X_train <- data_train$X; X_cal <- data_cal$X; X_test <- data_test$X
  Y_train <- data_train$Y; Y_cal <- data_cal$Y; Y_test <- data_test$Y
  #X_test <- seq(0, 1, length = n_test)
  # get predictor using learning algorithm specified by lrnr
  predictor <- train_predictor(X_train, Y_train, lrnr)

  #
  preds_bin <- do_conformal_calibration(X_cal, Y_cal, X_test, predictor, alpha = alpha, calibrator = binning_calibrator, nbin = 10)
  preds_bin2 <- do_conformal_calibration(X_cal, Y_cal, X_test, predictor, alpha = alpha, calibrator = binning_calibrator, nbin = 5)
  preds_iso <- do_conformal_calibration(X_cal, Y_cal, X_test, predictor, alpha = alpha, calibrator = iso_calibrator)
  preds_cond <- do_conformal_conditional(X_cal, Y_cal, X_test, predictor, alpha = alpha, lambd = 1e-5)
  preds_marg <- do_conformal_marginal(X_cal, Y_cal, X_test, predictor, alpha = alpha)

  preds_bin$method <- "binning"
  preds_bin2$method <- "binning2"
  preds_iso$method <- "isotonic"
  preds_cond$method <- "conditional"
  preds_marg$method <- "marginal"

  preds_oracle <- copy(preds_cond)
  preds_oracle$f <- data_test$mu
  preds_oracle$lower <- qnorm(0.05, data_test$mu, data_test$sigma2)
  preds_oracle$upper <- qnorm(0.95, data_test$mu, data_test$sigma2)
  preds_oracle$method <- "Oracle"

  all_preds <- rbindlist(list(preds_iso, preds_cond, preds_marg))
  #all_preds <- preds_cond
  all_preds$X <- rep(as.vector(X_test), nrow(all_preds) / nrow(X_test))
  preds_oracle$X <- as.vector(X_test)
  preds_oracle$method <- NULL
  library(ggplot2)
  fwrite(all_preds, file = paste0(dir_path, "/conformal/plots/curves_", n, "_", b, ".csv"))



  plt <- ggplot(all_preds, aes(x = X, color = method)) +
    geom_step(aes(y = lower))  +
    geom_step(aes(y = upper)) +
    geom_line(data = preds_oracle, aes(x = X, y = lower), color = "black")  +
    geom_line(data = preds_oracle, aes(x = X, y = upper), color = "black") +
    theme(legend.position="bottom") + theme_bw() + labs(color = "")


  ggsave(plot = plt, filename =  paste0(dir_path, "/conformal/plots/curves_", n, "_", b, ".pdf") )
  return(plt)

}

plt1 <- plot_curves(50)

plt2 <- plot_curves(100)

plt3 <- plot_curves(300)

plt4 <- plot_curves(500)

plt5 <- plot_curves(1000)

plt6 <- plot_curves(5000)

