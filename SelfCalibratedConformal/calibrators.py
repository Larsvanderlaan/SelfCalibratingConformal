import numpy as np
 


def calibrator_isotonic(f : np.ndarray, y : np.ndarray, max_depth = 20, min_child_weight = 20):
  # Compute isotonic regression of labels 'y' on model predictions 'f'
  # 'calibrator' is a 1D function that maps original model predictions to calibrated model predictions.
  data = xgb.DMatrix(data=f.reshape(-1, 1), label=y)
  # Train isotonic regression model with XGBoost
  iso_fit = xgb.train(params={'max_depth': max_depth,
                                'min_child_weight': min_child_weight,
                                'monotone_constraints': '(1)',
                                'eta': 1, 'gamma': 0,
                                'lambda': 0},
                        dtrain=data, num_boost_round=1)
  # Create function for prediction
  def transform(x):
        data_pred = xgb.DMatrix(data=x.reshape(-1, 1))
        pred = iso_fit.predict(data_pred)
        return pred
  return transform
 
def calibrator_CART(f : np.ndarray, y : np.ndarray, max_depth = 10, min_child_weight = 50):
  # Compute isotonic regression of labels 'y' on model predictions 'f'
  # 'calibrator' is a 1D function that maps original model predictions to calibrated model predictions.
  data = xgb.DMatrix(data=f.reshape(-1, 1), label=y)
  # Train isotonic regression model with XGBoost
  iso_fit = xgb.train(params={'max_depth': max_depth,
                                'min_child_weight': min_child_weight,
                                'eta': 1, 'gamma': 0,
                                'lambda': 0},
                        dtrain=data, num_boost_round=1)
  # Create function for prediction
  def transform(x):
        data_pred = xgb.DMatrix(data=x.reshape(-1, 1))
        pred = iso_fit.predict(data_pred)
        return pred
  return transform


def calibrator_histogram(f : np.ndarray, y : np.ndarray, num_bin = 10, binning_method = "quantile"):
  # Use quantile-binning 'num_bin' equal-frequency bins of 'f' using quantile-binning.
  grid = make_grid(f, num_bin, binning_method = binning_method)
  bin_ids = match_grid_value(f, grid, return_index = True, all_inside = True)
  bin_preds = [ np.mean(y[bin_ids == bin_id]) for bin_id in sorted(set(bin_ids))]
   
  def transform(x):
    bin_ids = match_grid_value(x, grid, return_index = True, all_inside = True)
    values = [bin_preds[bin_id] for bin_id in bin_ids]
    return np.array(values)
  
  return transform
