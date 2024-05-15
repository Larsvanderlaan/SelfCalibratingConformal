import numpy as np
import pandas as pd
import math
from scipy.interpolate import interp1d
from SelfCalibratingConformal.calibrators import *
from SelfCalibratingConformal.utils import *

class MondrianCP:
  def __init__(self, predictor : callable, num_bins = 20, binning_method = "quantile"):
    self.predictor = predictor
    self.num_bins = num_bins
    self.binning_method = binning_method
  def calibrate(self, x_train : np.ndarray, y_train : np.ndarray,  alpha = 0.1):
    f_train = np.array(self.predictor(x_train)) 
    num_bins = min(self.num_bins, len(set(f_train)))  
    scores = abs(y_train - f_train)
    f_grid =  make_grid(f_train, num_bins, binning_method = self.binning_method)
    bin_index = match_grid_value(f_train, f_grid, return_index = True, all_inside = True)
    bin_ids = sorted(set(bin_index))
    alphas_adjusted = [min(1,math.ceil((1 - alpha)*(sum(bin_index == bin_id)+1)) / sum(bin_index == bin_id)) for bin_id in bin_ids]
    quantiles = [np.quantile(scores[bin_index == bin_id],alpha, method = 'inverted_cdf') for bin_id, alpha in zip(bin_ids, alphas_adjusted)]
    f_grid.pop()
    fit_info = pd.DataFrame([pd.Series(f_grid), pd.Series(quantiles)]).T
    fit_info.columns = ["prediction", "quantile"]
    self.fit_info = fit_info
    return None
  def predict(self, x : np.ndarray):
    predictions = np.array(self.predictor(x))
    return predictions
  def predict_interval(self, x : np.ndarray):
    fit_info = self.fit_info
    predictions = np.array(self.predictor(x))
    index_match = match_grid_value(predictions, fit_info.prediction, return_index = True, all_inside = True)
    quantiles =  fit_info.loc[index_match, "quantile"]
    output = pd.DataFrame([pd.Series([[f - q, f + q] for q, f in zip(quantiles, predictions)], index = range(0, len(predictions)))]).T
    output.columns = ["prediction_interval"]
    return output.loc[:,"prediction_interval"]
  


 

class kernelCP:
    def __init__(self, predictor: callable, num_bin_predictor=100):
        self.predictor = predictor
        self.num_bin_predictor = num_bin_predictor

    def calibrate(self, x_train: np.ndarray, y_train: np.ndarray, alpha=0.1, lambd = -1):
        f_train = np.array(self.predictor(x_train))
        f_grid = np.array(make_grid(f_train, self.num_bin_predictor))
        
        lbs, ubs, preds = run_conditional_kernel(f_train.reshape(-1,1), y_train.reshape(-1,1), 
                                                f_grid.reshape(-1,1),
                                                 predictor=lambda x: x,
                                                 alpha=alpha,
                                                 lambd=lambd)

        predictions_interval = np.hstack((lbs.reshape(-1,1), ubs.reshape(-1,1))) #np.array([[row[0], row[1]] for row in np.concatenate((lbs, ubs), axis = 1)])
        predictions_interval = predictions_interval.tolist()
        fit_info = pd.DataFrame({
            "f_grid": pd.Series(f_grid),
            "prediction_interval": predictions_interval
        })
        self.fit_info = fit_info

    def predict(self, x: np.ndarray):
        return np.array(self.predictor(x))

    def predict_interval(self, x: np.ndarray):
        f = np.array(self.predictor(x))
        f_grid = self.fit_info['f_grid']
        interval = np.array(self.fit_info['prediction_interval']).tolist()
        interval_lower = [row[0] for row in interval]
        interval_upper = [row[1] for row in interval]
        quadratic_interp_lower = interp1d(f_grid, interval_lower, kind='quadratic', bounds_error=False, fill_value="extrapolate" )
        quadratic_interp_upper = interp1d(f_grid, interval_upper, kind='quadratic', bounds_error=False, fill_value="extrapolate" )

        return np.array([quadratic_interp_lower(f), quadratic_interp_upper(f)]).T

