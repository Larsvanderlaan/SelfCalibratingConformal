import numpy as np
import xgboost as xgb
from SelfCalibratingConformal.utils import *


def calibrator_isotonic(f: np.ndarray, y: np.ndarray, max_depth=20, min_child_weight=20):
    """
    Creates a 1D calibration function based on isotonic regression using XGBoost. This function
    fits an XGBoost model to predict `y` from `f` ensuring a monotonic relationship.

    Args:
        f (np.ndarray): Array of uncalibrated predictions (features).
        y (np.ndarray): Array of actual outcomes (labels).
        max_depth (int, optional): Maximum depth of each tree used in the XGBoost model. Defaults to 20.
        min_child_weight (int, optional): Minimum sum of instance weight needed in a child node. Defaults to 20.

    Returns:
        function: A function that takes an array of model predictions and returns calibrated predictions.
    """
    data = xgb.DMatrix(data=f.reshape(-1, 1), label=y)
    iso_fit = xgb.train(params={
        'max_depth': max_depth,
        'min_child_weight': min_child_weight,
        'monotone_constraints': '(1)',
        'eta': 1, 'gamma': 0,
        'lambda': 0
    }, dtrain=data, num_boost_round=1)

    def transform(x):
        data_pred = xgb.DMatrix(data=x.reshape(-1, 1))
        pred = iso_fit.predict(data_pred)
        return pred

    return transform

def calibrator_CART(f: np.ndarray, y: np.ndarray, max_depth=10, min_child_weight=50):
    """
    Trains a non-isotonic regression model using XGBoost, creating a calibration function that
    maps model predictions to calibrated outputs without enforcing a monotonic relationship.

    Args:
        f (np.ndarray): Array of uncalibrated predictions (features).
        y (np.ndarray): Array of actual outcomes (labels).
        max_depth (int, optional): Maximum depth of each tree used in the XGBoost model. Defaults to 10.
        min_child_weight (int, optional): Minimum sum of instance weight needed in a child node. Defaults to 50.

    Returns:
        function: A function that takes an array of model predictions and returns calibrated predictions.
    """
    data = xgb.DMatrix(data=f.reshape(-1, 1), label=y)
    cart_fit = xgb.train(params={
        'max_depth': max_depth,
        'min_child_weight': min_child_weight,
        'eta': 1, 'gamma': 0,
        'lambda': 0
    }, dtrain=data, num_boost_round=1)

    def transform(x):
        data_pred = xgb.DMatrix(data=x.reshape(-1, 1))
        pred = cart_fit.predict(data_pred)
        return pred

    return transform

def calibrator_histogram(f: np.ndarray, y: np.ndarray, num_bin=10, binning_method="quantile"):
    """
    Creates a calibration function based on histogram binning. It divides the prediction space into
    bins and assigns the mean of actual outcomes within each bin as the calibrated prediction.

    Args:
        f (np.ndarray): Array of uncalibrated predictions.
        y (np.ndarray): Array of actual outcomes.
        num_bin (int, optional): Number of bins for histogram binning. Defaults to 10.
        binning_method (str, optional): Method for creating bins ('quantile' or 'fixed'). Defaults to "quantile".

    Returns:
        function: A function that maps original predictions to calibrated predictions based on the bin averages.
    """
    grid = make_grid(f, num_bin, binning_method=binning_method)
    bin_ids = match_grid_value(f, grid, return_index=True, all_inside=True)
    bin_preds = [np.mean(y[bin_ids == bin_id]) for bin_id in sorted(set(bin_ids))]

    def transform(x):
        bin_ids = match_grid_value(x, grid, return_index=True, all_inside=True)
        values = [bin_preds[bin_id] for bin_id in bin_ids]
        return np.array(values)

    return transform
