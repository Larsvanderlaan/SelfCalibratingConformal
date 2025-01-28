import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline, interp1d
from statsmodels.gam.smooth_basis import BSplines
from statsmodels.gam.generalized_additive_model import GLMGam
from statsmodels.nonparametric.kernel_regression import KernelReg
import matplotlib.pyplot as plt
# Package imports
from selfcalibratingconformal.calibrators import *
from selfcalibratingconformal.utils import *
 
class SelfCalibratingConformalPredictor:
    def __init__(self, predictor: callable, calibrator = calibrator_isotonic, 
                 calibrator_params = {'max_depth': 12, 'min_child_weight': 20},
                 algo_params = {'num_bin_predictor': 100, 'num_bin_y': 80, 'binning_method': "quantile"}):
        """
        Initializes a SelfCalibratingConformal predictor which estimates prediction intervals using
        various calibration methods based on the provided predictor and calibration function.

        Parameters:
        predictor (callable): Function for making point predictions.
        calibrator (callable): Calibration function to adjust predictor outputs, defaulting to isotonic calibration.
        calibrator_params (dict): Parameters for the calibration function.
        algo_params (dict): Algorithmic parameters including:
            num_bin_predictor (int): Number of bins for predictor values for grid approximation.
            num_bin_y (int): Number of bins for output values (y) for grid approximation.
            binning_method (str): Binning strategy, either 'quantile' for equal-frequency bins or 'fixed' for equal-width bins.
        """
        self.predictor = predictor
        self.calibrator = calibrator
        self.calibrator_params = calibrator_params
        self.num_bin_predictor = algo_params['num_bin_predictor']
        self.num_bin_y = algo_params['num_bin_y']
        self.binning_method = algo_params['binning_method']

    def calibrate(self, x_train: np.ndarray, y_train: np.ndarray, alpha=0.1, y_range=None, scoring_method="calibrated", hist_shrinkage_num_bin=5):
        """
        Calibrates the predictor based on training data.

        Parameters:
        - x_train (np.ndarray): Feature/covariate data for calibration.
        - y_train (np.ndarray): Outcome variable data for calibration.
        - alpha (float): Miscoverage/significance level for the prediction intervals.
        - y_range (tuple, optional): Range of y values for binning. If None, it will be calculated from y_train.
        - scoring_method (str): Type of conformity scores used. Options include:
          - 'calibrated' (default): Uses absolute residual conformity scores centered around the isotonic calibrated predictions.
          - 'debiased': Employs absolute residual conformity scores centered around a debiased version of the original model predictions. 
            This debiasing is achieved by subtracting the observed bias within each bin of the isotonic calibration solution. 
            This method may result in smoother prediction bands compared to the 'calibrated' method; however, the point predictions used in calculating the conformity scores could be less accurately calibrated. 
            This approach can be particularly beneficial when calibration data is limited, as it may better leverage the predictive strength of the original model.
          - 'original': Uses absolute residual conformity scores centered around the original, uncalibrated model predictions.
        - hist_shrinkage_num_bin (int, optional): Specifies the number of bins used in histogram binning calibration. This parameter helps derive calibrated point predictions from the Venn-Abers multi-prediction by adjusting the midpoint towards a histogram-binned calibrated prediction. The degree of shrinkage is proportional to the width of the Venn-Abers multi-prediction, reflecting the uncertainty in calibrating the midpoint prediction.
        """
        # Store calibration settings internally for reference.
        self.settings = {'x_train': x_train, 'y_train': y_train, 'scoring_method': scoring_method, 'alpha': alpha}
        
        if y_range is None:
            y_range = [min(y_train), max(y_train)]

        y_grid = make_grid(y_train, self.num_bin_y, y_range, binning_method=self.binning_method)
        y_interp = make_grid(y_train, 1000, y_range, binning_method="quantile")
        preds_train = np.array(self.predictor(x_train))
        preds_grid = make_grid(preds_train, self.num_bin_predictor, binning_method=self.binning_method)

        preds_grid_indices = list(range(len(preds_grid)))
        multipreds_venn_abers_grid = pd.Series([[] for _ in preds_grid], index=preds_grid_indices)
        predictions_interval = pd.Series([[] for _ in preds_grid], index=preds_grid_indices)

        list_preds_augmented = [np.hstack([preds_train, pred]) for pred in preds_grid]
        list_y_augmented = [np.hstack([y_train, y_val]) for y_val in y_grid]

        def process_prediction(index_pred, pred, list_preds_augmented, list_y_augmented):
            preds_augmented = list_preds_augmented[index_pred]
            multipred_venn_abers = np.zeros(len(y_grid))
            thresholds = np.zeros(len(y_grid))
            test_scores = np.zeros(len(y_grid))
            
            for index_y, y_val in enumerate(y_grid):
                y_augmented = list_y_augmented[index_y]
                calibrator = self.calibrator(f=preds_augmented, y=y_augmented, **self.calibrator_params)
                preds_augmented_calibrated = calibrator(preds_augmented)
                pred_calibrated = preds_augmented_calibrated[-1]
                
                level_set = np.where(preds_augmented_calibrated == pred_calibrated)[0]
                conformity_scores = self._compute_conformity_scores(y_augmented[level_set], pred_calibrated, preds_augmented[level_set], scoring_method)
                threshold = np.quantile(conformity_scores, 1 - alpha, method='inverted_cdf')
                test_score = conformity_scores[-1]
                
                test_scores[index_y] = test_score
                thresholds[index_y] = threshold
                multipred_venn_abers[index_y] = pred_calibrated
            
            test_scores_interp = np.interp(y_interp, y_grid, test_scores)
            thresholds_interp = np.interp(y_interp, y_grid, thresholds)
            prediction_set = [y for y, score, threshold in zip(y_interp, test_scores_interp, thresholds_interp) if score <= threshold]
            predictions_interval[index_pred] = [min(prediction_set), max(prediction_set)]
            multipreds_venn_abers_grid[index_pred] = multipred_venn_abers

        for index_pred, pred in enumerate(preds_grid):
            process_prediction(index_pred, pred, list_preds_augmented, list_y_augmented)

        baseline_prediction = calibrator_histogram(preds_train, y_train, num_bin=hist_shrinkage_num_bin)(preds_grid)
        y_max, y_min = max(y_train), min(y_train)
        predictions_point = [(max(value) + min(value)) / 2 + (max(value) - min(value)) / (y_max - y_min) * 
                             (baseline_prediction[index] - (max(value) + min(value)) / 2) for index, value in enumerate(multipreds_venn_abers_grid)]
        predictions_venn_abers = [[min(value), max(value)] for value in multipreds_venn_abers_grid]

        fit_info_conformal = pd.DataFrame({
            "prediction_uncal": pd.Series(preds_grid, index=preds_grid_indices),
            "prediction_cal": pd.Series(predictions_point),
            "prediction_venn_abers": pd.Series(predictions_venn_abers),
            "prediction_interval": predictions_interval
        })

        self.fit_info = fit_info_conformal
        
    def _compute_conformity_scores(self, y_values, calibrated_prediction, original_predictions, scoring_method):
        """
        Computes the conformity scores based on the specified scoring method.

        :param y_values: The actual outcome values (np.ndarray).
        :param calibrated_prediction: The predicted value after calibration (float).
        :param original_predictions: Predictions before calibration (np.ndarray).
        :param scoring_method: Specifies the method to compute conformity scores. Options are:
          - 'calibrated': Uses the absolute difference between y_values and calibrated_prediction.
          - 'debiased': Uses the absolute difference between y_values and a debiased version of the original predictions. Debiased predictions are calculated by adjusting the original predictions to match the calibrated prediction on average.
          - 'original': Uses the absolute difference between y_values and original_predictions.

        :return: An array of conformity scores (np.ndarray).
        """
      
        if scoring_method == "calibrated":
          return abs(y_values - calibrated_prediction)
        elif scoring_method == "debiased":
          return abs(y_values - (original_predictions - np.mean(original_predictions) + calibrated_prediction))
        elif scoring_method == "original":
          return abs(y_values - original_predictions)
        else:
          return abs(y_values - calibrated_prediction)
        
    def predict_point(self, x: np.ndarray, calibrate = True, smooth = False):
        """
        Generates a point prediction for given features, optionally using calibration.
        
        Args:
            x (np.ndarray): Input features.
            calibrate (bool): If True, apply calibration to the prediction. Defaults to True.
            smooth (bool): If True, apply smoothing to the calibrated prediction. Defaults to False.
        
        Returns:
            np.ndarray: Predicted values.
        """
        f = np.array(self.predictor(x))
        if calibrate:
          return self._extrapolate(self.fit_info['prediction_uncal'], self.fit_info['prediction_cal'], f, smooth = smooth)
        else:
         return f

    def predict_venn_abers(self, x: np.ndarray, smooth = False):
        """
        Provides a range of predictions (Venn-Abers intervals) for given features.
        
        Args:
            x (np.ndarray): Input features.
            smooth (bool): If True, apply smoothing to the prediction intervals. Defaults to False.
        
        Returns:
            np.ndarray: Array containing lower and upper bounds of predictions.
        """
        f = np.array(self.predictor(x))
        f_grid = self.fit_info['prediction_uncal']
        bounds = [(row[0], row[1]) for row in self.fit_info['prediction_venn_abers']]
        lower = self._extrapolate(f_grid, [b[0] for b in bounds], f, smooth = smooth)
        upper = self._extrapolate(f_grid, [b[1] for b in bounds], f, smooth = smooth)

        return np.array([lower, upper]).T
      
    def predict_interval(self, x: np.ndarray, smooth = False):
        """
        Outputs prediction intervals for the given input features.
        
        Args:
            x (np.ndarray): Input features.
            smooth (bool): If True, smoothing is applied to the intervals. Defaults to False.
        
        Returns:
            np.ndarray: Array containing lower and upper bounds of the interval predictions.
        """
        f = np.array(self.predictor(x))
        #index_match = match_grid_value(f, self.fit_info['prediction_uncal'], return_index=True)
        #np.array(self.fit_info.loc[index_match, 'prediction_interval'])
        f_grid = self.fit_info['prediction_uncal']
        bounds = [(row[0], row[1]) for row in self.fit_info['prediction_interval']]
        lower = self._extrapolate(f_grid, [b[0] for b in bounds], f, smooth = smooth)
        upper = self._extrapolate(f_grid, [b[1] for b in bounds], f, smooth = smooth)

        return np.array([lower, upper]).T

    def _extrapolate(self, x_grid, y_grid, x_new, smooth=False):
      """
        Performs extrapolation or smoothing on a given set of x values based on provided data grids.
        
        Args:
            x_grid (array-like): The grid of x-values (1D) for which y-values are known.
            y_grid (array-like): The corresponding y-values for the x-values in x_grid.
            x_new (array-like): The new x-values on which to perform extrapolation or smoothing.
            smooth (bool, optional): If True, performs smoothing using locally linear kernel regression. 
            Otherwise, nearest neighbor interpolation is performed.
        
        Returns:
            np.ndarray: The extrapolated or smoothed y-values corresponding to x_new.
      """
      y_grid = np.array(y_grid)
      if not smooth:
          interp = interp1d(x_grid, y_grid, kind='nearest', bounds_error=False, fill_value="extrapolate")
          preds = interp(x_new)
      else:
          # Ensure input arrays are numpy arrays and correctly shaped
          x_grid = np.array(x_grid).reshape(-1, 1)
          y_grid = np.array(y_grid)  # Ensure y_grid is a numpy array
          x_new = np.array(x_new).reshape(-1, 1)
          smoother = KernelReg(y_grid, x_grid, var_type='c')
          smoother.fit()
          preds, std_dev = smoother.fit(x_new)
          # bs = BSplines(x_grid, df=[15], degree=[3])
          # #arg_smoothing {'df': [20], 'degree': [1]}
          # gam = GLMGam(y_grid, smoother=bs)
          # gam.fit()
          # gam.select_penweight(criterion='aic') 
          # gam_results = gam.fit()
          # x_new = x_new[(x_new >= np.min(x_grid)) & (x_new <= np.max(x_grid))]
          # preds = gam_results.predict(bs.transform(x_new))
    
      return np.array(preds)

    def check_coverage(self, x_test, y_test, boolean=None, smooth=False):
      """
        Computes how frequently actual y_test values fall within the predicted intervals.
        
        Args:
            x_test (array-like): Input features for the test dataset.
            y_test (array-like): Actual target values for the test dataset.
            boolean (array-like, optional): Specifies which indices are considered in calculations.
            smooth (bool, optional): Whether to apply smoothing to interval predictions.
        
        Returns:
            list: Coverage percentage and median interval width.
      """
      # Predict intervals using the model's method, possibly applying smoothing
      intervals = self.predict_interval(x_test, smooth=smooth)
    
      # Create a list of indicators (1 or 0) where 1 indicates the actual y_test value
      # falls within the predicted interval. Consider only entries specified by the
      # `boolean` array if it is not None.
      indicators = [
          lower <= y_test[index] <= upper
          for index, (lower, upper) in enumerate(intervals)
          if boolean is None or boolean[index] == 1
      ]

      # Calculate the coverage as the mean of the indicators
      coverage = np.mean(indicators)
    
      # Calculate the median width of the intervals that are included in the coverage calculation
      width = np.median([upper - lower for index, (lower, upper) in enumerate(intervals)
                       if boolean is None or boolean[index] == 1])

      return [coverage, width]

    def plot(self, x = None, y = None, smooth = False):
      """
        Plots the predictions, actual outcomes, and prediction intervals for a given set of data.
        
        Args:
            x (array-like, optional): Features data; if None, uses training data.
            y (array-like, optional): Actual outcomes; if None, uses training outcomes.
            smooth (bool): Whether to apply smoothing to the plots.
        
        Returns:
            tuple: Matplotlib figure and axes containing the plot.
      """
      if x is None:
        x = self.settings['x_train']
        y = self.settings['y_train']
       
      pred = np.array(self.predictor(x))
      pred_cal = self.predict_point(x, smooth = smooth)
      venn_abers = self.predict_venn_abers(x, smooth = smooth)
      intervals = self.predict_interval(x, smooth = smooth)

      # Extract lower and upper bounds
      interval_lower = np.array([min(interval) for interval in intervals])
      interval_upper = np.array([max(interval) for interval in intervals])
      venn_lower = np.array([min(va) for va in venn_abers])
      venn_upper = np.array([max(va) for va in venn_abers])
  
      # Define color map for different plot elements
      good_colors = plt.get_cmap('tab10').colors
      colors = {
        'Original': 'grey',
        'Outcome': 'purple',
        'Calibrated': 'black',
        'Venn-Abers': good_colors[3],
        'Interval': good_colors[0]
      }

      # Sort predictions and apply the same order to all arrays
      sorted_indices = np.argsort(pred)
      s_pred = pred[sorted_indices]
      s_pred_cal = pred_cal[sorted_indices]
      s_venn_lower = venn_lower[sorted_indices]
      s_venn_upper = venn_upper[sorted_indices]
      s_interval_lower = interval_lower[sorted_indices]
      s_interval_upper = interval_upper[sorted_indices]
       
      fig, ax = plt.subplots(figsize=(8, 6))
 
      # Plot Venn Abers intervals
      ax.fill_between(s_pred, s_interval_lower, s_interval_upper, color=colors['Interval'], alpha=0.1)
      ax.plot(s_pred, s_interval_lower, marker='None', linestyle='-', color=colors['Interval'], label='Prediction Interval', alpha=0.3)
      ax.plot(s_pred, s_interval_upper, marker='None', linestyle='-', color=colors['Interval'], alpha=0.3)
      ax.fill_between(s_pred, s_venn_lower, s_venn_upper, color=colors['Venn-Abers'], alpha=0.3, label='Venn-Abers Multi-Prediction')
      ax.plot(s_pred, s_pred_cal, marker='None', linestyle='-', color=colors['Calibrated'], label='Calibrated Prediction')
      ax.plot(s_pred, s_pred, marker='None', linestyle='dashed', color=colors['Original'], label='Original Prediction')
      if y is not None:
        s_outcome = y[sorted_indices]
        sample_indices = np.random.choice(len(s_pred), min(1000, len(s_pred)), replace=False)
        ax.plot(s_pred[sample_indices], s_outcome[sample_indices], marker='o', linestyle='None', color=colors['Outcome'], label='Outcome', markersize=3, alpha=0.05)
       
      # Configure legend
      handles, labels = ax.get_legend_handles_labels()
      by_label = dict(zip(labels, handles))
      ax.legend(by_label.values(), by_label.keys())
      ax.set_title('Calibration Plot for SC-CP', fontsize=25)
      ax.set_xlabel('Original Prediction (uncalibrated)', fontsize=20)
      ax.set_ylabel('Predicted Outcome', fontsize=20)
      ax.grid(False)
      plt.show()
      fig.set_size_inches(6, 6)
      return fig, ax

