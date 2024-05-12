import numpy as np
import pandas as pd
from python import utils
from python import calibrators
from scipy.interpolate import UnivariateSpline
from statsmodels.gam.smooth_basis import BSplines
from statsmodels.gam.generalized_additive_model import GLMGam


class SelfCalibratedConformalPredictor:
    def __init__(self, predictor: callable, calibrator= calibrator_isotonic, 
                 calibrator_params={'max_depth': 15, 'min_child_weight': 20},
                 algo_params = {'num_bin_predictor': 100, 'num_bin_y': 100, 'binning_method': "quantile"}):
        """
        Initializes a SelfCalibratedConformal predictor.

        :param predictor: Callable for making point predictions.
        :param calibrator: Function to calibrate predictions, defaulting to isotonic calibration.
        :param calibrator_params: Dictionary of parameters for the calibrator.
        :param num_bin_predictor: Number of bins for the predictor value grid to approximate the algorithm.
        :param num_bin_y: Number of bins for the y-value grid to approximate the algorithm. Linear interpolation is used to estimate values between grid points.
        :param binning_method: Binning strategy, 'quantile' for equal-frequency bins or 'fixed' for equal-width bins.
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
        self.settings = {
            'x_train': x_train,
            'y_train': y_train,
            'scoring_method': scoring_method,
            'alpha': alpha
        }
        if y_range is None:
            y_range = [min(y_train), max(y_train)]

        y_grid = make_grid(y_train, self.num_bin_y, y_range, binning_method=self.binning_method)
        y_interp = make_grid(y_train, 1000, y_range, binning_method="quantile")
        
        preds_train = np.array(self.predictor(x_train))
        preds_grid = make_grid(preds_train, self.num_bin_predictor, binning_method=self.binning_method)

        preds_grid_indices = list(range(len(preds_grid)))
        preds_grid_calibrated = pd.Series([[] for _ in preds_grid], index=preds_grid_indices)
        predictions_interval = pd.Series([[] for _ in preds_grid], index=preds_grid_indices)

        # Precompute augmented datasets for each grid value of predictor and target
        list_preds_augmented = [np.hstack([preds_train, pred]) for pred in preds_grid]
        list_y_augmented = [np.hstack([y_train, y_val]) for y_val in y_grid]

        for index_f, pred in enumerate(preds_grid):
            preds_augmented = list_preds_augmented[index_f]
            preds_calibrated = np.zeros(len(y_grid))
            thresholds = np.zeros(len(y_grid))
            scores = np.zeros(len(y_grid))

            for index_y, y_val in enumerate(y_grid):
                y_augmented = list_y_augmented[index_y]
                calibrator = self.calibrator(f=preds_augmented, y=y_augmented, **self.calibrator_params)
                preds_augmented_calibrated = calibrator(preds_augmented)
                pred_calibrated = preds_augmented_calibrated[-1]
                
                level_set = [index for index, value in enumerate(preds_augmented_calibrated) if value == pred_calibrated]
                conformity_scores = self.compute_conformity_scores(y_augmented[level_set], pred_calibrated, preds_augmented[level_set], scoring_method)
                
                threshold = np.quantile(conformity_scores, 1 - alpha, method='inverted_cdf')
                score = conformity_scores[-1]
                
                scores[index_y] = score
                thresholds[index_y] = threshold
                preds_calibrated[index_y] = pred_calibrated

            preds_grid_calibrated[index_f] = preds_calibrated
            scores_interp = np.interp(y_interp, y_grid, scores)
            thresholds_interp = np.interp(y_interp, y_grid, thresholds)
            prediction_set = [y for y, score, threshold in zip(y_interp, scores_interp, thresholds_interp) if score <= threshold]
            predictions_interval[index_f] = [min(prediction_set), max(prediction_set)]


        # Build data frame with all results
        baseline_prediction = calibrator_histogram(preds_train, y_train, num_bin = hist_shrinkage_num_bin)(preds_grid)
        y_max, y_min = max(y_train), min(y_train)
        predictions_point = [(max(value) + min(value)) / 2 + (max(value) - min(value)) / (y_max - y_min) * 
                             (baseline_prediction[index] - (max(value) + min(value)) / 2) for index, value in enumerate(preds_grid_calibrated)]
        predictions_venn_abers = [[min(value), max(value)] for value in preds_grid_calibrated]

        fit_info_conformal = pd.DataFrame({
            "prediction_uncal": pd.Series(preds_grid, index=preds_grid_indices),
            "prediction_cal": pd.Series(predictions_point),
            "prediction_venn_abers": pd.Series(predictions_venn_abers),
            "prediction_interval": predictions_interval
        })
        
  
  
        self.fit_info = fit_info_conformal
        
    def compute_conformity_scores(self, y_values, calibrated_prediction, original_predictions, scoring_method):
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
        Outputs a calibrated point prediction for the given input derived from the Venn-Abers multi-prediction.
        """
        f = np.array(self.predictor(x))
        if calibrate:
          return self.extrapolate(self.fit_info['prediction_uncal'], self.fit_info['prediction_cal'], f, smooth = smooth)
        else:
         return f

    def predict_venn_abers(self, x: np.ndarray, smooth = False):
        """
        Outputs the range of the Venn-Abers calibrated multi-prediction for the given input.
        """
        f = np.array(self.predictor(x))
        f_grid = self.fit_info['prediction_uncal']
        bounds = [(row[0], row[1]) for row in self.fit_info['prediction_venn_abers']]
        lower = self.extrapolate(f_grid, [b[0] for b in bounds], f, smooth = smooth)
        upper = self.extrapolate(f_grid, [b[1] for b in bounds], f, smooth = smooth)

        return np.array([lower, upper]).T
      
    def predict_interval(self, x: np.ndarray, smooth = False):
        """
        Outputs a self-calibrated prediction interval for the given input.
        """
        f = np.array(self.predictor(x))
        #index_match = match_grid_value(f, self.fit_info['prediction_uncal'], return_index=True)
        #np.array(self.fit_info.loc[index_match, 'prediction_interval'])
        f_grid = self.fit_info['prediction_uncal']
        bounds = [(row[0], row[1]) for row in self.fit_info['prediction_interval']]
        lower = self.extrapolate(f_grid, [b[0] for b in bounds], f, smooth = smooth)
        upper = self.extrapolate(f_grid, [b[1] for b in bounds], f, smooth = smooth)

        return np.array([lower, upper]).T

    def extrapolate(self, x_grid, y_grid, x_new, smooth=False):
      """
      Performs extrapolation (or smoothing) on a given set of x values based on provided data grids.
    
      Parameters:
          x_grid (array-like): The grid of x-values for which y-values are known.
          y_grid (array-like): The corresponding y-values for the x-values in x_grid.
          x (array-like): The new x-values on which to perform extrapolation or smoothing.
          smooth (bool, optional): If False, performs simple nearest-neighbor extrapolation.
                                  If True, performs smoothing using cubic splines. Default is False.
    
      Returns:
          np.array: The extrapolated or smoothed y-values corresponding to the x-values provided.
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
          bs = BSplines(x_grid, df=[15], degree=[3])
          gam = GLMGam(y_grid,  smoother=bs).fit()
          x_new = x_new[(x_new >= np.min(x_grid)) & (x_new <= np.max(x_grid))]
          preds = gam.predict(bs.transform(x_new))
    
      return np.array(preds)

      
    def plot(self, x = None, y = None, smooth = False):
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
      colors = {
        'Original': 'grey',
        'Outcome': 'purple',
        'Calibrated': 'black',
        'Venn-Abers': 'red',
        'Interval': 'blue'
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
      ax.fill_between(s_pred, s_venn_lower, s_venn_upper, color=colors['Venn-Abers'], alpha=0.3, label='Venn-Abers Multi-prediction')
      ax.plot(s_pred, s_pred_cal, marker='None', linestyle='-', color=colors['Calibrated'], label='Calibrated Prediction')
      ax.plot(s_pred, s_pred, marker='None', linestyle='dashed', color=colors['Original'], label='Original Prediction')
      if y is not None:
        s_outcome = y[sorted_indices]
        ax.plot(s_pred, s_outcome, marker='o', linestyle='None', color=colors['Outcome'], label='Outcome', markersize=3, alpha=0.1)
       
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

