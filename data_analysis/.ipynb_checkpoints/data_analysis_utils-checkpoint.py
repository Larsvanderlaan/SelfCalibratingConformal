
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
import os 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from matplotlib.colors import CSS4_COLORS
import random
 
 

################################################ 
####### Model Training #########################
################################################ 

 

def prepare_data(filepath, random_state=3, p_train=0.5, p_cal=0.3, response_column="UTILIZATION_reg", col_names = "default", p_train_nonwhite = None, test_subset_nonwhite = False, log_transform_y = False):
    """
    Load data, preprocess features, and split into train, calibration, and test sets.

    Parameters:
    - filepath (str): Path to the CSV file containing the data.
    - random_state (int): Seed used by the random number generator for reproducibility.
    - p_train (float): Proportion of data to be used for training.
    - p_cal (float): Proportion of data to be used for calibration.
    - p_test (float): Proportion of data to be used for testing.
    - response_column (str): Name of the column to be used as the response variable.

    Returns:
    - X_train (np.ndarray): Features for the training set.
    - y_train (np.ndarray): Response for the training set.
    - X_cal (np.ndarray): Features for the calibration set.
    - y_cal (np.ndarray): Response for the calibration set.
    - X_test (np.ndarray): Features for the test set.
    - y_test (np.ndarray): Response for the test set.
    """

    p_test= 1 - p_train -p_cal
    # Load data
    df = pd.read_csv(filepath)

    # Filter out unnecessary columns and prepare response variable
    if col_names == "default":
      col_names = ['AGE', 'PCS42', 'MCS42', 'K6SUM42', 'PERWT16F', 'REGION=1',
                   'REGION=2', 'REGION=3', 'REGION=4', 'SEX=1', 'SEX=2', 'MARRY=1',
                   'MARRY=2', 'MARRY=3', 'MARRY=4', 'MARRY=5', 'MARRY=6', 'MARRY=7',
                   'MARRY=8', 'MARRY=9', 'MARRY=10', 'FTSTU=-1', 'FTSTU=1', 'FTSTU=2',
                   'FTSTU=3', 'ACTDTY=1', 'ACTDTY=2', 'ACTDTY=3', 'ACTDTY=4',
                   'HONRDC=1', 'HONRDC=2', 'HONRDC=3', 'HONRDC=4', 'RTHLTH=-1',
                   'RTHLTH=1', 'RTHLTH=2', 'RTHLTH=3', 'RTHLTH=4', 'RTHLTH=5',
                   'MNHLTH=-1', 'MNHLTH=1', 'MNHLTH=2', 'MNHLTH=3', 'MNHLTH=4',
                   'MNHLTH=5', 'HIBPDX=-1', 'HIBPDX=1', 'HIBPDX=2', 'CHDDX=-1',
                   'CHDDX=1', 'CHDDX=2', 'ANGIDX=-1', 'ANGIDX=1', 'ANGIDX=2',
                   'MIDX=-1', 'MIDX=1', 'MIDX=2', 'OHRTDX=-1', 'OHRTDX=1', 'OHRTDX=2',
                   'STRKDX=-1', 'STRKDX=1', 'STRKDX=2', 'EMPHDX=-1', 'EMPHDX=1',
                   'EMPHDX=2', 'CHBRON=-1', 'CHBRON=1', 'CHBRON=2', 'CHOLDX=-1',
                   'CHOLDX=1', 'CHOLDX=2', 'CANCERDX=-1', 'CANCERDX=1', 'CANCERDX=2',
                   'DIABDX=-1', 'DIABDX=1', 'DIABDX=2', 'JTPAIN=-1', 'JTPAIN=1',
                   'JTPAIN=2', 'ARTHDX=-1', 'ARTHDX=1', 'ARTHDX=2', 'ARTHTYPE=-1',
                   'ARTHTYPE=1', 'ARTHTYPE=2', 'ARTHTYPE=3', 'ASTHDX=1', 'ASTHDX=2',
                   'ADHDADDX=-1', 'ADHDADDX=1', 'ADHDADDX=2', 'PREGNT=-1', 'PREGNT=1',
                   'PREGNT=2', 'WLKLIM=-1', 'WLKLIM=1', 'WLKLIM=2', 'ACTLIM=-1',
                   'ACTLIM=1', 'ACTLIM=2', 'SOCLIM=-1', 'SOCLIM=1', 'SOCLIM=2',
                   'COGLIM=-1', 'COGLIM=1', 'COGLIM=2', 'DFHEAR42=-1', 'DFHEAR42=1',
                   'DFHEAR42=2', 'DFSEE42=-1', 'DFSEE42=1', 'DFSEE42=2',
                   'ADSMOK42=-1', 'ADSMOK42=1', 'ADSMOK42=2', 'PHQ242=-1', 'PHQ242=0',
                   'PHQ242=1', 'PHQ242=2', 'PHQ242=3', 'PHQ242=4', 'PHQ242=5',
                   'PHQ242=6', 'EMPST=-1', 'EMPST=1', 'EMPST=2', 'EMPST=3', 'EMPST=4',
                   'POVCAT=1', 'POVCAT=2', 'POVCAT=3', 'POVCAT=4', 'POVCAT=5',
                   'INSCOV=1', 'INSCOV=2', 'INSCOV=3', 'RACE']

    y = df[response_column].values
    X = df[col_names].values
    if log_transform_y:
      y = np.log(1 + y)

    # Split data into train, calibration, and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - p_train, random_state=random_state)
    X_cal, X_test, y_cal, y_test = train_test_split(X_test, y_test, test_size=p_test / (p_cal + p_test), random_state=random_state)

    if p_train_nonwhite != None:
      indices_one = np.where(X_train[:, -1] == 1)[0]
      indices_zero = np.where(X_train[:, -1] == 0)[0]
      p_cur = np.mean(X_train[:,-1]==0)
      sample_size = int(p_train_nonwhite*p_cur * len(indices_zero)) 
 
      random_indices_zero = np.random.choice(indices_zero, sample_size, replace=False)
   
      combined_indices = np.concatenate((indices_one, random_indices_zero))
      X_train = X_train[combined_indices,:] 
      y_train = y_train[combined_indices] 
 
      print(np.mean(X_train[:,-1]==0))

    if test_subset_nonwhite:
      subset_race = X_test[:,-1] == 0
      X_test = X_test[subset_race,:] 
      y_test = y_test[subset_race] 

    return X_train, y_train, X_cal, y_cal, X_test, y_test

 
 

def train_xgb_model(X_train, y_train, cross_validate=True):
    """
    Trains an XGBoost regression model with or without cross-validation based on a parameter.

    Parameters:
    - X_train (np.ndarray): Training data features.
    - y_train (np.ndarray): Training data labels.
    - cross_validate (bool): Flag to determine whether to perform cross-validation. Defaults to True.

    Returns:
    - model (xgb.XGBRegressor): The trained XGBoost model.
    - best_parameters (dict): Best parameters found during cross-validation, if applicable.
    """

    if cross_validate:
        # Define the model and set up the parameter grid for cross-validation
        xgb_model = xgb.XGBRegressor()
        param_grid = {
            'max_depth': [4,5,6,7],
            'learning_rate': [0.005, 0.01, 0.02],
            'n_estimators': [50, 100, 200]
        }

        # Configure and fit GridSearchCV
        cv = GridSearchCV(xgb_model, param_grid, scoring='neg_mean_squared_error', cv=5)
        cv.fit(X_train, y_train)

        # Retrieve the best parameters and the best model
        best_parameters = cv.best_params_
        model = cv.best_estimator_
        print("Best parameters:", best_parameters)
    else:
        # Default parameters selected in an earlier run using cross-validation
        param_optimal = {
            'max_depth': 9,
            'learning_rate': 0.05,
            'n_estimators': 200
        }
        model = xgb.XGBRegressor(**param_optimal)
        model.fit(X_train, y_train)

    if cross_validate:
        return model 
    else:
        return model





################################################ 
####### Utility functions ###################
################################################ 



def compute_coverage(intervals, y, boolean=None):
    # Using a list comprehension to check if each y value is within the corresponding interval
    # It checks boolean[index] only if boolean is not None
    indicators = [lower <= y[index] <= upper for index, (lower, upper) in enumerate(intervals)
                  if boolean is None or boolean[index] == 1]

    # Compute the coverage as the mean of indicators
    coverage = np.mean(indicators)
    width = np.median([np.diff(row) for row in intervals if boolean is None or boolean[index] == 1])
    return [coverage, width]
  

def calculate_coverage_in_bins(bin_ids, intervals, outcome):
    # Unique sorted predictions from bin_id
 
    unique_bins = sorted(set(bin_ids))

    # List to store coverage levels for each unique prediction
    coverage_levels = []

    # Iterate over each unique prediction value
    for bin_id in unique_bins:
        # Filter data for the current prediction
        indices = [i for i, value in enumerate(bin_ids) if value == bin_id]
        sub_intervals = [intervals[i] for i in indices]
        sub_outcome = [outcome[i] for i in indices]
         
        #print(sub_intervals)
 
        # Calculate coverage: outcome value is within the interval
        coverage = [interval[0] <= sub_outcome[i] and interval[1] >= sub_outcome[i] for i, interval in enumerate(sub_intervals)]
        width = [interval[1] - interval[0]for i, interval in enumerate(sub_intervals)]
        # Calculate and store the average coverage for each unique prediction
        coverage_levels.append((bin_id, np.mean(width), np.mean(coverage)))

    return coverage_levels

 
  
def plot_prediction_intervals(pred, outcome, pred_cal, intervals, venn_abers):
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
    s_outcome = outcome[sorted_indices]
    s_pred_cal = pred_cal[sorted_indices]
    s_venn_lower = venn_lower[sorted_indices]
    s_venn_upper = venn_upper[sorted_indices]
    s_interval_lower = interval_lower[sorted_indices]
    s_interval_upper = interval_upper[sorted_indices]


    plt.figure(figsize=(8, 6))
    fig, ax = plt.subplots()
    
 
    # Plot Venn Abers intervals
    ax.fill_between(s_pred, s_venn_lower, s_venn_upper, color=colors['Venn-Abers'], alpha=0.3, label='Venn-Abers Multi-prediction')
    ax.fill_between(s_pred, s_interval_lower, s_interval_upper, color=colors['Interval'], alpha=0.1)
    ax.plot(s_pred, s_interval_lower, marker='None', linestyle='dashed', color=colors['Interval'], label='Prediction Interval')
    ax.plot(s_pred, s_interval_upper, marker='None', linestyle='dashed', color=colors['Interval'])
    ax.plot(s_pred, s_pred, marker='None', linestyle='dashed', color=colors['Original'], label='Original Prediction')
    sample_indices = np.random.choice(len(s_pred), min(1000, len(s_pred)), replace=False)
    ax.plot(s_pred[s_outcome[sample_indices]], s_outcome, marker='o', linestyle='None', color=colors['Outcome'], label='Outcome', markersize=3, alpha=0.01)
    ax.plot(s_pred, s_pred_cal, marker='None', linestyle='-', color=colors['Calibrated'], label='Calibrated Prediction')

    # Configure legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    ax.title('Calibrated Point and Interval Predictions against Model Predictions')
    ax.xlabel('Original Prediction (uncalibrated)')
    ax.ylabel('Outcome')
    ax.grid(True)
    print(fig)
    return fig, ax

 
 

def plot_prediction_intervals_baseline(pred, outcome, pred_cal, dict_of_intervals):
    """
    Plot calibrated predictions and prediction intervals against original predictions.
    
    Parameters:
    - pred: Array of original predictions.
    - outcome: Array of actual outcomes.
    - pred_cal: Array of calibrated predictions.
    - dict_of_intervals: Dictionary of label: prediction intervals pairs.
    """
    # Define color map for different plot elements
    colors = {
        'Original': 'grey',
        'Outcome': 'purple',
        'Calibrated': 'black'
    }

    # Sort indices for plotting
    sorted_indices = np.argsort(pred)
    s_pred = pred[sorted_indices]
    s_outcome = outcome[sorted_indices]
    s_pred_cal = pred_cal[sorted_indices]

    # Setup the plot
    plt.figure(figsize=(8, 6))
    fig, ax = plt.subplots()
    ax.plot(s_pred, s_pred, color=colors['Original'], marker='None', linestyle='dashed', label='Original Prediction')
    sample_indices = np.random.choice(len(s_pred), min(1000, len(s_pred)), replace=False)
    ax.scatter(s_pred[sample_indices], s_outcome[sample_indices], color=colors['Outcome'], label='Outcome', s=10, alpha=0.01)
    ax.plot(s_pred, s_pred_cal, color=colors['Calibrated'], label='Calibrated Prediction')

    # Plot prediction intervals using a predefined set of good plotting colors
    good_colors = plt.get_cmap('tab10').colors
    for i, (label, intervals) in enumerate(dict_of_intervals.items()):
        color_index = i % len(good_colors)  # Cycle through colors if there are more labels than colors
        interval_color = good_colors[color_index]

        interval_lower = np.array([min(interval) for interval in intervals])
        interval_upper = np.array([max(interval) for interval in intervals])
         
        s_interval_lower = interval_lower[sorted_indices]
        s_interval_upper = interval_upper[sorted_indices]

        ax.plot(s_pred, s_interval_lower, color=interval_color, linestyle='-', label=label, linewidth=2)
        ax.plot(s_pred, s_interval_upper, color=interval_color, linestyle='-', linewidth=2)
        if label == "SC-CP":
          ax.fill_between(s_pred, s_interval_lower, s_interval_upper, color=interval_color, alpha=0.1)

    # Legend handling to avoid duplicate labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    # Final plot adjustments
    ax.title('Calibrated Point and Interval Predictions against Model Predictions')
    ax.xlabel('Original Prediction (uncalibrated)')
    ax.ylabel('Outcome')
    ax.grid(True)
    print(fig)

    return fig, ax
  


def plot_prediction_intervals_baseline_grid(pred, outcome, pred_cal, dict_of_intervals):
    """
    Plot calibrated predictions and prediction intervals against original predictions in separate subplots.

    Parameters:
    - pred: Array of original predictions.
    - outcome: Array of actual outcomes.
    - pred_cal: Array of calibrated predictions.
    - dict_of_intervals: Dictionary of label: prediction intervals pairs.
    """
    # Define color map for different plot elements
    colors = {
        'Original': 'grey',
        'Outcome': 'purple',
        'Calibrated': 'black'
    }

    # Sort indices for plotting
    sorted_indices = np.argsort(pred)
    s_pred = pred[sorted_indices]
    s_outcome = outcome[sorted_indices]
    s_pred_cal = pred_cal[sorted_indices]

    # Number of intervals
    n_intervals = len(dict_of_intervals)

    # Create subplots with a shared x-axis
    fig, axes = plt.subplots((n_intervals + 2 - 1) // 2, 2, figsize=(8, 2 * n_intervals), sharex=True, sharey=True)
    axes = axes.flatten()
    if n_intervals == 1:
        axes = [axes]  # Make it iterable if there's only one subplot

    global_min = min(s_outcome.min(), s_pred_cal.min())
    global_max = max(s_outcome.max(), s_pred_cal.max())
    # Plot prediction intervals
    good_colors = plt.get_cmap('tab10').colors
    for i, (label, intervals) in enumerate(dict_of_intervals.items()):
        ax = axes[i]
        color_index = i % len(good_colors)
        interval_color = good_colors[color_index]

        interval_lower = np.array([min(interval) for interval in intervals])
        interval_upper = np.array([max(interval) for interval in intervals])
        s_interval_lower = interval_lower[sorted_indices]
        s_interval_upper = interval_upper[sorted_indices]

        # Plotting on each subplot
        ax.plot(s_pred, s_pred, color=colors['Original'], marker='None', linestyle='dashed', label='Original Prediction')
        ax.plot(s_pred, s_pred_cal, color=colors['Calibrated'], label='Calibrated Prediction')
        sample_indices = np.random.choice(len(s_pred), min(1000, len(s_pred)), replace=False)
        ax.scatter(s_pred[sample_indices], s_outcome[sample_indices], color=colors['Outcome'], label='Outcome', s=10, alpha=0.1)
        #ax.scatter(s_pred, s_outcome, color=colors['Outcome'], label='Outcome', s=10, alpha=0.1)
        ax.plot(s_pred, s_interval_lower, color=interval_color, linestyle='-', label='', linewidth=2)
        ax.plot(s_pred, s_interval_upper, color=interval_color, linestyle='-', label='', linewidth=2)
        ax.fill_between(s_pred, s_interval_lower, s_interval_upper, color=interval_color, alpha=0.1)
        #ax.legend()

        ax.set_ylim(global_min, global_max)
        # Labels and titles
        ax.set_ylabel('Predicted Outcome', fontsize=20)
        ax.set_title(f'{label}', fontsize=25)
        ax.grid(True)

    # Common X label

    fig.text(0.5, 0.02, 'Original Prediction (uncalibrated)', ha='center', fontsize=20)
    fig.suptitle('Prediction bands', fontsize=25)
    #plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.set_size_inches(10, 10)
    #fig.suptitle('') 
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    
    return fig, axes

