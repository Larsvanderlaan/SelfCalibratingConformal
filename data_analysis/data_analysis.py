
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
import os 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import random



 

path_to_directory = '~/repositories/conformal-venn-abers'
full_path = os.path.expanduser(path_to_directory)
os.chdir(full_path)

from SelfCalibratingConformal.SelfCalibratingConformalPredictor import *
from data_analysis.data_analysis_utils import *
from data_analysis.competitors import *
from data_analysis.condconf import *
random.seed(10)

################################################ 
####### Conformal Prediction ###################
################################################ 

def run_regression_analysis(random_state=10, poor_calibration=False, cross_validate_xgb=False,
                            p_train_nonwhite=None, test_subset_nonwhite=False, p_train=0.5, p_cal=0.3):
    filepath = './data_analysis/meps_21_reg.csv'
    # Change to the specified directory
    full_path = os.path.expanduser(path_to_directory)
    os.chdir(full_path)

    # Import external utilities (make sure this module is available in your path)

    # Data transformation based on calibration setting
    if poor_calibration:
        transform = lambda y: np.log(1 + np.maximum(y, 0))
    else:
        transform = lambda x: x

    X_train, y_train, X_cal, y_cal, X_test, y_test = prepare_data(filepath, 
        random_state=random_state,
        p_train=p_train, p_cal=p_cal,
        p_train_nonwhite=p_train_nonwhite, test_subset_nonwhite=test_subset_nonwhite, 
        log_transform_y=not poor_calibration)
    model = train_xgb_model(X_train, y_train, cross_validate=cross_validate_xgb)

    y_train = transform(y_train)
    y_cal = transform(y_cal)
    y_test = transform(y_test)


    def predictor(x):
        return transform(model.predict(x))

    y_hat = predictor(X_test)

    # Compute intervals using SC-CP
    cp_sc = SelfCalibratingConformalPredictor(predictor, calibrator_params={'max_depth': 15, 'min_child_weight': 20})
    cp_sc.calibrate(X_cal, y_cal, scoring_method="calibrated", hist_shrinkage_num_bin=10)
    y_hat_sc = cp_sc.predict_point(X_test)
    y_hat_venn_abers = cp_sc.predict_venn_abers(X_test)
    intervals_sc = cp_sc.predict_interval(X_test)
    
    # Conditional CP using kernel smoothing
    print("Kernel")
    cp_kernel = kernelCP(predictor, num_bin_predictor= 100)
    cp_kernel.calibrate(X_cal[1:500, :], y_cal[1:500], lambd = 0.005)
    intervals_kernel = cp_kernel.predict_interval(X_test)
    print("End Kernel")
    
    # Compute unconditional CP
    cp_uncond = MondrianCP(predictor, num_bins=1)
    cp_uncond.calibrate(X_cal, y_cal)
    intervals_uncond = cp_uncond.predict_interval(X_test)

    # Compute intervals using Mondrian CP with quantile binning of predictions
    ## 5 bins
    cp_mondrian_5 = MondrianCP(predictor, num_bins=5)
    cp_mondrian_5.calibrate(X_cal, y_cal)
    intervals_mondrian_5 = cp_mondrian_5.predict_interval(X_test)
    ## 10 bins
    cp_mondrian_10 = MondrianCP(predictor, num_bins=10)
    cp_mondrian_10.calibrate(X_cal, y_cal)
    intervals_mondrian_10 = cp_mondrian_10.predict_interval(X_test)
    ## Data-dependent bins being same number of bins in isotonic calibration solution in SC-CP
    cp_mondrian_opt = MondrianCP(predictor, num_bins=len(set(y_hat_sc)))
    cp_mondrian_opt.calibrate(X_cal, y_cal)
    intervals_mondrian_opt = cp_mondrian_opt.predict_interval(X_test)


    plot_comparison = plot_prediction_intervals_baseline_grid(y_hat, y_test, y_hat_sc, 
            dict_of_intervals = {
            "SC-CP": intervals_sc,
            "Marginal": intervals_uncond,
            "Mondrian (10 bins)": intervals_mondrian_10,
            "Kernel": intervals_kernel
          })
     
    

    
    
    results = {
        "Method": [],
        "Coverage_0": [],
        "Coverage_1": [],
        "Average Width_0": [],
        "Average Width_1": [],
        "cal_error_0": [],
        "cal_error_1": []
    }
    names = ["Marginal", "Mondrian (5 bins)", "Mondrian (10 bins)", f"Mondrian ({len(set(y_hat_sc))} bins)", "Kernel", "SC-CP"]
    methods = [intervals_uncond, intervals_mondrian_5, intervals_mondrian_10, intervals_mondrian_opt, intervals_kernel, intervals_sc]

    for method, name in zip(methods, names):
        coverage_data = calculate_coverage_in_bins(X_test[:, -1], method, y_test)
        subgroup_indicator = [x[0] for x in coverage_data]
        coverage_0 = [x[2] for i, x in enumerate(coverage_data) if subgroup_indicator[i] == 0]
        coverage_1 = [x[2] for i, x in enumerate(coverage_data) if subgroup_indicator[i] == 1]
        width_0 = [x[1] for i, x in enumerate(coverage_data) if subgroup_indicator[i] == 0]
        width_1 = [x[1] for i, x in enumerate(coverage_data) if subgroup_indicator[i] == 1]

        if name == "SC-CP":
          cal_error_0 = np.mean((y_hat_sc - y_test)[X_test[:, -1] == 0] > 0)
          cal_error_1 = np.mean((y_hat_sc - y_test)[X_test[:, -1] == 1] > 0)
        else:
          cal_error_0 = np.mean((y_hat - y_test)[X_test[:, -1] == 0] > 0)
          cal_error_1 = np.mean((y_hat - y_test)[X_test[:, -1] == 1] > 0)
        results["Method"].append(name)
        results["Coverage_0"].append(np.mean(coverage_0))
        results["Coverage_1"].append(np.mean(coverage_1))
        results["Average Width_0"].append(np.mean(width_0))
        results["Average Width_1"].append(np.mean(width_1))
        results["cal_error_0"].append(np.mean(cal_error_0))
        results["cal_error_1"].append(np.mean(cal_error_1))

    df = pd.DataFrame(results)
    df = df.map(lambda x: float(f"{x:.3g}") if isinstance(x, (int, float)) else x)
    df = df.map(lambda x: f'{x:g}' if isinstance(x, (int, float)) else x)

    # Output the table in LaTeX format
     
    #print(df.to_latex(index=False))

    
    return df, cp_sc.plot(X_test, y_test), plot_comparison, cp_sc
    
    
df_no_transform, plot_sc_no_transform, plot_comparison_no_transform, cp_sc_no_transform = run_regression_analysis(10, poor_calibration=True)
df_transform, plot_sc_transform, plot_comparison_transform, cp_sc_transform = run_regression_analysis(10, poor_calibration=False)

fig, axs = plot_comparison_no_transform
#fig.set_size_inches(10, 10)
#plt.tight_layout(rect=[0, 0.03, 1, 1.02])
fig.savefig('data_analysis/comparison_poorcal.pdf')
fig, axs = plot_comparison_transform
fig.savefig('data_analysis/comparison_goodcal.pdf')

fig, axs = plot_sc_no_transform
#fig.set_size_inches(6, 6)
fig.savefig('data_analysis/SCCP_poorcal.pdf')
fig, axs = plot_sc_transform
#fig.set_size_inches(6, 6)
fig.savefig('data_analysis/SCCP_goodcal.pdf')

 

# Function to format the number to three significant figures
df_no_transform = df_no_transform.map(lambda x: f'{x:g}' if isinstance(x, (int, float)) else x)
print(df_no_transform.to_latex(index=False))

df_transform = df_transform.map(lambda x: f'{x:g}' if isinstance(x, (int, float)) else x)
print(df_transform.to_latex(index=False))


# 
# 
# #plots = [plot_prediction_intervals(y_hat, y_test, y_hat_sc, intervals_sc, y_hat_venn_abers),
#           #    plot_prediction_intervals(y_hat, y_test, y_hat_sc, intervals_mondrian_10, y_hat_venn_abers),
#            #   plot_prediction_intervals(y_hat, y_test, y_hat_sc, intervals_uncond, y_hat_venn_abers)]
# 
#     coverage_sc = calculate_coverage_in_bins(y_hat_sc, intervals_sc, y_test)
#     coverage_mondrian = calculate_coverage_in_bins(y_hat_sc,  intervals_mondrian_10, y_test)
#     coverage_uncond = calculate_coverage_in_bins(y_hat_sc, intervals_uncond, y_test)
# 
#     predictions_sc, width_sc, coverages_sc = zip(*coverage_sc)
#     predictions_mondrian, width_mondrian, coverages_mondrian = zip(*coverage_mondrian)
#     predictions_uncond, width_uncond, coverages_uncond = zip(*coverage_uncond)
# 
#     # Create the plot
#     plt.figure(figsize=(10, 6))
#     # Plot the coverage levels
#     plt.plot(predictions_uncond, coverages_uncond, label='Unconditional Intervals', marker='o')
#     plt.plot(predictions_mondrian, coverages_mondrian, label='Mondrian Intervals', marker='o')
#     plt.plot(predictions_sc, coverages_sc, label='Self-consistent Intervals', marker='o')
# 
#     plt.axhline(y=0.9, color='red', linestyle='dashed', label='Target Coverage (0.9)')
# 
#     # Set plot details
#     plt.title('Coverage Comparison for Different Interval Types')
#     plt.xlabel('Predictions')
#     plt.ylabel('Average Coverage')
#     plt.legend()
#     plt.grid(True)
# 
#     # Show the plot
#     plt.show()
