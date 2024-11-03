# Self-Calibrating Conformal Prediction: Providing Model Calibration and Predictive Inference

This repository provides a Python implementation of **Self-Calibrating Conformal Prediction** (previously termed *Self-Consistent Conformal Prediction*), associated with the [NeurIPS 2024 conference paper](https://arxiv.org/abs/2402.07307). It also contains code to reproduce the experiments from the paper.


## Installation

Run the following command to install the package:

```bash
pip install SelfCalibratingConformal
```

## Abstract

In machine learning, model calibration and predictive inference are essential for producing reliable predictions and quantifying uncertainty to support decision-making. Recognizing the complementary roles of point and interval predictions, we introduce Self-Calibrating Conformal Prediction, a method that combines Venn-Abers calibration and conformal prediction to deliver calibrated point predictions alongside prediction intervals with finite-sample validity conditional on these predictions. To achieve this, we extend the original Venn-Abers procedure from binary classification to regression. Our theoretical framework supports analyzing conformal prediction methods that involve calibrating model predictions and subsequently constructing conditionally valid prediction intervals on the same data, where the conditioning set or conformity scores may depend on the calibrated predictions. Real-data experiments show that our method improves interval efficiency through model calibration and offers a practical alternative to feature-conditional validity.

## Example

Example code demonstrating the use of **Self-Calibrating Conformal Prediction** can be found in the `vignette.ipynb` file. This notebook includes code to perform model calibration, generate prediction intervals, and evaluate model coverage.

```{python}
# See vignette.ipynb for loading data

# Fit an XGBoost model (or any predictive model)
params = {
    'max_depth': 4,
    'learning_rate': 0.05,
    'n_estimators': 100
}
model = xgb.XGBRegressor(**params)
model.fit(X_train, y_train)

# Define a predictor function using the fitted model
def predictor(x):
    return model.predict(np.array(x))

# Set significance level for conformal prediction
alpha = 0.1

# Apply Self-Calibrating Conformal Prediction
conformal_predictor = SelfCalibratingConformalPredictor(predictor, algo_params={'num_bin_predictor': 200, 'num_bin_y': 100, 'binning_method': "quantile"})
conformal_predictor.calibrate(X_cal, y_cal, alpha=alpha)

# Calibrated point predictions derived from Venn-Abers
prediction_calibrated = conformal_predictor.predict_point(X_test)

# Worst and best case bounds for Venn-Abers calibrated multi-prediction
prediction_venn_abers = conformal_predictor.predict_venn_abers(X_test)

# Self-calibrated prediction interval
prediction_interval = conformal_predictor.predict_interval(X_test)

# Evaluate coverage and average interval width on the test set
coverage, width = conformal_predictor.check_coverage(X_test, y_test)
```


## Citation
van der Laan, L., & Alaa, A. M. (2024). Self-Calibrating Conformal Prediction. arXiv preprint arXiv:2402.07307.



@misc{vanderlaan2024selfcalibratingconformalprediction,
      title={Self-Calibrating Conformal Prediction}, 
      author={Lars van der Laan and Ahmed M. Alaa},
      year={2024},
      eprint={2402.07307},
      archivePrefix={arXiv},
      primaryClass={stat.ML},
      url={https://arxiv.org/abs/2402.07307}, 
}
