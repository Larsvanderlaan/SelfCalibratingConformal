# Self-Calibrating Conformal

`selfcalibratingconformal` provides post-hoc calibration and conformal prediction tools for black-box regression models.

The modernized package ships two primary entrypoints:

- `SelfCalibratingConformalPredictor` for calibrated point predictions and intervals under the original squared-error style workflow.
- `VennAbersQuantileConformalPredictor` for Venn-Abers conformal prediction based on a quantile predictor of conformity scores.

The package is designed to stay backward compatible with the original regression API while making it easier to plug in custom predictors, calibrators, and conformity scores.
