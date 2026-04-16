# Guides

## Regression workflow

1. Train or define any black-box predictor `f(x)`.
2. Wrap it in `SelfCalibratingConformalPredictor`.
3. Calibrate on an independent calibration set.
4. Use `predict_point`, `predict_venn_abers`, and `predict_interval` on new data.
5. Use `check_coverage` and `plot` to inspect empirical behavior on held-out data.

## Quantile-loss workflow

1. Choose a conformity score `S(x, y)`, typically `|y - mu(x)|`.
2. Train a predictor for the `(1 - alpha)` quantile of that score.
3. Wrap the quantile predictor in `VennAbersQuantileConformalPredictor`.
4. Calibrate using isotonic quantile calibration.
5. Form intervals from the score level set `{ y : S(x, y) <= threshold(x, y) }`.
6. Inspect both marginal coverage and threshold calibration with `check_threshold_calibration`.

## Customization hooks

- Provide any callable or `.predict` model as the base predictor.
- Swap in a custom calibrator with the same signature as the built-ins.
- Provide a custom conformity score on either the regression or quantile workflow.
- Use the advanced notebook when you want custom-score examples instead of the default absolute-residual construction.
