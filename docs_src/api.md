# API

## `SelfCalibratingConformalPredictor`

Use this class when you already have a point predictor for `y` and want:

- a calibrated point prediction
- a Venn-Abers style set of calibrated predictions
- a prediction interval induced by those calibrated predictions

Core methods:

- `fit(...)` / `calibrate(...)`
- `predict(...)` / `predict_point(...)`
- `predict_venn_abers(...)`
- `predict_interval(...)`
- `check_coverage(...)`
- `plot(...)`

Customization points:

- `predictor` can be a callable or any object with `.predict(...)`
- `calibrator` can be swapped for a user-supplied calibrator
- `conformity_score` can replace the built-in regression residual score logic

## `VennAbersQuantileConformalPredictor`

Use this class when you have:

- a predictor for the `(1 - alpha)` quantile of a conformity score
- optionally a center predictor `mu(x)` for the default score `|y - mu(x)|`

Core methods:

- `fit(...)` / `calibrate(...)`
- `predict_score_quantile(...)`
- `predict_score_venn_abers(...)`
- `predict_interval(...)`
- `check_coverage(...)`
- `check_threshold_calibration(...)`

Customization points:

- `score_quantile_predictor` can be a callable or any object with `.predict(...)`
- `score_fn(x, y)` can replace the default absolute-residual conformity score
- `calibrator` can be replaced as long as it returns a transform over predicted score quantiles
- `interval_solver_params` controls the numeric `y` grid used for custom score level sets

## Configuration types

- `RegressionAlgoConfig`
- `QuantileAlgoConfig`
- `IntervalSolverConfig`
- `HistogramConfig`

Dict-based configs still work for backward compatibility, but they emit deprecation warnings so users can migrate gradually to typed configs.
