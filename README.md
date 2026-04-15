# Self-Calibrating Conformal

`selfcalibratingconformal` is a Python package for post-hoc calibration and conformal prediction with black-box models.

It now supports two complementary workflows:

- `SelfCalibratingConformalPredictor` for calibrated point predictions, Venn-Abers style point sets, and prediction intervals based on the original regression method.
- `VennAbersQuantileConformalPredictor` for conformal prediction based on a predictor of the `(1 - alpha)` quantile of a conformity score, calibrated with isotonic Venn-Abers quantile loss.

The modernization keeps the original regression API working while adding typed configs, clearer extension points, tests, docs, and notebook guides.

## Supported Python versions

- Python `3.9` through `3.12`

## Installation

```bash
pip install selfcalibratingconformal
```

For development:

```bash
pip install -e ".[dev,docs]"
```

The package accepts either plain callables or objects exposing `.predict(...)`, which makes it easy to wrap scikit-learn models, custom estimators, or lightweight functions.

## Quickstart: Regression

```python
import numpy as np

from selfcalibratingconformal import SelfCalibratingConformalPredictor


class MeanPredictor:
    def predict(self, x):
        x = np.asarray(x)
        return 1.5 * x[:, 0]


predictor = MeanPredictor()
model = SelfCalibratingConformalPredictor(predictor)
model.fit(X_cal, y_cal, alpha=0.1)

point_predictions = model.predict_point(X_test)
venn_bounds = model.predict_venn_abers(X_test)
intervals = model.predict_interval(X_test)
coverage, width = model.check_coverage(X_test, y_test)
```

## Quickstart: Venn-Abers Quantile CP

```python
import numpy as np

from selfcalibratingconformal import VennAbersQuantileConformalPredictor


class CenterPredictor:
    def predict(self, x):
        x = np.asarray(x)
        return 1.5 * x[:, 0]


class ScoreQuantilePredictor:
    def predict(self, x):
        x = np.asarray(x)
        return np.full(x.shape[0], 0.75)


center = CenterPredictor()
score_quantile = ScoreQuantilePredictor()
cp = VennAbersQuantileConformalPredictor(
    score_quantile_predictor=score_quantile,
    center_predictor=center,
    alpha=0.1,
)
cp.fit(X_cal, y_cal)

score_threshold = cp.predict_score_quantile(X_test)
intervals = cp.predict_interval(X_test)
coverage, width = cp.check_coverage(X_test, y_test)
threshold_diagnostics = cp.check_threshold_calibration(X_test, y_test)
```

## Quantile-loss method

The quantile predictor follows the paper-aligned recipe:

1. Define a conformity score `S(x, y)`, by default `|y - mu(x)|`.
2. Train a model for the `(1 - alpha)` quantile of that score.
3. Calibrate the score-quantile predictor with isotonic Venn-Abers quantile loss.
4. Form the interval from the level set `{ y : S(x, y) <= threshold(x, y) }`.

For the default absolute-residual score, the resulting interval is symmetric around the center model `mu(x)`.

If you provide a custom score function, the package numerically solves the score level set over a configurable `y` grid and returns the outermost enclosing interval in this first release.

## Customization

- Pass a custom calibrator anywhere a built-in calibrator is accepted.
- Supply a custom regression conformity score to `SelfCalibratingConformalPredictor`.
- Supply a custom quantile score function `score_fn(x, y)` to `VennAbersQuantileConformalPredictor`.
- Use typed configs like `RegressionAlgoConfig`, `QuantileAlgoConfig`, and `IntervalSolverConfig` for clearer setup, while dicts remain supported with deprecation warnings.

## Compatibility

- `SelfCalibratingConformalPredictor` remains the main regression entrypoint.
- `fit(...)` aliases `calibrate(...)`, and `predict(...)` aliases calibrated point prediction.
- Legacy import paths are still available for the original regression class and the new quantile class.

## Documentation

- Notebook guides: [quickstart_regression.ipynb](/Users/larsvanderlaan/repos/selfcalibratingconformal/quickstart_regression.ipynb), [quickstart_quantile_cp.ipynb](/Users/larsvanderlaan/repos/selfcalibratingconformal/quickstart_quantile_cp.ipynb), [advanced_customization.ipynb](/Users/larsvanderlaan/repos/selfcalibratingconformal/advanced_customization.ipynb)
- Package docs: GitHub Pages via MkDocs Material
- Release assets: built distributions and docs are attached to GitHub Releases
- The test suite covers public API compatibility, custom hooks, coverage diagnostics, and the new quantile-loss workflow.

## Legacy line

The pre-modernization package state is preserved on the `legacy/1.x` branch and the `v1.11.0` tag.
