# Self-Calibrating Conformal

`selfcalibratingconformal` is a Python package for post-hoc calibration and conformal prediction with black-box regression models.

It supports two workflows:

- `SelfCalibratingConformalPredictor` for calibrated point predictions, Venn-Abers style prediction sets, and adaptive prediction intervals
- `VennAbersQuantileConformalPredictor` for conformal prediction based on a calibrated predictor of the `(1 - alpha)` quantile of a conformity score

In both workflows, interval width can adapt across data-adaptive bins learned by isotonic regression.

## Installation

```bash
python -m pip install selfcalibratingconformal
```

For development:

```bash
python -m pip install -e ".[dev,docs]"
```

## Quickstart

```python
import numpy as np

from selfcalibratingconformal import SelfCalibratingConformalPredictor


class MeanPredictor:
    def predict(self, x):
        x = np.asarray(x)
        return 1.5 * x[:, 0]


model = SelfCalibratingConformalPredictor(MeanPredictor())
model.fit(X_cal, y_cal, alpha=0.1)

point_predictions = model.predict_point(X_test)
intervals = model.predict_interval(X_test)
coverage, width = model.check_coverage(X_test, y_test)
```

## Workflows

### Regression workflow

Use `SelfCalibratingConformalPredictor` when you already have a point predictor for `y` and want calibrated predictions and intervals.

### Quantile workflow

Use `VennAbersQuantileConformalPredictor` when your workflow predicts the `(1 - alpha)` quantile of a conformity score and you want intervals built from a calibrated threshold.

## Documentation

- [API](./api/)
- [Guides](./guides/)
- [Regression quickstart notebook](https://github.com/Larsvanderlaan/SelfCalibratingConformal/blob/master/quickstart_regression.ipynb)
- [Quantile quickstart notebook](https://github.com/Larsvanderlaan/SelfCalibratingConformal/blob/master/quickstart_quantile_cp.ipynb)
- [Advanced customization notebook](https://github.com/Larsvanderlaan/SelfCalibratingConformal/blob/master/advanced_customization.ipynb)

## Papers

- [Self-Calibrating Conformal Prediction](https://proceedings.neurips.cc/paper_files/paper/2024/file/c1c49aba08e6c90f2b1f85751f497a2f-Paper-Conference.pdf)
- [Generalized Venn and Venn-Abers Calibration with Applications in Conformal Prediction](https://openreview.net/pdf?id=kl2SA1N03E)

## References

- van der Laan, L. and Alaa, A. M. (2024). *Self-Calibrating Conformal Prediction*.
- van der Laan, L. and Alaa, A. M. (2025). *Generalized Venn and Venn-Abers Calibration with Applications in Conformal Prediction*.
- Vovk, V., Petej, I., and Fedorova, V. (2015). *Large-scale probabilistic predictors with and without guarantees of validity*.
- Angelopoulos, A. N. and Bates, S. (2023). *Conformal prediction: A gentle introduction*.
- Romano, Y., Patterson, E., and Candes, E. (2019). *Conformalized quantile regression*.
