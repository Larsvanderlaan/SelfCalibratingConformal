import numpy as np
import pytest

from selfcalibratingconformal import (
    RegressionAlgoConfig,
    SelfCalibratingConformalPredictor,
    VennAbersQuantileConformalPredictor,
)
from selfcalibratingconformal.configs import (
    IntervalSolverConfig,
    coerce_interval_solver_config,
    coerce_quantile_algo_config,
    coerce_regression_algo_config,
)


class TinyPredictor:
    def predict(self, x):
        x = np.asarray(x)
        return np.zeros(x.shape[0])


def test_config_coercion_from_dicts():
    regression = coerce_regression_algo_config({"num_bin_predictor": 8, "num_bin_y": 12})
    quantile = coerce_quantile_algo_config({"num_bin_predictor": 9, "num_bin_score": 55})
    interval = coerce_interval_solver_config({"num_grid": 120, "margin": 0.2})
    assert isinstance(regression, RegressionAlgoConfig)
    assert quantile.num_bin_score == 55
    assert isinstance(interval, IntervalSolverConfig)


def test_regression_predict_before_fit_raises():
    model = SelfCalibratingConformalPredictor(TinyPredictor())
    with pytest.raises(RuntimeError):
        model.predict_interval(np.zeros((3, 1)))


def test_quantile_predict_before_fit_raises():
    model = VennAbersQuantileConformalPredictor(
        score_quantile_predictor=TinyPredictor(),
        center_predictor=TinyPredictor(),
    )
    with pytest.raises(RuntimeError):
        model.predict_interval(np.zeros((3, 1)))
