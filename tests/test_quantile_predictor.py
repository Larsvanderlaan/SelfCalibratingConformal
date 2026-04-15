import numpy as np

from selfcalibratingconformal import (
    IntervalSolverConfig,
    QuantileAlgoConfig,
    VennAbersQuantileConformalPredictor,
)


class CenterPredictor:
    def predict(self, x):
        x = np.asarray(x)
        return 1.25 * x[:, 0]


class UnderConfidentScoreQuantilePredictor:
    def predict(self, x):
        x = np.asarray(x)
        return 0.3 + 0.15 * np.abs(x[:, 0])


def make_quantile_data(seed=0, n=240):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 1))
    noise = rng.normal(scale=0.35 + 0.15 * np.abs(x[:, 0]), size=n)
    y = 1.25 * x[:, 0] + noise
    return x, y


def test_quantile_predictor_default_absolute_residual_is_symmetric():
    x_cal, y_cal = make_quantile_data(seed=10, n=120)
    x_test, _ = make_quantile_data(seed=11, n=20)
    predictor = VennAbersQuantileConformalPredictor(
        score_quantile_predictor=UnderConfidentScoreQuantilePredictor(),
        center_predictor=CenterPredictor(),
        alpha=0.1,
        algo_params=QuantileAlgoConfig(num_bin_predictor=15, num_bin_score=60),
    )
    predictor.fit(x_cal, y_cal)
    center = CenterPredictor().predict(x_test)
    intervals = predictor.predict_interval(x_test)
    assert np.allclose(intervals[:, 1] - center, center - intervals[:, 0])
    assert predictor.predict_score_quantile(x_test).shape == (20,)
    assert predictor.predict_score_venn_abers(x_test).shape == (20, 2)


def test_quantile_predictor_supports_custom_score_function():
    x_cal, y_cal = make_quantile_data(seed=12, n=120)
    x_test, _ = make_quantile_data(seed=13, n=15)

    def score_fn(x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        return np.abs(y - 1.25 * x[:, 0]) + 0.1 * np.abs(x[:, 0])

    predictor = VennAbersQuantileConformalPredictor(
        score_quantile_predictor=UnderConfidentScoreQuantilePredictor(),
        score_fn=score_fn,
        alpha=0.1,
        algo_params={"num_bin_predictor": 12, "num_bin_score": 50},
        interval_solver_params=IntervalSolverConfig(num_grid=300, margin=0.2),
    )
    predictor.fit(x_cal, y_cal)
    intervals = predictor.predict_interval(x_test)
    assert intervals.shape == (15, 2)
    assert np.all(intervals[:, 1] >= intervals[:, 0])


def test_quantile_coverage_and_threshold_calibration_are_reported():
    x_cal, y_cal = make_quantile_data(seed=14, n=140)
    x_test, y_test = make_quantile_data(seed=15, n=100)
    predictor = VennAbersQuantileConformalPredictor(
        score_quantile_predictor=UnderConfidentScoreQuantilePredictor(),
        center_predictor=CenterPredictor(),
        alpha=0.1,
        algo_params={"num_bin_predictor": 18, "num_bin_score": 80},
    )
    predictor.fit(x_cal, y_cal)
    coverage, width = predictor.check_coverage(x_test, y_test)
    diagnostics = predictor.check_threshold_calibration(x_test, y_test, num_bin=6)
    uncalibrated = UnderConfidentScoreQuantilePredictor().predict(x_test)
    scores = np.abs(y_test - CenterPredictor().predict(x_test))
    uncalibrated_gap = abs(np.mean(scores <= uncalibrated) - (1.0 - predictor.alpha))
    calibrated_gap = abs(diagnostics["marginal_coverage"] - (1.0 - predictor.alpha))
    assert 0.0 <= coverage <= 1.0
    assert width >= 0.0
    assert calibrated_gap <= uncalibrated_gap + 0.05
    assert diagnostics["bins"].shape[0] > 0
