import numpy as np
import matplotlib

matplotlib.use("Agg")

from selfcalibratingconformal import SelfCalibratingConformalPredictor


class LinearPredictor:
    def predict(self, x):
        x = np.asarray(x)
        return 0.8 * x[:, 0]


def make_regression_data(seed=0, n=120):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 1))
    y = x[:, 0] + rng.normal(scale=0.4, size=n)
    return x, y


def test_backward_compatible_regression_api_shapes():
    x_cal, y_cal = make_regression_data(seed=1, n=80)
    x_test, y_test = make_regression_data(seed=2, n=30)
    predictor = SelfCalibratingConformalPredictor(
        LinearPredictor(),
        algo_params={"num_bin_predictor": 16, "num_bin_y": 18, "binning_method": "quantile"},
    )
    predictor.calibrate(x_cal, y_cal, alpha=0.1)
    assert predictor.fit_info is not None
    assert predictor.predict(x_test).shape == (30,)
    assert predictor.predict_point(x_test).shape == (30,)
    assert predictor.predict_venn_abers(x_test).shape == (30, 2)
    assert predictor.predict_interval(x_test).shape == (30, 2)
    coverage, width = predictor.check_coverage(x_test, y_test)
    assert 0.0 <= coverage <= 1.0
    assert width >= 0.0


def test_regression_custom_conformity_hook_runs():
    x_cal, y_cal = make_regression_data(seed=3, n=60)
    x_test, _ = make_regression_data(seed=4, n=10)

    def score_fn(y_values, calibrated_prediction, original_predictions):
        return np.abs(y_values - calibrated_prediction) + 0.05 * np.abs(original_predictions)

    predictor = SelfCalibratingConformalPredictor(
        LinearPredictor(),
        algo_params={"num_bin_predictor": 12, "num_bin_y": 14},
        conformity_score=score_fn,
    )
    predictor.fit(x_cal, y_cal, alpha=0.1)
    intervals = predictor.predict_interval(x_test)
    assert intervals.shape == (10, 2)
    assert np.all(intervals[:, 1] >= intervals[:, 0])


def test_regression_plot_and_additional_prediction_branches():
    x_cal, y_cal = make_regression_data(seed=5, n=70)
    x_test, y_test = make_regression_data(seed=6, n=12)

    def positional_score(y_values, calibrated_prediction, original_predictions):
        return np.abs(y_values - calibrated_prediction) + 0.01 * np.abs(original_predictions)

    predictor = SelfCalibratingConformalPredictor(
        LinearPredictor(),
        algo_params={"num_bin_predictor": 10, "num_bin_y": 12, "smooth_kind": "nearest"},
        conformity_score=positional_score,
    )
    predictor.fit(x_cal, y_cal, alpha=0.1, y_range=(-3, 3), scoring_method="original")
    raw = predictor.predict(x_test, calibrate=False)
    debiased = predictor.fit(x_cal, y_cal, alpha=0.1, scoring_method="debiased").predict_point(x_test, smooth=True)
    venn = predictor.predict_venn_abers(x_test, smooth=True)
    fig, ax = predictor.plot(x_test, y_test, smooth=True)
    assert raw.shape == debiased.shape == (12,)
    assert venn.shape == (12, 2)
    assert fig is not None
    assert ax is not None
