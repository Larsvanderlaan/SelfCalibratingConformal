import numpy as np
import pytest

from selfcalibratingconformal.adapters import adapt_predictor
from selfcalibratingconformal.calibrators import _weighted_quantile, calibrator_CART
from selfcalibratingconformal.configs import (
    HistogramConfig,
    IntervalSolverConfig,
    QuantileAlgoConfig,
    RegressionAlgoConfig,
    coerce_histogram_config,
)
from selfcalibratingconformal.diagnostics import coverage_summary, threshold_calibration_summary
from selfcalibratingconformal.scores import build_absolute_residual_score, regression_conformity_score
from selfcalibratingconformal.utils import (
    ensure_1d_array,
    make_dense_grid,
    match_grid_value,
    safe_median,
    safe_quantile,
)


class PredictMethodOnly:
    def predict(self, x):
        x = np.asarray(x)
        return x[:, 0] + 1.0


def test_adapters_support_callables_and_predict_methods():
    callable_adapter = adapt_predictor(lambda x: np.asarray(x)[:, 0] - 1.0)
    method_adapter = adapt_predictor(PredictMethodOnly())
    x = np.array([[0.0], [1.0]])
    assert np.allclose(callable_adapter(x), np.array([-1.0, 0.0]))
    assert np.allclose(method_adapter(x), np.array([1.0, 2.0]))


def test_adapter_rejects_invalid_predictor():
    with pytest.raises(TypeError):
        adapt_predictor(object())(np.zeros((2, 1)))


def test_config_validation_errors_and_type_errors():
    with pytest.raises(ValueError):
        RegressionAlgoConfig(num_bin_predictor=1).validate()
    with pytest.raises(ValueError):
        RegressionAlgoConfig(num_bin_y=1).validate()
    with pytest.raises(ValueError):
        RegressionAlgoConfig(interpolation_grid_size=5).validate()
    with pytest.raises(ValueError):
        RegressionAlgoConfig(binning_method="bad").validate()
    with pytest.raises(ValueError):
        RegressionAlgoConfig(smooth_kind="spline").validate()
    with pytest.raises(ValueError):
        HistogramConfig(num_bin=1).validate()
    with pytest.raises(ValueError):
        HistogramConfig(binning_method="bad").validate()
    with pytest.raises(ValueError):
        QuantileAlgoConfig(num_bin_predictor=1).validate()
    with pytest.raises(ValueError):
        QuantileAlgoConfig(num_bin_score=9).validate()
    with pytest.raises(ValueError):
        QuantileAlgoConfig(interpolation_grid_size=10).validate()
    with pytest.raises(ValueError):
        QuantileAlgoConfig(binning_method="bad").validate()
    with pytest.raises(ValueError):
        QuantileAlgoConfig(score_range_padding=-1.0).validate()
    with pytest.raises(ValueError):
        IntervalSolverConfig(num_grid=10).validate()
    with pytest.raises(ValueError):
        IntervalSolverConfig(margin=-1.0).validate()
    with pytest.raises(ValueError):
        IntervalSolverConfig(y_min=1.0, y_max=0.0).validate()
    with pytest.raises(TypeError):
        coerce_histogram_config("bad", warning_name="hist")


def test_scores_and_calibrator_helpers_cover_error_paths():
    y_values = np.array([0.0, 1.0])
    original = np.array([0.2, 0.8])
    calibrated = 0.5
    assert np.allclose(
        regression_conformity_score(y_values, calibrated, original, "calibrated"),
        np.array([0.5, 0.5]),
    )
    assert np.allclose(
        regression_conformity_score(y_values, calibrated, original, "debiased"),
        np.abs(y_values - (original - np.mean(original) + calibrated)),
    )
    assert np.allclose(
        regression_conformity_score(y_values, calibrated, original, "original"),
        np.abs(y_values - original),
    )
    with pytest.raises(ValueError):
        regression_conformity_score(y_values, calibrated, original, "bad")
    with pytest.raises(ValueError):
        build_absolute_residual_score(None)
    score_fn = build_absolute_residual_score(PredictMethodOnly())
    assert score_fn(np.array([[1.0], [2.0]]), 0.5).shape == (2,)
    assert _weighted_quantile(np.array([1.0, 2.0, 3.0]), 0.5) == 2.0
    with pytest.raises(ValueError):
        _weighted_quantile(np.array([]), 0.5)
    assert calibrator_CART(np.array([0.0, 1.0]), np.array([0.0, 1.0]))(np.array([0.5])).shape == (1,)


def test_diagnostics_and_utils_helpers_cover_branches():
    assert np.allclose(ensure_1d_array(5.0), np.array([5.0]))
    dense = make_dense_grid(np.array([2.0, 2.0]), grid_size=5)
    assert dense[0] < 2.0 < dense[-1]
    assert np.allclose(match_grid_value(np.array([0.1, 1.9]), np.array([0.0, 1.0, 2.0])), np.array([0.0, 1.0]))
    assert np.allclose(
        match_grid_value(np.array([3.0]), np.array([0.0, 1.0, 2.0]), return_index=True, all_inside=True),
        np.array([2]),
    )
    assert safe_quantile(np.array([1.0, 2.0, 3.0]), 0.5) == 2.0
    assert safe_median([]) == 0.0
    intervals = np.array([[0.0, 1.0], [1.0, 2.0]])
    y_true = np.array([0.5, 3.0])
    assert coverage_summary(intervals, y_true) == [0.5, 1.0]
    summary = threshold_calibration_summary(
        thresholds=np.array([1.0, 1.0, 2.0, 2.0]),
        scores=np.array([0.5, 1.5, 1.5, 2.5]),
        alpha=0.1,
        num_bin=2,
    )
    assert "bins" in summary
    assert summary["bins"].shape[0] >= 1
