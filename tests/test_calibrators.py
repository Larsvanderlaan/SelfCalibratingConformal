import numpy as np

from selfcalibratingconformal.calibrators import (
    calibrator_histogram,
    calibrator_isotonic,
    calibrator_quantile_isotonic,
)


def test_isotonic_calibrator_is_monotone():
    f = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([0.0, 2.0, 1.0, 4.0])
    transform = calibrator_isotonic(f, y)
    preds = transform(np.array([0.0, 1.0, 2.0, 3.0]))
    assert np.all(np.diff(preds) >= -1e-12)


def test_quantile_isotonic_calibrator_is_monotone():
    f = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([0.1, 0.5, 0.4, 0.9])
    transform = calibrator_quantile_isotonic(f, y, quantile_level=0.75)
    preds = transform(np.array([0.0, 1.0, 2.0, 3.0]))
    assert np.all(np.diff(preds) >= -1e-12)


def test_histogram_quantile_calibrator_returns_non_negative_scores():
    f = np.linspace(0, 1, 20)
    y = np.abs(np.sin(f))
    transform = calibrator_histogram(f, y, num_bin=5, quantile_level=0.9)
    preds = transform(np.array([0.1, 0.5, 0.9]))
    assert preds.shape == (3,)
    assert np.all(preds >= 0)
