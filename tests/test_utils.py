import numpy as np

from selfcalibratingconformal.utils import linear_extrapolate, make_grid


def test_make_grid_quantile_preserves_added_values():
    grid = make_grid(np.array([1, 2, 3, 4, 5]), num_bin=2, values_to_add=[2.5])
    assert 2.5 in grid
    assert np.all(np.diff(grid) >= 0)


def test_linear_extrapolate_supports_nearest_and_linear():
    x_grid = np.array([0.0, 1.0, 2.0])
    y_grid = np.array([0.0, 2.0, 4.0])
    x_new = np.array([0.25, 1.5])
    nearest = linear_extrapolate(x_grid, y_grid, x_new, kind="nearest")
    linear = linear_extrapolate(x_grid, y_grid, x_new, kind="linear")
    assert np.allclose(nearest, np.array([0.0, 2.0]))
    assert np.allclose(linear, np.array([0.5, 3.0]))
