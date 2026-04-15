from __future__ import annotations

from typing import Iterable

import numpy as np


def ensure_1d_array(values) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        array = array.reshape(1)
    return np.ravel(array)


def make_grid(values, num_bin, values_to_add=None, binning_method="quantile"):
    values = ensure_1d_array(values)
    unique_values = np.unique(values)
    if unique_values.size <= num_bin:
        grid_values = unique_values
    elif binning_method == "quantile":
        qs = np.linspace(0, 1, num=num_bin + 1)
        grid_values = np.unique(np.quantile(values, qs, method="inverted_cdf"))
    elif binning_method == "fixed":
        grid_values = np.linspace(float(np.min(values)), float(np.max(values)), num=num_bin + 1)
    else:
        raise ValueError("binning_method must be 'quantile' or 'fixed'.")

    if values_to_add is not None:
        grid_values = np.unique(
            np.concatenate([ensure_1d_array(grid_values), ensure_1d_array(values_to_add)])
        )
    return ensure_1d_array(grid_values)


def make_dense_grid(values, grid_size: int, range_padding: float = 0.0) -> np.ndarray:
    values = ensure_1d_array(values)
    lower = float(np.min(values))
    upper = float(np.max(values))
    if lower == upper:
        padding = max(1.0, abs(lower) * 0.1)
        lower -= padding
        upper += padding
    else:
        span = upper - lower
        lower -= range_padding * span
        upper += range_padding * span
    return np.linspace(lower, upper, grid_size)


def match_grid_value(values, grid, return_index=False, all_inside=False):
    values = ensure_1d_array(values)
    grid = ensure_1d_array(grid)
    bin_index = np.searchsorted(grid, values, side="right")
    bin_index = np.array([i if i == 0 else i - 1 for i in bin_index], dtype=int)
    if all_inside:
        bin_index = np.clip(bin_index, 0, len(grid) - 1)
    if not return_index:
        return grid[bin_index]
    return bin_index


def linear_extrapolate(x_grid, y_grid, x_new, kind="nearest"):
    x_grid = ensure_1d_array(x_grid)
    y_grid = ensure_1d_array(y_grid)
    x_new = ensure_1d_array(x_new)
    order = np.argsort(x_grid)
    x_grid = x_grid[order]
    y_grid = y_grid[order]
    unique_x, unique_idx = np.unique(x_grid, return_index=True)
    unique_y = y_grid[unique_idx]
    if unique_x.size == 1:
        return np.repeat(unique_y[0], len(x_new))
    if kind == "linear":
        return np.interp(x_new, unique_x, unique_y, left=unique_y[0], right=unique_y[-1])
    nearest_idx = np.abs(unique_x[:, None] - x_new[None, :]).argmin(axis=0)
    return unique_y[nearest_idx]


def safe_quantile(values, q: float) -> float:
    values = ensure_1d_array(values)
    return float(np.quantile(values, q, method="inverted_cdf"))


def safe_median(values: Iterable[float]) -> float:
    array = ensure_1d_array(list(values))
    if array.size == 0:
        return 0.0
    return float(np.median(array))
