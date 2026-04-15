from __future__ import annotations

import numpy as np

from selfcalibratingconformal.utils import ensure_1d_array, make_grid, match_grid_value

try:
    import xgboost as xgb
except ImportError:  # pragma: no cover - optional dependency
    xgb = None


def _step_transform(grid: np.ndarray, values: np.ndarray):
    grid = ensure_1d_array(grid)
    values = ensure_1d_array(values)

    def transform(x):
        ids = match_grid_value(x, grid, return_index=True, all_inside=True)
        return values[ids]

    return transform


def _weighted_quantile(values: np.ndarray, q: float) -> float:
    values = np.sort(ensure_1d_array(values))
    if values.size == 0:
        raise ValueError("Cannot compute a quantile for an empty block.")
    index = int(np.ceil(q * values.size) - 1)
    index = max(0, min(values.size - 1, index))
    return float(values[index])


def _isotonic_block_fit(x: np.ndarray, y: np.ndarray, reducer):
    order = np.argsort(x, kind="mergesort")
    x_sorted = ensure_1d_array(x)[order]
    y_sorted = ensure_1d_array(y)[order]
    blocks: list[dict[str, np.ndarray | float]] = []
    for x_value, y_value in zip(x_sorted, y_sorted):
        blocks.append(
            {
                "x_left": float(x_value),
                "x_right": float(x_value),
                "values": np.array([float(y_value)]),
                "prediction": float(reducer(np.array([float(y_value)]))),
            }
        )
        while len(blocks) >= 2 and blocks[-2]["prediction"] > blocks[-1]["prediction"]:
            right = blocks.pop()
            left = blocks.pop()
            merged_values = np.concatenate([left["values"], right["values"]])
            blocks.append(
                {
                    "x_left": left["x_left"],
                    "x_right": right["x_right"],
                    "values": merged_values,
                    "prediction": float(reducer(merged_values)),
                }
            )

    intervals = [
        (float(block["x_left"]), float(block["x_right"]), float(block["prediction"]))
        for block in blocks
    ]

    def transform(x_new):
        x_new = ensure_1d_array(x_new)
        result = np.empty_like(x_new, dtype=float)
        for index, value in enumerate(x_new):
            assigned = False
            for left, right, prediction in intervals:
                if left <= value <= right:
                    result[index] = prediction
                    assigned = True
                    break
            if assigned:
                continue
            if value < intervals[0][0]:
                result[index] = intervals[0][2]
            else:
                result[index] = intervals[-1][2]
        return result

    return transform


def calibrator_isotonic(f: np.ndarray, y: np.ndarray, **kwargs):
    del kwargs
    return _isotonic_block_fit(f, y, reducer=np.mean)


def calibrator_quantile_isotonic(
    f: np.ndarray,
    y: np.ndarray,
    quantile_level: float = 0.9,
    **kwargs,
):
    del kwargs
    if not 0 < quantile_level < 1:
        raise ValueError("quantile_level must lie strictly between 0 and 1.")
    return _isotonic_block_fit(
        f,
        y,
        reducer=lambda values: _weighted_quantile(values, quantile_level),
    )


def calibrator_CART(f: np.ndarray, y: np.ndarray, max_depth=10, min_child_weight=50):
    if xgb is not None:  # pragma: no cover - optional dependency branch
        data = xgb.DMatrix(data=ensure_1d_array(f).reshape(-1, 1), label=ensure_1d_array(y))
        cart_fit = xgb.train(
            params={
                "max_depth": max_depth,
                "min_child_weight": min_child_weight,
                "eta": 1,
                "gamma": 0,
                "lambda": 0,
            },
            dtrain=data,
            num_boost_round=1,
        )

        def transform(x):  # pragma: no cover - optional dependency branch
            data_pred = xgb.DMatrix(data=ensure_1d_array(x).reshape(-1, 1))
            return cart_fit.predict(data_pred)

        return transform
    num_bin = max(2, min(20, int(max_depth)))
    return calibrator_histogram(f, y, num_bin=num_bin, binning_method="quantile")


def calibrator_histogram(
    f: np.ndarray,
    y: np.ndarray,
    num_bin=10,
    binning_method="quantile",
    quantile_level: float | None = None,
):
    grid = make_grid(f, num_bin, binning_method=binning_method)
    bin_ids = match_grid_value(f, grid, return_index=True, all_inside=True)
    preds = []
    for bin_id in sorted(set(bin_ids.tolist())):
        values = ensure_1d_array(y)[bin_ids == bin_id]
        if quantile_level is None:
            preds.append(float(np.mean(values)))
        else:
            preds.append(_weighted_quantile(values, quantile_level))
    return _step_transform(grid, np.array(preds, dtype=float))
