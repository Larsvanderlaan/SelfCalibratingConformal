from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from selfcalibratingconformal.utils import make_grid, safe_median


def coverage_summary(
    intervals: np.ndarray,
    y_true: np.ndarray,
    boolean: np.ndarray | None = None,
) -> list[float]:
    selected = slice(None) if boolean is None else np.asarray(boolean).astype(bool)
    filtered_intervals = intervals[selected]
    filtered_y = y_true[selected]
    indicators = (
        (filtered_intervals[:, 0] <= filtered_y)
        & (filtered_y <= filtered_intervals[:, 1])
    )
    widths = filtered_intervals[:, 1] - filtered_intervals[:, 0]
    return [float(np.mean(indicators)), float(safe_median(widths))]


def threshold_calibration_summary(
    thresholds: np.ndarray,
    scores: np.ndarray,
    alpha: float,
    num_bin: int = 10,
) -> dict[str, Any]:
    thresholds = np.asarray(thresholds, dtype=float)
    scores = np.asarray(scores, dtype=float)
    empirical = (scores <= thresholds).astype(float)
    target = 1.0 - alpha
    grid = np.asarray(make_grid(thresholds, num_bin=num_bin, binning_method="quantile"))
    if grid.size < 2:
        grid = np.array([thresholds.min(), thresholds.max()])

    bin_index = np.searchsorted(grid, thresholds, side="right") - 1
    bin_index = np.clip(bin_index, 0, len(grid) - 1)

    rows = []
    for index in sorted(set(bin_index.tolist())):
        mask = bin_index == index
        if not np.any(mask):
            continue
        rows.append(
            {
                "bin_id": int(index),
                "count": int(np.sum(mask)),
                "mean_threshold": float(np.mean(thresholds[mask])),
                "empirical_coverage": float(np.mean(empirical[mask])),
                "target_coverage": float(target),
                "gap": float(np.mean(empirical[mask]) - target),
            }
        )

    frame = pd.DataFrame(rows)
    return {
        "marginal_coverage": float(np.mean(empirical)),
        "target_coverage": float(target),
        "mean_absolute_gap": float(np.mean(np.abs(frame["gap"]))) if not frame.empty else 0.0,
        "max_absolute_gap": float(np.max(np.abs(frame["gap"]))) if not frame.empty else 0.0,
        "bins": frame,
    }
