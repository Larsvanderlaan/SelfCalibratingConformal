from __future__ import annotations

from typing import Any

import numpy as np

from selfcalibratingconformal.adapters import adapt_predictor
from selfcalibratingconformal.utils import ensure_1d_array


def regression_conformity_score(
    y_values: np.ndarray,
    calibrated_prediction: float,
    original_predictions: np.ndarray,
    scoring_method: str,
) -> np.ndarray:
    y_values = ensure_1d_array(y_values)
    original_predictions = ensure_1d_array(original_predictions)
    if scoring_method == "calibrated":
        return np.abs(y_values - calibrated_prediction)
    if scoring_method == "debiased":
        adjusted = original_predictions - np.mean(original_predictions) + calibrated_prediction
        return np.abs(y_values - adjusted)
    if scoring_method == "original":
        return np.abs(y_values - original_predictions)
    raise ValueError(
        "scoring_method must be 'calibrated', 'debiased', or 'original'."
    )


def build_absolute_residual_score(center_predictor: Any):
    if center_predictor is None:
        raise ValueError(
            "center_predictor is required when score_fn is not provided."
        )
    center_adapter = adapt_predictor(center_predictor)

    def score_fn(x: Any, y: Any) -> np.ndarray:
        center = center_adapter(x)
        outcome = ensure_1d_array(y)
        if outcome.size == 1 and center.size > 1:
            outcome = np.repeat(outcome, center.size)
        return np.abs(outcome - center)

    return score_fn
