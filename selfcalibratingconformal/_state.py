from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RegressionFitState:
    prediction_grid: np.ndarray
    calibrated_prediction: np.ndarray
    venn_abers_bounds: np.ndarray
    interval_bounds: np.ndarray
    raw_venn_paths: np.ndarray

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "prediction_uncal": list(self.prediction_grid),
                "prediction_cal": list(self.calibrated_prediction),
                "prediction_venn_abers": list(self.venn_abers_bounds),
                "prediction_interval": list(self.interval_bounds),
            }
        )


@dataclass
class QuantileFitState:
    predictor_grid: np.ndarray
    score_grid: np.ndarray
    calibrated_score_threshold: np.ndarray
    venn_abers_threshold_bounds: np.ndarray
    score_interval_bounds: np.ndarray
    threshold_paths: np.ndarray

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "prediction_uncal": list(self.predictor_grid),
                "prediction_cal": list(self.calibrated_score_threshold),
                "prediction_venn_abers": list(self.venn_abers_threshold_bounds),
                "prediction_interval": list(self.score_interval_bounds),
            }
        )
