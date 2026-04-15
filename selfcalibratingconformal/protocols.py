from __future__ import annotations

from typing import Any, Callable, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class PredictorProtocol(Protocol):
    def predict(self, x: Any) -> Any:
        ...


CalibratorTransform = Callable[[np.ndarray], np.ndarray]


@runtime_checkable
class CalibratorProtocol(Protocol):
    def __call__(self, f: np.ndarray, y: np.ndarray, **kwargs: Any) -> CalibratorTransform:
        ...


@runtime_checkable
class RegressionConformityScoreProtocol(Protocol):
    def __call__(
        self,
        y_values: np.ndarray,
        calibrated_prediction: float,
        original_predictions: np.ndarray,
    ) -> np.ndarray:
        ...


@runtime_checkable
class QuantileConformityScoreProtocol(Protocol):
    def __call__(self, x: Any, y: Any) -> np.ndarray:
        ...
