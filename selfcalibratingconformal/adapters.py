from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from selfcalibratingconformal.protocols import PredictorProtocol
from selfcalibratingconformal.utils import ensure_1d_array


@dataclass(frozen=True)
class PredictionAdapter:
    predictor: Any

    def __call__(self, x: Any) -> np.ndarray:
        if callable(self.predictor) and not hasattr(self.predictor, "predict"):
            values = self.predictor(x)
        elif isinstance(self.predictor, PredictorProtocol):
            values = self.predictor.predict(x)
        elif hasattr(self.predictor, "predict"):
            values = self.predictor.predict(x)
        else:
            raise TypeError(
                "Predictor must be a callable or expose a .predict method."
            )
        return ensure_1d_array(values)


def adapt_predictor(predictor: Any) -> PredictionAdapter:
    return PredictionAdapter(predictor=predictor)
