from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from selfcalibratingconformal._state import QuantileFitState
from selfcalibratingconformal.adapters import adapt_predictor
from selfcalibratingconformal.calibrators import (
    calibrator_histogram,
    calibrator_quantile_isotonic,
)
from selfcalibratingconformal.configs import (
    QuantileAlgoConfig,
    coerce_interval_solver_config,
    coerce_quantile_algo_config,
)
from selfcalibratingconformal.diagnostics import coverage_summary, threshold_calibration_summary
from selfcalibratingconformal.scores import build_absolute_residual_score
from selfcalibratingconformal.utils import (
    ensure_1d_array,
    linear_extrapolate,
    make_dense_grid,
    make_grid,
)


class VennAbersQuantileConformalPredictor:
    def __init__(
        self,
        score_quantile_predictor: Any,
        center_predictor: Any | None = None,
        score_fn=None,
        calibrator=calibrator_quantile_isotonic,
        alpha: float = 0.1,
        calibrator_params: dict[str, Any] | None = None,
        algo_params: QuantileAlgoConfig | dict[str, Any] | None = None,
        interval_solver_params=None,
    ):
        self.score_quantile_predictor = score_quantile_predictor
        self._score_quantile_adapter = adapt_predictor(score_quantile_predictor)
        self.center_predictor = center_predictor
        self._center_adapter = None if center_predictor is None else adapt_predictor(center_predictor)
        self._uses_default_absolute_residual_score = score_fn is None
        self.score_fn = score_fn or build_absolute_residual_score(center_predictor)
        self.calibrator = calibrator
        self.alpha = alpha
        self.quantile_level = 1.0 - alpha
        self.calibrator_params = calibrator_params or {}
        self.algo_config = coerce_quantile_algo_config(algo_params)
        self.interval_solver_config = coerce_interval_solver_config(interval_solver_params)
        self.fit_state: QuantileFitState | None = None
        self.fit_info = None
        self.settings = {}

    def fit(self, x_train, y_train):
        return self.calibrate(x_train, y_train)

    def calibrate(self, x_train, y_train):
        x_train = np.asarray(x_train)
        y_train = ensure_1d_array(y_train)
        if y_train.size == 0:
            raise ValueError("Calibration outcomes cannot be empty.")

        self.algo_config.validate()
        self.interval_solver_config.validate()

        score_quantile_train = self._score_quantile_adapter(x_train)
        calibration_scores = ensure_1d_array(self.score_fn(x_train, y_train))
        if np.any(calibration_scores < 0):
            raise ValueError("The quantile conformity score must be non-negative.")

        predictor_grid = make_grid(
            score_quantile_train,
            self.algo_config.num_bin_predictor,
            binning_method=self.algo_config.binning_method,
        )
        score_grid = make_dense_grid(
            calibration_scores,
            self.algo_config.num_bin_score,
            range_padding=self.algo_config.score_range_padding,
        )
        score_grid = np.maximum(score_grid, 0.0)

        threshold_paths = np.zeros((len(predictor_grid), len(score_grid)))
        accepted_score_bounds = np.zeros((len(predictor_grid), 2))

        for pred_index, pred in enumerate(predictor_grid):
            preds_augmented = np.concatenate([score_quantile_train, np.array([pred])])
            thresholds = np.zeros(len(score_grid))
            for score_index, score_candidate in enumerate(score_grid):
                score_augmented = np.concatenate([calibration_scores, np.array([score_candidate])])
                calibrator = self.calibrator(
                    f=preds_augmented,
                    y=score_augmented,
                    quantile_level=self.quantile_level,
                    **self.calibrator_params,
                )
                calibrated = ensure_1d_array(calibrator(preds_augmented))
                thresholds[score_index] = calibrated[-1]

            threshold_paths[pred_index] = thresholds
            accepted = score_grid[score_grid <= thresholds]
            if accepted.size == 0:
                accepted = np.array([0.0])
            accepted_score_bounds[pred_index] = np.array(
                [float(np.min(accepted)), float(np.max(accepted))]
            )

        histogram_transform = calibrator_histogram(
            score_quantile_train,
            calibration_scores,
            num_bin=min(10, max(4, self.algo_config.num_bin_predictor // 10)),
            binning_method=self.algo_config.binning_method,
            quantile_level=self.quantile_level,
        )
        baseline = ensure_1d_array(histogram_transform(predictor_grid))
        venn_bounds = np.column_stack(
            [np.min(threshold_paths, axis=1), np.max(threshold_paths, axis=1)]
        )
        midpoints = np.mean(venn_bounds, axis=1)
        widths = venn_bounds[:, 1] - venn_bounds[:, 0]
        score_span = max(float(np.max(score_grid) - np.min(score_grid)), 1e-8)
        point_threshold = midpoints + widths / score_span * (baseline - midpoints)
        point_threshold = np.maximum(point_threshold, 0.0)

        self.fit_state = QuantileFitState(
            predictor_grid=predictor_grid,
            score_grid=score_grid,
            calibrated_score_threshold=point_threshold,
            venn_abers_threshold_bounds=venn_bounds,
            score_interval_bounds=accepted_score_bounds,
            threshold_paths=threshold_paths,
        )
        self.fit_info = self.fit_state.to_frame()
        self.settings = {
            "x_train": x_train,
            "y_train": y_train,
            "alpha": self.alpha,
            "algo_config": asdict(self.algo_config),
            "interval_solver_config": asdict(self.interval_solver_config),
        }
        return self

    def _check_fitted(self) -> QuantileFitState:
        if self.fit_state is None:
            raise RuntimeError("Call calibrate or fit before prediction.")
        return self.fit_state

    def _predictor_values(self, x) -> np.ndarray:
        return self._score_quantile_adapter(x)

    def predict_score_quantile(self, x):
        state = self._check_fitted()
        pred = self._predictor_values(x)
        return np.maximum(
            linear_extrapolate(
                state.predictor_grid,
                state.calibrated_score_threshold,
                pred,
                kind="linear",
            ),
            0.0,
        )

    def predict_score_venn_abers(self, x):
        state = self._check_fitted()
        pred = self._predictor_values(x)
        lower = np.maximum(
            linear_extrapolate(
                state.predictor_grid,
                state.venn_abers_threshold_bounds[:, 0],
                pred,
                kind="linear",
            ),
            0.0,
        )
        upper = np.maximum(
            linear_extrapolate(
                state.predictor_grid,
                state.venn_abers_threshold_bounds[:, 1],
                pred,
                kind="linear",
            ),
            0.0,
        )
        return np.column_stack([lower, upper])

    def _score_threshold_path(self, pred_value: float):
        state = self._check_fitted()
        path = np.array(
            [
                linear_extrapolate(
                    state.predictor_grid,
                    state.threshold_paths[:, score_index],
                    np.array([pred_value]),
                    kind="linear",
                )[0]
                for score_index in range(len(state.score_grid))
            ],
            dtype=float,
        )
        return np.maximum(path, 0.0)

    def _predict_score_acceptance_radius(self, pred_value: float) -> tuple[float, float]:
        state = self._check_fitted()
        lower = linear_extrapolate(
            state.predictor_grid,
            state.score_interval_bounds[:, 0],
            np.array([pred_value]),
            kind="linear",
        )[0]
        upper = linear_extrapolate(
            state.predictor_grid,
            state.score_interval_bounds[:, 1],
            np.array([pred_value]),
            kind="linear",
        )[0]
        return float(max(0.0, lower)), float(max(0.0, upper))

    def predict_interval(self, x):
        state = self._check_fitted()
        x = np.asarray(x)
        pred = self._predictor_values(x)
        if self._center_adapter is not None and self._uses_default_absolute_residual_score:
            center = self._center_adapter(x)
            radii = np.array([self._predict_score_acceptance_radius(value)[1] for value in pred])
            return np.column_stack([center - radii, center + radii])

        intervals = np.zeros((len(pred), 2))
        y_train = self.settings["y_train"]
        y_min = self.interval_solver_config.y_min
        y_max = self.interval_solver_config.y_max
        if y_min is None or y_max is None:
            train_min = float(np.min(y_train))
            train_max = float(np.max(y_train))
            span = max(train_max - train_min, 1e-8)
            margin = self.interval_solver_config.margin * span
            y_min = train_min - margin if y_min is None else y_min
            y_max = train_max + margin if y_max is None else y_max

        y_grid = np.linspace(y_min, y_max, self.interval_solver_config.num_grid)
        for index, pred_value in enumerate(pred):
            threshold_path = self._score_threshold_path(pred_value)
            score_values = ensure_1d_array(
                self.score_fn(x[index : index + 1], y_grid)
            )
            threshold_values = np.interp(
                score_values,
                state.score_grid,
                threshold_path,
                left=threshold_path[0],
                right=threshold_path[-1],
            )
            accepted = y_grid[score_values <= threshold_values]
            if accepted.size == 0:
                nearest = np.argmin(np.abs(score_values - threshold_values))
                accepted = np.array([y_grid[nearest]])
            intervals[index] = np.array([float(np.min(accepted)), float(np.max(accepted))])
        return intervals

    def check_coverage(self, x_test, y_test, boolean=None):
        intervals = self.predict_interval(x_test)
        y_test = ensure_1d_array(y_test)
        mask = None if boolean is None else ensure_1d_array(boolean).astype(bool)
        return coverage_summary(intervals, y_test, mask)

    def check_threshold_calibration(self, x_test, y_test, num_bin: int = 10):
        thresholds = self.predict_score_quantile(x_test)
        scores = ensure_1d_array(self.score_fn(x_test, y_test))
        return threshold_calibration_summary(
            thresholds=thresholds,
            scores=scores,
            alpha=self.alpha,
            num_bin=num_bin,
        )
