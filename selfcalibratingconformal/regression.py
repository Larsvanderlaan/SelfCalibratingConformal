from __future__ import annotations

from dataclasses import asdict
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from selfcalibratingconformal._state import RegressionFitState
from selfcalibratingconformal.adapters import adapt_predictor
from selfcalibratingconformal.calibrators import calibrator_histogram, calibrator_isotonic
from selfcalibratingconformal.configs import (
    HistogramConfig,
    RegressionAlgoConfig,
    coerce_regression_algo_config,
)
from selfcalibratingconformal.diagnostics import coverage_summary
from selfcalibratingconformal.scores import regression_conformity_score
from selfcalibratingconformal.utils import (
    ensure_1d_array,
    linear_extrapolate,
    make_dense_grid,
    make_grid,
)


class SelfCalibratingConformalPredictor:
    def __init__(
        self,
        predictor: Any,
        calibrator=calibrator_isotonic,
        calibrator_params: dict[str, Any] | None = None,
        algo_params: RegressionAlgoConfig | dict[str, Any] | None = None,
        conformity_score=None,
    ):
        self.predictor = predictor
        self._predictor_adapter = adapt_predictor(predictor)
        self.calibrator = calibrator
        self.calibrator_params = calibrator_params or {}
        self.algo_config = coerce_regression_algo_config(algo_params)
        self.conformity_score = conformity_score
        self.fit_state: RegressionFitState | None = None
        self.fit_info = None
        self.settings = {}

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        alpha=0.1,
        y_range=None,
        scoring_method="calibrated",
        hist_shrinkage_num_bin=5,
    ):
        return self.calibrate(
            x_train=x_train,
            y_train=y_train,
            alpha=alpha,
            y_range=y_range,
            scoring_method=scoring_method,
            hist_shrinkage_num_bin=hist_shrinkage_num_bin,
        )

    def calibrate(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        alpha=0.1,
        y_range=None,
        scoring_method="calibrated",
        hist_shrinkage_num_bin=5,
    ):
        x_train = np.asarray(x_train)
        y_train = ensure_1d_array(y_train)
        if y_train.size == 0:
            raise ValueError("Calibration outcomes cannot be empty.")

        self.algo_config.validate()
        histogram_config = HistogramConfig(num_bin=hist_shrinkage_num_bin)
        histogram_config.validate()

        self.settings = {
            "x_train": x_train,
            "y_train": y_train,
            "scoring_method": scoring_method,
            "alpha": alpha,
            "algo_config": asdict(self.algo_config),
        }
        if y_range is None:
            y_grid_base = y_train
        else:
            y_grid_base = np.array([y_range[0], y_range[1]], dtype=float)

        y_grid = make_grid(
            y_train if y_range is None else y_grid_base,
            self.algo_config.num_bin_y,
            values_to_add=y_grid_base if y_range is not None else None,
            binning_method=self.algo_config.binning_method,
        )
        y_interp = make_dense_grid(y_grid, self.algo_config.interpolation_grid_size)
        preds_train = self._predictor_adapter(x_train)
        preds_grid = make_grid(
            preds_train,
            self.algo_config.num_bin_predictor,
            binning_method=self.algo_config.binning_method,
        )

        raw_paths = np.zeros((len(preds_grid), len(y_grid)))
        interval_bounds = np.zeros((len(preds_grid), 2))

        for pred_index, pred in enumerate(preds_grid):
            preds_augmented = np.concatenate([preds_train, np.array([pred])])
            thresholds = np.zeros(len(y_grid))
            test_scores = np.zeros(len(y_grid))
            multipred_venn_abers = np.zeros(len(y_grid))
            for y_index, y_val in enumerate(y_grid):
                y_augmented = np.concatenate([y_train, np.array([y_val])])
                calibrator = self.calibrator(
                    f=preds_augmented,
                    y=y_augmented,
                    **self.calibrator_params,
                )
                preds_augmented_calibrated = ensure_1d_array(calibrator(preds_augmented))
                pred_calibrated = float(preds_augmented_calibrated[-1])

                level_mask = preds_augmented_calibrated == pred_calibrated
                conformity_scores = self._compute_conformity_scores(
                    y_augmented[level_mask],
                    pred_calibrated,
                    preds_augmented[level_mask],
                    scoring_method,
                )
                threshold = float(
                    np.quantile(conformity_scores, 1 - alpha, method="inverted_cdf")
                )
                test_score = float(conformity_scores[-1])

                thresholds[y_index] = threshold
                test_scores[y_index] = test_score
                multipred_venn_abers[y_index] = pred_calibrated

            raw_paths[pred_index] = multipred_venn_abers
            score_interp = np.interp(y_interp, y_grid, test_scores)
            threshold_interp = np.interp(y_interp, y_grid, thresholds)
            accepted = y_interp[score_interp <= threshold_interp]
            if accepted.size == 0:
                accepted = np.array([y_interp[np.argmin(np.abs(score_interp - threshold_interp))]])
            interval_bounds[pred_index] = np.array([float(np.min(accepted)), float(np.max(accepted))])

        baseline_transform = calibrator_histogram(
            preds_train,
            y_train,
            num_bin=histogram_config.num_bin,
            binning_method=histogram_config.binning_method,
        )
        baseline_prediction = ensure_1d_array(baseline_transform(preds_grid))
        venn_bounds = np.column_stack([np.min(raw_paths, axis=1), np.max(raw_paths, axis=1)])
        midpoints = np.mean(venn_bounds, axis=1)
        widths = venn_bounds[:, 1] - venn_bounds[:, 0]
        y_span = max(float(np.max(y_train) - np.min(y_train)), 1e-8)
        calibrated_prediction = midpoints + widths / y_span * (baseline_prediction - midpoints)

        self.fit_state = RegressionFitState(
            prediction_grid=preds_grid,
            calibrated_prediction=calibrated_prediction,
            venn_abers_bounds=venn_bounds,
            interval_bounds=interval_bounds,
            raw_venn_paths=raw_paths,
        )
        self.fit_info = self.fit_state.to_frame()
        return self

    def _check_fitted(self) -> RegressionFitState:
        if self.fit_state is None:
            raise RuntimeError("Call calibrate or fit before prediction.")
        return self.fit_state

    def _compute_conformity_scores(
        self,
        y_values,
        calibrated_prediction,
        original_predictions,
        scoring_method,
    ):
        y_values = ensure_1d_array(y_values)
        original_predictions = ensure_1d_array(original_predictions)
        if self.conformity_score is not None:
            try:
                values = self.conformity_score(
                    y_values=y_values,
                    calibrated_prediction=calibrated_prediction,
                    original_predictions=original_predictions,
                )
            except TypeError:
                values = self.conformity_score(
                    y_values,
                    calibrated_prediction,
                    original_predictions,
                )
            return ensure_1d_array(values)
        return regression_conformity_score(
            y_values=y_values,
            calibrated_prediction=calibrated_prediction,
            original_predictions=original_predictions,
            scoring_method=scoring_method,
        )

    def _extrapolate(self, x_grid, y_grid, x_new, smooth=False):
        config = self.algo_config
        kind = config.smooth_kind if smooth else "nearest"
        return linear_extrapolate(x_grid, y_grid, x_new, kind=kind)

    def predict(self, x: np.ndarray, calibrate=True, smooth=False):
        return self.predict_point(x, calibrate=calibrate, smooth=smooth)

    def predict_point(self, x: np.ndarray, calibrate=True, smooth=False):
        state = self._check_fitted()
        f = self._predictor_adapter(x)
        if calibrate:
            return self._extrapolate(
                state.prediction_grid,
                state.calibrated_prediction,
                f,
                smooth=smooth,
            )
        return f

    def predict_venn_abers(self, x: np.ndarray, smooth=False):
        state = self._check_fitted()
        f = self._predictor_adapter(x)
        lower = self._extrapolate(
            state.prediction_grid,
            state.venn_abers_bounds[:, 0],
            f,
            smooth=smooth,
        )
        upper = self._extrapolate(
            state.prediction_grid,
            state.venn_abers_bounds[:, 1],
            f,
            smooth=smooth,
        )
        return np.column_stack([lower, upper])

    def predict_interval(self, x: np.ndarray, smooth=False):
        state = self._check_fitted()
        f = self._predictor_adapter(x)
        lower = self._extrapolate(
            state.prediction_grid,
            state.interval_bounds[:, 0],
            f,
            smooth=smooth,
        )
        upper = self._extrapolate(
            state.prediction_grid,
            state.interval_bounds[:, 1],
            f,
            smooth=smooth,
        )
        return np.column_stack([lower, upper])

    def check_coverage(self, x_test, y_test, boolean=None, smooth=False):
        intervals = self.predict_interval(x_test, smooth=smooth)
        y_test = ensure_1d_array(y_test)
        mask = None if boolean is None else ensure_1d_array(boolean).astype(bool)
        return coverage_summary(intervals, y_test, mask)

    def plot(self, x=None, y=None, smooth=False):
        state = self._check_fitted()
        if x is None:
            x = self.settings["x_train"]
            y = self.settings["y_train"]

        pred = self._predictor_adapter(x)
        pred_cal = self.predict_point(x, smooth=smooth)
        venn_abers = self.predict_venn_abers(x, smooth=smooth)
        intervals = self.predict_interval(x, smooth=smooth)

        sorted_indices = np.argsort(pred)
        s_pred = pred[sorted_indices]
        s_pred_cal = pred_cal[sorted_indices]
        s_venn_lower = venn_abers[sorted_indices, 0]
        s_venn_upper = venn_abers[sorted_indices, 1]
        s_interval_lower = intervals[sorted_indices, 0]
        s_interval_upper = intervals[sorted_indices, 1]

        good_colors = plt.get_cmap("tab10").colors
        colors = {
            "Original": "grey",
            "Outcome": "#a03d73",
            "Calibrated": "black",
            "Venn-Abers": good_colors[3],
            "Interval": good_colors[0],
        }

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.fill_between(
            s_pred,
            s_interval_lower,
            s_interval_upper,
            color=colors["Interval"],
            alpha=0.12,
        )
        ax.plot(
            s_pred,
            s_interval_lower,
            linestyle="-",
            color=colors["Interval"],
            label="Prediction Interval",
            alpha=0.4,
        )
        ax.plot(s_pred, s_interval_upper, linestyle="-", color=colors["Interval"], alpha=0.4)
        ax.fill_between(
            s_pred,
            s_venn_lower,
            s_venn_upper,
            color=colors["Venn-Abers"],
            alpha=0.3,
            label="Venn-Abers Multi-Prediction",
        )
        ax.plot(
            s_pred,
            s_pred_cal,
            linestyle="-",
            color=colors["Calibrated"],
            label="Calibrated Prediction",
        )
        ax.plot(
            s_pred,
            s_pred,
            linestyle="dashed",
            color=colors["Original"],
            label="Original Prediction",
        )
        if y is not None:
            s_outcome = ensure_1d_array(y)[sorted_indices]
            sample_size = min(1000, len(s_pred))
            sample_indices = np.linspace(0, len(s_pred) - 1, sample_size, dtype=int)
            ax.plot(
                s_pred[sample_indices],
                s_outcome[sample_indices],
                marker="o",
                linestyle="None",
                color=colors["Outcome"],
                label="Outcome",
                markersize=3,
                alpha=0.1,
            )

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        ax.set_title("Calibration Plot for SC-CP", fontsize=18)
        ax.set_xlabel("Original Prediction (uncalibrated)")
        ax.set_ylabel("Predicted Outcome")
        ax.grid(False)
        return fig, ax
