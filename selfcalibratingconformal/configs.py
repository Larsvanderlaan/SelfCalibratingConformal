from __future__ import annotations

import warnings
from dataclasses import dataclass, fields
from typing import Any, Mapping, Type, TypeVar


T = TypeVar("T")


def _coerce_dataclass(
    value: Any,
    cls: Type[T],
    *,
    warning_name: str,
    default: T,
) -> T:
    if value is None:
        return default
    if isinstance(value, cls):
        return value
    if isinstance(value, Mapping):
        warnings.warn(
            f"{warning_name} dictionaries remain supported for backward compatibility, "
            f"but passing {cls.__name__} is preferred.",
            DeprecationWarning,
            stacklevel=3,
        )
        allowed = {field.name for field in fields(cls)}
        filtered = {key: val for key, val in value.items() if key in allowed}
        return cls(**filtered)
    raise TypeError(f"{warning_name} must be a {cls.__name__}, mapping, or None.")


@dataclass(frozen=True)
class RegressionAlgoConfig:
    num_bin_predictor: int = 100
    num_bin_y: int = 80
    binning_method: str = "quantile"
    interpolation_grid_size: int = 1000
    smooth_kind: str = "linear"

    def validate(self) -> None:
        if self.num_bin_predictor < 2:
            raise ValueError("num_bin_predictor must be at least 2.")
        if self.num_bin_y < 2:
            raise ValueError("num_bin_y must be at least 2.")
        if self.interpolation_grid_size < 10:
            raise ValueError("interpolation_grid_size must be at least 10.")
        if self.binning_method not in {"quantile", "fixed"}:
            raise ValueError("binning_method must be 'quantile' or 'fixed'.")
        if self.smooth_kind not in {"linear", "nearest"}:
            raise ValueError("smooth_kind must be 'linear' or 'nearest'.")


@dataclass(frozen=True)
class HistogramConfig:
    num_bin: int = 5
    binning_method: str = "quantile"

    def validate(self) -> None:
        if self.num_bin < 2:
            raise ValueError("num_bin must be at least 2.")
        if self.binning_method not in {"quantile", "fixed"}:
            raise ValueError("binning_method must be 'quantile' or 'fixed'.")


@dataclass(frozen=True)
class QuantileAlgoConfig:
    num_bin_predictor: int = 100
    num_bin_score: int = 120
    binning_method: str = "quantile"
    interpolation_grid_size: int = 800
    score_range_padding: float = 0.05

    def validate(self) -> None:
        if self.num_bin_predictor < 2:
            raise ValueError("num_bin_predictor must be at least 2.")
        if self.num_bin_score < 10:
            raise ValueError("num_bin_score must be at least 10.")
        if self.interpolation_grid_size < 50:
            raise ValueError("interpolation_grid_size must be at least 50.")
        if self.binning_method not in {"quantile", "fixed"}:
            raise ValueError("binning_method must be 'quantile' or 'fixed'.")
        if self.score_range_padding < 0:
            raise ValueError("score_range_padding must be non-negative.")


@dataclass(frozen=True)
class IntervalSolverConfig:
    num_grid: int = 600
    y_min: float | None = None
    y_max: float | None = None
    margin: float = 0.1

    def validate(self) -> None:
        if self.num_grid < 50:
            raise ValueError("num_grid must be at least 50.")
        if self.margin < 0:
            raise ValueError("margin must be non-negative.")
        if (
            self.y_min is not None
            and self.y_max is not None
            and self.y_min >= self.y_max
        ):
            raise ValueError("y_min must be smaller than y_max.")


def coerce_regression_algo_config(value: Any) -> RegressionAlgoConfig:
    config = _coerce_dataclass(
        value,
        RegressionAlgoConfig,
        warning_name="algo_params",
        default=RegressionAlgoConfig(),
    )
    config.validate()
    return config


def coerce_histogram_config(value: Any, *, warning_name: str) -> HistogramConfig:
    config = _coerce_dataclass(
        value,
        HistogramConfig,
        warning_name=warning_name,
        default=HistogramConfig(),
    )
    config.validate()
    return config


def coerce_quantile_algo_config(value: Any) -> QuantileAlgoConfig:
    config = _coerce_dataclass(
        value,
        QuantileAlgoConfig,
        warning_name="algo_params",
        default=QuantileAlgoConfig(),
    )
    config.validate()
    return config


def coerce_interval_solver_config(value: Any) -> IntervalSolverConfig:
    config = _coerce_dataclass(
        value,
        IntervalSolverConfig,
        warning_name="interval_solver_params",
        default=IntervalSolverConfig(),
    )
    config.validate()
    return config
