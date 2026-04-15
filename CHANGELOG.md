# Changelog

## 1.12.0

- Modernized the package configuration with `pyproject.toml`.
- Preserved the original regression predictor API while refactoring the internals into modular components.
- Added `VennAbersQuantileConformalPredictor` for quantile-loss conformal calibration using Venn-Abers style isotonic calibration.
- Added typed configuration objects, predictor adapters, diagnostics helpers, and user-supplied score/calibrator hooks.
- Added tests, CI workflows, documentation pages, and notebook-based guides.
- Preserved the pre-modernization state on the `legacy/1.x` branch and the `v1.11.0` tag.
