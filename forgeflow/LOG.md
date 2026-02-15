# ForgeFlow Log

Last updated: 2026-02-15

## 2026-02-11
- Initialized project log file.
- Step 1 completed: CSV ingestion pipeline runs from `main.py`.
- Step 2 completed: structured parsing and validation.
- Added `read_csv_records` to convert rows into typed records (`int`, `float`, `str`).
- Added CSV quality stats: `total_data_rows`, `valid_rows`, `skipped_bad_rows`.
- Verified by running `python main.py`.

## 2026-02-14
- Step 3A completed: added domain adapter skeleton for domain-specific mapping.
- Added `forgeflow/adapters/base.py` with adapter protocol and required-field validation.
- Added `forgeflow/adapters/tabular.py` as the first concrete adapter.
- Updated `main.py` to run records through adapter states before output.
- Added adapter stats: `total_records`, `valid_states`, `skipped_state_rows`.
- Added focused inline comments on adapter contract, fail-fast schema checks, and state-mapping flow.
- Step 3B completed: converted adapter states into model-ready numeric feature vectors.
- Extended adapter protocol with `feature_names`, `to_feature_vector`, and `build_feature_matrix`.
- Added feature stats: `total_states`, `valid_vectors`, `skipped_feature_rows`, `n_features`.
- Step 4 completed: added next-step supervised dataset builder (`x_t -> x_{t+1}`).
- Step 5 started: added linear dynamics baseline model with `fit/predict/MAE`.
- Updated `main.py` to train and report baseline results on generated supervised samples.
- Added `forgeflow/Input/test/linear_xy.csv` as pure numeric sanity-check data (`y = 2x + 1`).
- Added `forgeflow/adapters/linear_xy.py` for direct feature/target extraction from tabular rows.
- Added `linear_demo.py` to validate baseline linear fitting on the new CSV.
- Simplified entrypoint: `main.py` now runs only the linear sanity-check workflow.
- Removed redundant demo file `linear_demo.py` and old placeholder test file `forgeflow/metrics/test/test.py`.
- Added `forgeflow/metrics/split.py` to perform deterministic train/validation split.
- Updated `main.py` to report both `train_mae` and `val_mae` instead of train-only error.
- Added inference dataset `forgeflow/Input/test/linear_xy_infer.csv` (x-only).
- Extended `LinearXYAdapter` with infer feature builders for unlabeled records.
- Updated `main.py` to run post-training inference and save predictions to `forgeflow/Output/predictions.csv`.
- Added `forgeflow/evaluation/anomaly.py` with 3-sigma residual detector (`ResidualSigmaRule`).
- Updated `main.py` to fit anomaly thresholds from validation residuals and emit anomaly fields in prediction output.
- Note: current inference CSV is x-only, so anomaly fields are not fully verifiable until y labels are provided.
- Added `forgeflow/evaluation/metrics.py` for MAE/RMSE/MaxAE calculation.
- Added `forgeflow/evaluation/policy.py` for task-level eval thresholds and PASS/FAIL gating.
- Updated `main.py` to write evaluation summary to `forgeflow/Output/eval_report.csv`.
- Updated `forgeflow/Input/test/linear_xy_infer.csv` with optional y labels to validate anomaly flagging behavior.
- Added configuration system (`forgeflow/config/runtime.py`) and default config file (`forgeflow/config/linear_xy.json`).
- Updated `main.py` to support `--config`, adapter/model registry lookup, and config-driven paths/thresholds.
- Added noisy scenario files: `forgeflow/Input/test/linear_xy_noisy.csv` and `forgeflow/Input/test/linear_xy_noisy_infer.csv`.
- Added noisy task config: `forgeflow/config/linear_xy_noisy.json`.
- Verified noisy run via `python main.py --config forgeflow/config/linear_xy_noisy.json` (PASS; 2 anomalies flagged).
- Framework refactor completed: introduced `forgeflow/core`, `forgeflow/interfaces`, and `forgeflow/plugins`.
- Added experiment-first structure under `experiments/linear_xy` and `experiments/linear_xy_noisy`.
- Updated `main.py` to call core runner with experiment config (`--config`), defaulting to `experiments/linear_xy/config.json`.
- Moved linear task data/config/output ownership to `experiments/`; removed duplicated copies under `forgeflow/Input`, `forgeflow/config`, and `forgeflow/Output`.
- Added compatibility forwarding in legacy `forgeflow/*` modules so previous imports continue to resolve to the new core implementation.
- Legacy compatibility layer removed after validation; `forgeflow` now keeps only `core/interfaces/plugins` plus `LOG.md`.
- Deleted obsolete files and cache artifacts (`processors.py`, old adapter/metrics/model/config/evaluation wrappers, and `__pycache__` outputs).

## 2026-02-15
- Added split reproducibility controls (`split.shuffle`, `split.seed`) and persisted them in evaluation reports for audit/replay.
- Added chunked inference execution (`infer.chunk_size`) so large inference CSVs can be processed without loading everything into memory at once.
- Improved runtime plugin loading: configs can now use direct class references via `adapter_ref` / `model_ref` (module path), while keeping backward compatibility with registry keys (`adapter` / `model`).
- Updated runner logging to model-capability style output (`summary()` when available, fallback to coefficient shape preview) instead of hardcoded slope/intercept assumptions.
- Added model summaries to built-in linear regressor and app polynomial regressor for consistent pipeline logs.
- Confirmed backward compatibility by running both:
  - `python main.py --config ForgeFlowApps/poly4_cubic/config/run.json`
  - `python main.py --config experiments/linear_xy/config.json`
- Stage assessment: framework is now suitable as a reusable "data -> adapter -> model -> eval -> infer -> report" baseline for new tasks.
- Future direction note: for larger datasets and stronger nonlinear relations, keep current adapter/runner/eval flow and add a PyTorch-based model plugin as an optional model backend.
