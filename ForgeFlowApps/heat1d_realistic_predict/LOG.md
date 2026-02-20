# Heat1D Realistic Predict Log

## 2026-02-21 - v2 freeze (phase-2 complete)

### Status
- `v2` is closed as a prediction-first benchmark under realistic synthetic observation conditions.
- Decision: keep this app as a stable baseline and move forward with parallel tracks (applied math + domain datasets).

### What was implemented
- Dataset generation:
  - `ForgeFlowApps/heat1d_realistic_predict/config/generate.json`
  - `ForgeFlowApps/heat1d_realistic_predict/scripts/build_dataset.py`
- Prediction pipeline:
  - `ForgeFlowApps/heat1d_realistic_predict/adapters/heat1d_realistic_adapter.py`
  - `ForgeFlowApps/heat1d_realistic_predict/config/run.json`
- Convergence validation:
  - `ForgeFlowApps/heat1d_realistic_predict/scripts/run_convergence_study.py`

### Data and realism protocol
- Dynamics: 1D advection-diffusion with spatially varying `kappa(x)`, time-varying source, and process noise.
- Observation corruption: Gaussian noise, heavy-tail noise, random/block missing, outliers, stuck sensors, drift, quantization, clipping.
- Training/inference target policy: CSV keeps observed target `y` only (no clean-target leakage in train/infer tables).

### Key outputs
- Data:
  - `ForgeFlowApps/heat1d_realistic_predict/data/processed/train.csv`
  - `ForgeFlowApps/heat1d_realistic_predict/data/infer/infer.csv`
- Reports:
  - `ForgeFlowApps/heat1d_realistic_predict/output/manifest.json`
  - `ForgeFlowApps/heat1d_realistic_predict/output/convergence_report.csv`
  - `ForgeFlowApps/heat1d_realistic_predict/output/eval_report.csv`
  - `ForgeFlowApps/heat1d_realistic_predict/output/predictions.csv`

### Final metrics snapshot
- Build stats:
  - `train_rows=8685`
  - `infer_rows=2853`
  - `missing_ratio=0.0987`
  - `outlier_ratio=0.0107`
  - `stuck_ratio=0.0226`
  - `stable_cfl=true`
- Validation (`eval_report.csv`):
  - `val_mae=0.085073`
  - `val_rmse=0.245231`
  - `val_maxae=3.419060`
  - `status=PASS`
- Inference (`predictions.csv`, residual-based):
  - `infer_mae=0.084447`
  - `infer_mse=0.063739`
  - `infer_rmse=0.252466`
  - `infer_maxae=3.413899`
- Temporal convergence (`convergence_report.csv`, pseudo-truth `dt_ref=0.015`):
  - `dt=0.24`: `l2=0.00377305`, `linf=0.00532181`
  - `dt=0.12`: `l2=0.00172829`, `linf=0.00243056`, `p_l2=1.1264`, `p_linf=1.1306`
  - `dt=0.06`: `l2=0.00073375`, `linf=0.00103036`, `p_l2=1.2360`, `p_linf=1.2381`

### Conclusions (EN)
- The pipeline is robust enough for realistic-noise one-step prediction benchmarking.
- Convergence behavior is consistent with first-order temporal expectation.
- This version is suitable as a reusable baseline before adding advanced estimators.

### 结论 (中文)
- `v2` 可以封存：预测链条、收敛验证和噪声场景都已跑通，结果可复现。
- 当前结论支持“在复杂观测噪声下，一阶时间离散 + 线性监督”作为稳定基线。
- 后续不再改动本版核心逻辑，进入并行推进：应用数学深化与领域数据迁移。

### Next (parallel tracks)
- Applied math track: higher-order schemes, uncertainty quantification, inverse/assimilation methods.
- Domain track: geography/astronomy real datasets with the same reproducible pipeline skeleton.

