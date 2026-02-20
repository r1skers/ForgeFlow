# USGS Water Temp Log

## 2026-02-21 - v1 freeze (phase-1 complete)

### Status
- `v1` is closed as a first-order validation of "applied math + reproducible pipeline" on real data.
- Decision: stop iterative tuning on this app and move to explicit PDE-focused work.

### What was implemented
- Real-data fetch + supervised dataset builder:
  - `ForgeFlowApps/usgs_water_temp/scripts/build_usgs_dataset.py`
- Supervised adapters and configs:
  - `ForgeFlowApps/usgs_water_temp/adapters/usgs_water_temp_adapter.py`
  - `ForgeFlowApps/usgs_water_temp/config/run.json`
  - `ForgeFlowApps/usgs_water_temp/config/run_temp_only.json`
  - `ForgeFlowApps/usgs_water_temp/config/run_full_features.json`
- Comparison / replication scripts:
  - `ForgeFlowApps/usgs_water_temp/scripts/compare_feature_sets.py`
  - `ForgeFlowApps/usgs_water_temp/scripts/run_multi_site_eval.py`

### Data protocol
- Source: USGS OGC API `daily`.
- Variable: `parameter_code=00010` (water temperature, degC), `statistic_id=00003` (daily mean).
- Window: `2023-01-01` to `2025-12-31`.
- Split: train through `2024-12-31`, infer on `2025` segment.
- Sites (3): `USGS-01491000`, `USGS-13192200`, `USGS-02450250`.

### Key outputs
- Single-site outputs:
  - `ForgeFlowApps/usgs_water_temp/output/predictions.csv`
  - `ForgeFlowApps/usgs_water_temp/output/eval_report.csv`
  - `ForgeFlowApps/usgs_water_temp/output/ab_feature_compare.csv`
- Multi-site outputs:
  - `ForgeFlowApps/usgs_water_temp/output/site_pull_v1.csv`
  - `ForgeFlowApps/usgs_water_temp/output/multi_site/site_metrics.csv`
  - `ForgeFlowApps/usgs_water_temp/output/multi_site/predictions_USGS-*.csv`
  - `ForgeFlowApps/usgs_water_temp/output/multi_site/eval_report_USGS-*.csv`

### Final metrics snapshot (infer, 4 errors per site)
- `USGS-01491000`: MAE `0.847357`, MSE `1.140710`, RMSE `1.068040`, MaxAE `2.960343`
- `USGS-13192200`: MAE `0.413245`, MSE `0.332216`, RMSE `0.576381`, MaxAE `2.422584`
- `USGS-02450250`: MAE `0.823036`, MSE `1.230200`, RMSE `1.109144`, MaxAE `5.165395`

### Conclusions (EN)
- One-step linear forecasting is valid as a baseline, but performance is strongly site-dependent.
- Current evidence supports a universal workflow, not universal fixed coefficients.
- `USGS-13192200` is much closer to linear assumptions than the other two sites.

### 结论 (中文)
- 一阶验证到此收口：流程跑通、结果可复现、指标可量化。
- 当前更稳妥的通用结论是“统一方法 + 站点校准”，不是“全站点统一参数”。
- 三站点误差分布差异明显，说明站点条件会显著影响可预测性。
- 下一阶段应转向显式 PDE 主线，不再继续堆这一版线性细节。

