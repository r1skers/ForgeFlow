# USGS Water Temperature

Goal: run a real-data end-to-end supervised pipeline on USGS daily water temperature.

Current milestone: `v1` (see `ForgeFlowApps/usgs_water_temp/LOG.md`).

## 1) Build Dataset from USGS API

```bash
python ForgeFlowApps/usgs_water_temp/scripts/build_usgs_dataset.py --site-id USGS-01491000 --start-date 2023-01-01 --end-date 2025-12-31 --train-until 2024-12-31
```

Optional API key:

```bash
set USGS_API_KEY=your_key_here
python ForgeFlowApps/usgs_water_temp/scripts/build_usgs_dataset.py --site-id USGS-01491000
```

This writes:

- `ForgeFlowApps/usgs_water_temp/data/processed/train.csv`
- `ForgeFlowApps/usgs_water_temp/data/infer/infer.csv`

Optional custom output paths (useful for multi-site pulls):

```bash
python ForgeFlowApps/usgs_water_temp/scripts/build_usgs_dataset.py --site-id USGS-13192200 --include-provisional --train-out data/processed/train_USGS-13192200.csv --infer-out data/infer/infer_USGS-13192200.csv
```

## 2) Train + Validate + Infer

```bash
python main.py --config ForgeFlowApps/usgs_water_temp/config/run.json
```

Outputs:

- `ForgeFlowApps/usgs_water_temp/output/predictions.csv`
- `ForgeFlowApps/usgs_water_temp/output/eval_report.csv`

## 3) Minimal A/B Feature Comparison

Temp-only baseline:

```bash
python main.py --config ForgeFlowApps/usgs_water_temp/config/run_temp_only.json
```

Full-feature model:

```bash
python main.py --config ForgeFlowApps/usgs_water_temp/config/run_full_features.json
```

Generate comparison report:

```bash
python ForgeFlowApps/usgs_water_temp/scripts/compare_feature_sets.py
```

A/B report:

- `ForgeFlowApps/usgs_water_temp/output/ab_feature_compare.csv`

## 4) Multi-site Replication (3 sites)

After generating per-site CSV files, run:

```bash
python ForgeFlowApps/usgs_water_temp/scripts/run_multi_site_eval.py
```

Output:

- `ForgeFlowApps/usgs_water_temp/output/multi_site/site_metrics.csv`

## Feature/Target

- features: `temp_t`, `doy_sin`, `doy_cos`, `dt_days`
- target: `y` (next-step water temperature)
