# Heat1D Realistic Predict

Goal: prediction-first benchmark on synthetic-but-realistic 1D heat dynamics.

## 1) Build Dataset

```bash
python ForgeFlowApps/heat1d_realistic_predict/scripts/build_dataset.py
```

Config:

- `ForgeFlowApps/heat1d_realistic_predict/config/generate.json`

Generated data:

- `ForgeFlowApps/heat1d_realistic_predict/data/processed/train.csv`
- `ForgeFlowApps/heat1d_realistic_predict/data/infer/infer.csv`
- `ForgeFlowApps/heat1d_realistic_predict/output/trajectory_true.csv`
- `ForgeFlowApps/heat1d_realistic_predict/output/trajectory_observed.csv`
- `ForgeFlowApps/heat1d_realistic_predict/output/manifest.json`

Training/inference CSV uses observed target only (`y`), no clean target column.

## 2) Convergence Check (before training)

```bash
python ForgeFlowApps/heat1d_realistic_predict/scripts/run_convergence_study.py
```

Output:

- `ForgeFlowApps/heat1d_realistic_predict/output/convergence_report.csv`

## 3) Train + Validate + Infer

```bash
python main.py --config ForgeFlowApps/heat1d_realistic_predict/config/run.json
```

Outputs:

- `ForgeFlowApps/heat1d_realistic_predict/output/predictions.csv`
- `ForgeFlowApps/heat1d_realistic_predict/output/eval_report.csv`

## Realism Modules Included

- Spatially varying diffusion coefficient `kappa(x)`
- Advection term and time-varying source forcing
- Process noise during time stepping
- Observation Gaussian noise + heavy-tail noise
- Random missing + block missing segments
- Outlier spikes
- Stuck-sensor segments
- Sensor drift
- Quantization and clipping
