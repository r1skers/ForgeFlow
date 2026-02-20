# Heat Kappa Inverse

Goal: estimate unknown diffusion coefficient `kappa` from observed heat-field decay.

This app is intentionally split into two stages:

- `stage1_data_gen`: generate supervised data from random `kappa` simulations.
- `stage2_inverse`: train/evaluate inverse regressor (`observations -> kappa`).

## Stage 1: Build Dataset

```bash
python ForgeFlowApps/heat_kappa_inverse/stage1_data_gen/scripts/build_dataset.py
```

Config:

- `ForgeFlowApps/heat_kappa_inverse/stage1_data_gen/config/generate.json`

Outputs:

- `ForgeFlowApps/heat_kappa_inverse/data/processed/train.csv`
- `ForgeFlowApps/heat_kappa_inverse/data/processed/infer_id.csv`
- `ForgeFlowApps/heat_kappa_inverse/data/processed/infer_ood.csv`
- `ForgeFlowApps/heat_kappa_inverse/data/processed/infer_id_noise_0p01.csv`
- `ForgeFlowApps/heat_kappa_inverse/data/processed/infer_id_noise_0p03.csv`
- `ForgeFlowApps/heat_kappa_inverse/data/processed/infer_ood_noise_0p01.csv`
- `ForgeFlowApps/heat_kappa_inverse/data/processed/infer_ood_noise_0p03.csv`
- `ForgeFlowApps/heat_kappa_inverse/stage1_data_gen/output/manifest.csv`

## Stage 2: Inverse Regression

Run clean ID/OOD:

```bash
python main.py --config ForgeFlowApps/heat_kappa_inverse/stage2_inverse/config/run_id.json
python main.py --config ForgeFlowApps/heat_kappa_inverse/stage2_inverse/config/run_ood.json
```

Summarize ID/OOD infer metrics:

```bash
python ForgeFlowApps/heat_kappa_inverse/stage2_inverse/scripts/summarize_infer_metrics.py --skip-missing
```

Run noise robustness sweep (1% / 3% on ID and OOD):

```bash
python main.py --config ForgeFlowApps/heat_kappa_inverse/stage2_inverse/config/run_id_noise_0p01.json
python main.py --config ForgeFlowApps/heat_kappa_inverse/stage2_inverse/config/run_id_noise_0p03.json
python main.py --config ForgeFlowApps/heat_kappa_inverse/stage2_inverse/config/run_ood_noise_0p01.json
python main.py --config ForgeFlowApps/heat_kappa_inverse/stage2_inverse/config/run_ood_noise_0p03.json
python ForgeFlowApps/heat_kappa_inverse/stage2_inverse/scripts/summarize_infer_metrics.py --skip-missing
python ForgeFlowApps/heat_kappa_inverse/stage2_inverse/scripts/plot_kappa_scatter.py
python ForgeFlowApps/heat_kappa_inverse/stage2_inverse/scripts/generate_summary_md.py
python ForgeFlowApps/heat_kappa_inverse/stage2_inverse/scripts/sweep_sigma_k.py
```

One-command equivalent (after dataset build):

```bash
mingw32-make run-heat-kappa-noise-sweep
```

Key outputs:

- `ForgeFlowApps/heat_kappa_inverse/output/predictions_id.csv`
- `ForgeFlowApps/heat_kappa_inverse/output/predictions_ood.csv`
- `ForgeFlowApps/heat_kappa_inverse/output/predictions_id_noise_0p01.csv`
- `ForgeFlowApps/heat_kappa_inverse/output/predictions_id_noise_0p03.csv`
- `ForgeFlowApps/heat_kappa_inverse/output/predictions_ood_noise_0p01.csv`
- `ForgeFlowApps/heat_kappa_inverse/output/predictions_ood_noise_0p03.csv`
- `ForgeFlowApps/heat_kappa_inverse/output/eval_report_id.csv`
- `ForgeFlowApps/heat_kappa_inverse/output/eval_report_ood.csv`
- `ForgeFlowApps/heat_kappa_inverse/output/infer_metrics_report.csv`
- `ForgeFlowApps/heat_kappa_inverse/output/kappa_scatter_report.png`
- `ForgeFlowApps/heat_kappa_inverse/output/summary.md`
- `ForgeFlowApps/heat_kappa_inverse/output/sigma_k_sweep_report.csv`
- `ForgeFlowApps/heat_kappa_inverse/output/sigma_k_recommendation.md`

## Feature Definition

- `decay_rate_l2`: `-log(l2_t2 / l2_t1) / (t2 - t1)`
- `mean_abs_t1`, `mean_abs_t2`: compact field-amplitude summaries

Target:

- `kappa`
