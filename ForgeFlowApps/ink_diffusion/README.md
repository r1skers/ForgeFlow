# Ink Diffusion App

2D periodic ink diffusion using explicit finite differences.

App-local components:

- adapter: `ForgeFlowApps.ink_diffusion.adapters.ink_grid_adapter:InkGridAdapter`
- model: `ForgeFlowApps.ink_diffusion.models.ink_diffusion_explicit:InkDiffusionExplicitSimulator`

Run:

```bash
python main.py --config ForgeFlowApps/ink_diffusion/config/run.json
```

Build supervised samples from trajectory:

```bash
python ForgeFlowApps/ink_diffusion/scripts/build_supervised_samples.py
```

Build surrogate train/infer datasets from trajectory:

```bash
python ForgeFlowApps/ink_diffusion/scripts/build_surrogate_datasets.py
```

Build multi-kappa trajectories (app-local):

```bash
python ForgeFlowApps/ink_diffusion/scripts/build_multi_kappa_trajectories.py
```

Quick smoke run (small output):

```bash
python ForgeFlowApps/ink_diffusion/scripts/build_multi_kappa_trajectories.py --kappas 0.05 0.1 --total-time 0.4
```

Default output:

- `ForgeFlowApps/ink_diffusion/output/multi_kappa/trajectory_kappa_*.csv`
- `ForgeFlowApps/ink_diffusion/output/multi_kappa/manifest.csv`

Build kappa-aware surrogate datasets from the multi-kappa manifest:

```bash
python ForgeFlowApps/ink_diffusion/scripts/build_multi_kappa_surrogate_data.py
```

Quick smoke run (small dataset):

```bash
python ForgeFlowApps/ink_diffusion/scripts/build_multi_kappa_surrogate_data.py --spatial-stride 10
```

Default outputs:

- `ForgeFlowApps/ink_diffusion/data/processed/surrogate_kappa_train.csv`
- `ForgeFlowApps/ink_diffusion/data/processed/surrogate_kappa_infer_id.csv`
- `ForgeFlowApps/ink_diffusion/data/processed/surrogate_kappa_infer_ood.csv`

Run supervised surrogate baseline:

```bash
python main.py --config ForgeFlowApps/ink_diffusion/config/surrogate_run.json
```

Run kappa-aware surrogate baseline (in-domain infer):

```bash
python main.py --config ForgeFlowApps/ink_diffusion/config/surrogate_kappa_id_run.json
```

Run kappa-aware surrogate baseline (OOD infer):

```bash
python main.py --config ForgeFlowApps/ink_diffusion/config/surrogate_kappa_ood_run.json
```

Run rollout prediction evaluation (multi-step):

```bash
python ForgeFlowApps/ink_diffusion/scripts/run_surrogate_rollout_eval.py
```

Kappa-aware rollout example:

```bash
python ForgeFlowApps/ink_diffusion/scripts/run_surrogate_rollout_eval.py --config ForgeFlowApps/ink_diffusion/config/surrogate_kappa_id_run.json --trajectory ForgeFlowApps/ink_diffusion/output/multi_kappa/trajectory_kappa_0p1.csv
```

Default outputs:

- `ForgeFlowApps/ink_diffusion/output/surrogate_rollout_steps.csv`
- `ForgeFlowApps/ink_diffusion/output/surrogate_rollout_summary.csv`

Generate summary plots (simulation + surrogate):

```bash
python ForgeFlowApps/ink_diffusion/scripts/plot_report.py
```

Plot outputs now include:

- `simulation_report.png`
- `surrogate_report.png`
- `rollout_report.png` (if rollout CSV exists)
- `multi_kappa_report.png` (if multi-kappa manifest exists)

Run time-step convergence study (error + observed order):

```bash
python ForgeFlowApps/ink_diffusion/scripts/run_convergence_study.py
```

Default cases are `dt = 0.04, 0.02, 0.01`, all run to the same physical end time.
Report output: `ForgeFlowApps/ink_diffusion/output/convergence_report.csv`.
The report includes:

- reference-based observed order (`observed_order_l2`, `observed_order_linf`)
- triplet/Richardson-style observed order
  (`observed_order_l2_richardson`, `observed_order_linf_richardson`) when
  dt values form `(dt, dt/r, dt/r^2)` triplets.

Run spatial convergence study (grid refinement + observed order):

```bash
python ForgeFlowApps/ink_diffusion/scripts/run_spatial_convergence_study.py
```

Default levels are `stride = 4, 2, 1` over the base `100x100` grid
(coarse `25x25`, mid `50x50`, fine `100x100` reference).
Report output: `ForgeFlowApps/ink_diffusion/output/spatial_convergence_report.csv`.

Run unified verification summary (time + spatial + surrogate):

```bash
python -m forgeflow.core.verification.runner --config ForgeFlowApps/ink_diffusion/config/verification.json
```

Summary output: `ForgeFlowApps/ink_diffusion/output/verification_summary.csv`.

## Quick variable map

If the report looks too dense, read these first:

- `dt`: time step size (smaller -> lower time discretization error, slower runtime)
- `dx`, `dy`: spatial grid spacing (smaller -> higher spatial resolution, higher cost)
- `steps`: number of time updates
- `total_time`: physical horizon (`steps * dt`)
- `kappa`: diffusion coefficient
- `cfl_limit`: stability limit for explicit scheme
- `stable_cfl`: whether current `dt` is stable
- `mass_delta_abs`: absolute mass drift (near zero is expected for periodic/neumann)
- `error_l2_vs_ref`: global/average error to reference
- `error_linf_vs_ref`: worst-point error to reference

## Time convergence: old vs new order fields

- `observed_order_l2`, `observed_order_linf`:
  reference-based order (uses finest `dt` as reference). Good for trend checks.
- `observed_order_l2_richardson`, `observed_order_linf_richardson`:
  triplet/Richardson order from `(dt, dt/r, dt/r^2)`. Use this as the main order estimate.

Practical rule: use Richardson order for acceptance, keep reference-based order as a sanity trend.

## Minimal read path 

1. `status` is `PASS`
2. `stable_cfl=True` and `mass_delta_abs` close to `0`
3. Errors (`L2`, `Linf`) decrease under refinement
4. Richardson order is close to expected theory (time ~1 for current explicit Euler setup)
