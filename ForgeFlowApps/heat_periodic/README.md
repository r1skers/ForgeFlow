# Heat Periodic Benchmark

Step 1 (current): generate analytic initial condition CSV for the periodic 2D heat benchmark.

PDE assumption (this instance):

- `u_t = kappa * (u_xx + u_yy)`
- boundary: `periodic`

App-local adapter:

- `ForgeFlowApps.heat_periodic.adapters.heat_grid_adapter:HeatGridAdapter`

Initial condition:

- `u(x, y, 0) = sin(2*pi*x) * sin(2*pi*y)`

Generate:

```bash
python ForgeFlowApps/heat_periodic/scripts/build_initial_csv.py
```

Output:

- `ForgeFlowApps/heat_periodic/data/processed/initial.csv`

Run simulation:

```bash
python main.py --config ForgeFlowApps/heat_periodic/config/run.json
```

Run a longer-time simulation (clearer decay in snapshots):

```bash
python main.py --config ForgeFlowApps/heat_periodic/config/run_long_t.json
```

Run exact-solution time convergence:

```bash
python ForgeFlowApps/heat_periodic/scripts/run_exact_convergence_study.py
```

Output:

- `ForgeFlowApps/heat_periodic/output/exact_convergence_report.csv`

Run exact-solution spatial convergence:

```bash
python ForgeFlowApps/heat_periodic/scripts/run_exact_spatial_convergence_study.py
```

Output:

- `ForgeFlowApps/heat_periodic/output/exact_spatial_convergence_report.csv`

Build surrogate train/infer datasets from heat trajectory:

```bash
python ForgeFlowApps/heat_periodic/scripts/build_surrogate_datasets.py
```

Run supervised surrogate baseline:

```bash
python main.py --config ForgeFlowApps/heat_periodic/config/surrogate_run.json
```

Run surrogate rollout prediction:

```bash
python ForgeFlowApps/heat_periodic/scripts/run_surrogate_rollout_eval.py
```

Outputs:

- `ForgeFlowApps/heat_periodic/output/surrogate_rollout_steps.csv`
- `ForgeFlowApps/heat_periodic/output/surrogate_rollout_summary.csv`

Generate visual reports:

```bash
python ForgeFlowApps/heat_periodic/scripts/plot_report.py
```

Generate visual reports for the longer-time run:

```bash
python ForgeFlowApps/heat_periodic/scripts/plot_report.py --trajectory ForgeFlowApps/heat_periodic/output/trajectory_long_t.csv --simulation-eval ForgeFlowApps/heat_periodic/output/eval_report_long_t.csv --out-dir ForgeFlowApps/heat_periodic/output/report_long_t
```

Plot outputs:

- `ForgeFlowApps/heat_periodic/output/report/simulation_report.png`
- `ForgeFlowApps/heat_periodic/output/report/surrogate_report.png`
- `ForgeFlowApps/heat_periodic/output/report/rollout_report.png` (if rollout CSV exists)
- `ForgeFlowApps/heat_periodic/output/report_long_t/simulation_report.png` (long-time run)

Report fields are split into two views:

- `*_exact_semidiscrete`: temporal error against the semi-discrete exact mode
  (use these columns for time-order checks)
- `*_exact_continuous`: error against continuous PDE exact mode
  (includes fixed spatial-discretization bias)
