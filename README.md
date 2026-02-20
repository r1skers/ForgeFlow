# ForgeFlow

ForgeFlow v1 is landed.
It provides a reusable training-evaluation-inference framework, and has been validated with linear and polynomial-linear modeling tasks end to end.
It now also supports a simulation execution mode for PDE-style tasks.

## v1 Capabilities

- Config-driven pipeline execution (`python main.py --config ...`)
- Dual execution modes: `supervised` and `simulation`
- Pluggable `Adapter` and `Model` architecture
- Regression evaluation (`MAE`, `RMSE`, `MaxAE`) with `PASS/FAIL`
- Residual anomaly flagging via sigma rule
- Deterministic split controls (`shuffle`, `seed`) for reproducibility
- Chunked inference (`infer.chunk_size`) for large CSV files

## Repository Layout

- `forgeflow/core/`: runtime config, CSV IO, data split, evaluation, pipeline runner
- `forgeflow/interfaces/`: adapter/model contracts and shared types
- `forgeflow/plugins/`: registry wrappers and optional compatibility aliases
- `ForgeFlowApps/`: app-level tasks (config, data, adapters, models, outputs)
- `experiments/`: legacy demos kept for compatibility checks

## Config Styles

Both styles are supported:

- Registry-key style:
  - `adapter`: plugin key in `forgeflow/plugins/registry.py`
  - `model`: plugin key in `forgeflow/plugins/registry.py`
- Class-path style (recommended for app isolation):
  - `adapter_ref`: `package.module:ClassName`
  - `model_ref`: `package.module:ClassName`

`mode` controls which runtime branch is executed:

- `supervised` (default): split -> fit/predict -> metrics/anomaly -> inference output
- `simulation`: initial state -> time stepping simulation -> trajectory/eval report

## Quick Start (Install + Run)

Prerequisite: Python 3.10+.

Using `venv`:

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Using Conda:

```bash
conda create -n forgeflow python=3.13 -y
conda activate forgeflow
pip install -r requirements.txt
```

Sanity run (linear baseline):

```bash
python main.py --config ForgeFlowApps/linear_xy/config/run.json
```

## Quick Run

Default run (same as linear app config):

```bash
python main.py
```

Linear baseline:

```bash
python main.py --config ForgeFlowApps/linear_xy/config/run.json
```

Poly4 cubic app:

```bash
python main.py --config ForgeFlowApps/poly4_cubic/config/run.json
```

DEM diffusion simulation:

```bash
python main.py --config ForgeFlowApps/dem_diffusion/config/run.json
```

Ink diffusion simulation:

```bash
python main.py --config ForgeFlowApps/ink_diffusion/config/run.json
```

Ink diffusion surrogate regression:

```bash
python main.py --config ForgeFlowApps/ink_diffusion/config/surrogate_run.json
```

Ink diffusion kappa-aware surrogate regression (ID / OOD infer):

```bash
python main.py --config ForgeFlowApps/ink_diffusion/config/surrogate_kappa_id_run.json
python main.py --config ForgeFlowApps/ink_diffusion/config/surrogate_kappa_ood_run.json
```

Ink diffusion surrogate rollout prediction:

```bash
python ForgeFlowApps/ink_diffusion/scripts/run_surrogate_rollout_eval.py
```

Heat kappa inverse (recommended minimal run):

```bash
python ForgeFlowApps/heat_kappa_inverse/stage1_data_gen/scripts/build_dataset.py
python main.py --config ForgeFlowApps/heat_kappa_inverse/stage2_inverse/config/run_id.json
python main.py --config ForgeFlowApps/heat_kappa_inverse/stage2_inverse/config/run_ood.json
python ForgeFlowApps/heat_kappa_inverse/stage2_inverse/scripts/summarize_infer_metrics.py --skip-missing
```

Heat kappa inverse (full robustness report: noise + plots + markdown):

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

Optional `make` shortcuts:

```bash
make run-linear
make run-dem
make run-ink
make run-ink-convergence
make run-ink-spatial-convergence
make run-ink-verify
make build-ink-multi-kappa-trajectories
make build-ink-multi-kappa-data
make run-ink-surrogate-kappa-id
make run-ink-surrogate-kappa-ood
make run-ink-rollout
make build-ink-surrogate-data
make run-ink-surrogate
make plot-ink-report
make build-heat-kappa-data
make run-heat-kappa-id
make run-heat-kappa-ood
make run-heat-kappa-noise-sweep
make report-heat-kappa-infer
make plot-heat-kappa-scatter
make report-heat-kappa-summary
make report-heat-kappa-sigma-sweep
```

If `make` is unavailable on Windows, use the equivalent `python main.py --config ...` commands above, or run `mingw32-make`.

## Outputs

Each task writes:

- `output/eval_report.csv`

Mode-specific outputs:

- `supervised`: `output/predictions.csv` + `output/eval_report.csv`
- `simulation`: `output/trajectory.csv` + `output/eval_report.csv`

For supervised tasks, `predictions.csv` contains `y_pred`, and includes `residual`/`anomaly_flag` when labeled `y` is provided in inference input.

Verification suites can additionally output `output/verification_summary.csv` (step-level PASS/FAIL aggregation).

## Add a New App Task

1. Create `ForgeFlowApps/<task_name>/`.
2. Prepare `config/run.json`, `data/processed/train.csv`, `data/infer/infer.csv`.
3. Implement app adapter and model classes.
4. Set `adapter_ref` and `model_ref` in config.
5. Run `python main.py --config ForgeFlowApps/<task_name>/config/run.json`.

## Commit Convention

Use one unified format:

```text
<type>(<scope>): <summary>
```

Recommended `type` values:

- `feat`: new feature
- `fix`: bug fix
- `refactor`: code restructure without behavior change
- `perf`: performance improvement
- `docs`: documentation change
- `test`: test-related change
- `chore`: maintenance tasks
- `data`: dataset/config/output updates

Examples:

- `feat(core): support adapter_ref/model_ref dynamic loading`
- `refactor(runner): switch to model-capability summary logging`
- `data(poly4_cubic): add synthetic train and infer datasets`
- `docs(readme): mark ForgeFlow v1 milestone`

A reusable commit template is provided in `.gitmessage.txt`.
