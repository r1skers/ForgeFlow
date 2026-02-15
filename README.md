# ForgeFlow

ForgeFlow v1 is landed.
It provides a reusable training-evaluation-inference framework, and has been validated with linear and polynomial-linear modeling tasks end to end.

## v1 Capabilities

- Config-driven pipeline execution (`python main.py --config ...`)
- Pluggable `Adapter` and `Model` architecture
- Regression evaluation (`MAE`, `RMSE`, `MaxAE`) with `PASS/FAIL`
- Residual anomaly flagging via sigma rule
- Deterministic split controls (`shuffle`, `seed`) for reproducibility
- Chunked inference (`infer.chunk_size`) for large CSV files

## Repository Layout

- `forgeflow/core/`: runtime config, CSV IO, data split, evaluation, pipeline runner
- `forgeflow/interfaces/`: adapter/model contracts and shared types
- `forgeflow/plugins/`: built-in adapters/models and optional registry entries
- `ForgeFlowApps/`: app-level tasks (config, data, adapters, models, outputs)
- `experiments/`: baseline experiment bundles kept for compatibility demos

## Config Styles

Both styles are supported:

- Registry-key style:
  - `adapter`: plugin key in `forgeflow/plugins/registry.py`
  - `model`: plugin key in `forgeflow/plugins/registry.py`
- Class-path style (recommended for app isolation):
  - `adapter_ref`: `package.module:ClassName`
  - `model_ref`: `package.module:ClassName`

## Quick Run

Linear baseline:

```bash
python main.py --config experiments/linear_xy/config.json
```

Poly4 cubic app:

```bash
python main.py --config ForgeFlowApps/poly4_cubic/config/run.json
```

## Outputs

Each task writes:

- `output/predictions.csv`
- `output/eval_report.csv`

`predictions.csv` contains `y_pred`, and contains `residual`/`anomaly_flag` when labeled `y` is provided in inference input.

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
