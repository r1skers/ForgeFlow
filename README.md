# ForgeFlow

ForgeFlow v1 is a reproducible modeling-and-inference framework with pluggable adapters and models.

## v1 Scope

- Config-driven pipeline execution (`--config`)
- Pluggable adapter/model architecture
- Regression evaluation (`MAE`, `RMSE`, `MaxAE`) with `PASS/FAIL`
- Residual anomaly flagging (3-sigma rule)
- Example experiment bundles:
  - `experiments/linear_xy`
  - `experiments/linear_xy_noisy`

## Architecture

- `forgeflow/core/`
  - Runtime engine: config loading, IO, split, evaluation, pipeline runner
- `forgeflow/interfaces/`
  - Adapter/model contracts and shared typed structures
- `forgeflow/plugins/`
  - Concrete implementations selected at runtime (adapter/model registry)
- `experiments/<task>/`
  - `config.json`, `data/`, `output/` owned by each experiment

## Run

Default experiment:

```bash
python main.py
```

Noisy experiment:

```bash
python main.py --config experiments/linear_xy_noisy/config.json
```

## Outputs

Each experiment writes to its own `output/` directory:

- `output/predictions.csv`
- `output/eval_report.csv`

`predictions.csv` includes:

- `x`
- `y_pred`
- `residual` (if inference input contains `y`)
- `anomaly_flag` (if residual is available)

## Extend

To add a new domain task:

1. Add adapter/model plugin implementations in `forgeflow/plugins/`.
2. Register them in `forgeflow/plugins/registry.py`.
3. Create `experiments/<new_task>/config.json` + task data files.
4. Run with `python main.py --config experiments/<new_task>/config.json`.

## Notes

- v1 is focused on deterministic regression pipelines and tabular CSV inputs.
- Neural network plugins and richer uncertainty methods are planned for later versions.
