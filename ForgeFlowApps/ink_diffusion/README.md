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

Run supervised surrogate baseline:

```bash
python main.py --config ForgeFlowApps/ink_diffusion/config/surrogate_run.json
```

Generate summary plots (simulation + surrogate):

```bash
python ForgeFlowApps/ink_diffusion/scripts/plot_report.py
```
