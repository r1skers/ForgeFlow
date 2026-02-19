PYTHON ?= python

LINEAR_CONFIG := ForgeFlowApps/linear_xy/config/run.json
DEM_CONFIG := ForgeFlowApps/dem_diffusion/config/run.json
POLY4_CONFIG := ForgeFlowApps/poly4_cubic/config/run.json
SOLAR_CONFIG := ForgeFlowApps/solar_terms/config/run.json
INK_CONFIG := ForgeFlowApps/ink_diffusion/config/run.json
INK_SAMPLE_SCRIPT := ForgeFlowApps/ink_diffusion/scripts/build_supervised_samples.py
INK_SURROGATE_DATA_SCRIPT := ForgeFlowApps/ink_diffusion/scripts/build_surrogate_datasets.py
INK_SURROGATE_CONFIG := ForgeFlowApps/ink_diffusion/config/surrogate_run.json
INK_MULTI_KAPPA_TRAJECTORY_SCRIPT := ForgeFlowApps/ink_diffusion/scripts/build_multi_kappa_trajectories.py
INK_MULTI_KAPPA_DATA_SCRIPT := ForgeFlowApps/ink_diffusion/scripts/build_multi_kappa_surrogate_data.py
INK_ROLLOUT_SCRIPT := ForgeFlowApps/ink_diffusion/scripts/run_surrogate_rollout_eval.py
INK_SURROGATE_KAPPA_ID_CONFIG := ForgeFlowApps/ink_diffusion/config/surrogate_kappa_id_run.json
INK_SURROGATE_KAPPA_OOD_CONFIG := ForgeFlowApps/ink_diffusion/config/surrogate_kappa_ood_run.json
INK_REPORT_SCRIPT := ForgeFlowApps/ink_diffusion/scripts/plot_report.py
INK_CONVERGENCE_SCRIPT := ForgeFlowApps/ink_diffusion/scripts/run_convergence_study.py
INK_SPATIAL_CONVERGENCE_SCRIPT := ForgeFlowApps/ink_diffusion/scripts/run_spatial_convergence_study.py
INK_VERIFICATION_CONFIG := ForgeFlowApps/ink_diffusion/config/verification.json
HEAT_EXACT_SCRIPT := ForgeFlowApps/heat_periodic/scripts/run_exact_convergence_study.py
HEAT_EXACT_SPATIAL_SCRIPT := ForgeFlowApps/heat_periodic/scripts/run_exact_spatial_convergence_study.py
HEAT_SURROGATE_DATA_SCRIPT := ForgeFlowApps/heat_periodic/scripts/build_surrogate_datasets.py
HEAT_SURROGATE_CONFIG := ForgeFlowApps/heat_periodic/config/surrogate_run.json
HEAT_ROLLOUT_SCRIPT := ForgeFlowApps/heat_periodic/scripts/run_surrogate_rollout_eval.py
HEAT_PLOT_SCRIPT := ForgeFlowApps/heat_periodic/scripts/plot_report.py
HEAT_LONG_CONFIG := ForgeFlowApps/heat_periodic/config/run_long_t.json

.PHONY: run-linear run-dem run-poly4 run-solar run-ink build-ink-samples build-ink-surrogate-data build-ink-multi-kappa-trajectories build-ink-multi-kappa-data run-ink-surrogate run-ink-surrogate-kappa-id run-ink-surrogate-kappa-ood run-ink-rollout plot-ink-report run-ink-convergence run-ink-spatial-convergence run-ink-verify run-heat-long run-heat-exact-convergence run-heat-exact-spatial-convergence build-heat-surrogate-data run-heat-surrogate run-heat-rollout plot-heat-report plot-heat-long smoke

run-linear:
	$(PYTHON) main.py --config $(LINEAR_CONFIG)

run-dem:
	$(PYTHON) main.py --config $(DEM_CONFIG)

run-poly4:
	$(PYTHON) main.py --config $(POLY4_CONFIG)

run-solar:
	$(PYTHON) main.py --config $(SOLAR_CONFIG)

run-ink:
	$(PYTHON) main.py --config $(INK_CONFIG)

build-ink-samples:
	$(PYTHON) $(INK_SAMPLE_SCRIPT)

build-ink-surrogate-data:
	$(PYTHON) $(INK_SURROGATE_DATA_SCRIPT)

build-ink-multi-kappa-trajectories:
	$(PYTHON) $(INK_MULTI_KAPPA_TRAJECTORY_SCRIPT)

build-ink-multi-kappa-data:
	$(PYTHON) $(INK_MULTI_KAPPA_DATA_SCRIPT)

run-ink-surrogate:
	$(PYTHON) main.py --config $(INK_SURROGATE_CONFIG)

run-ink-surrogate-kappa-id:
	$(PYTHON) main.py --config $(INK_SURROGATE_KAPPA_ID_CONFIG)

run-ink-surrogate-kappa-ood:
	$(PYTHON) main.py --config $(INK_SURROGATE_KAPPA_OOD_CONFIG)

run-ink-rollout:
	$(PYTHON) $(INK_ROLLOUT_SCRIPT)

plot-ink-report:
	$(PYTHON) $(INK_REPORT_SCRIPT)

run-ink-convergence:
	$(PYTHON) $(INK_CONVERGENCE_SCRIPT)

run-ink-spatial-convergence:
	$(PYTHON) $(INK_SPATIAL_CONVERGENCE_SCRIPT)

run-ink-verify:
	$(PYTHON) -m forgeflow.core.verification.runner --config $(INK_VERIFICATION_CONFIG)

run-heat-long:
	$(PYTHON) main.py --config $(HEAT_LONG_CONFIG)

run-heat-exact-convergence:
	$(PYTHON) $(HEAT_EXACT_SCRIPT)

run-heat-exact-spatial-convergence:
	$(PYTHON) $(HEAT_EXACT_SPATIAL_SCRIPT)

build-heat-surrogate-data:
	$(PYTHON) $(HEAT_SURROGATE_DATA_SCRIPT)

run-heat-surrogate:
	$(PYTHON) main.py --config $(HEAT_SURROGATE_CONFIG)

run-heat-rollout:
	$(PYTHON) $(HEAT_ROLLOUT_SCRIPT)

plot-heat-report:
	$(PYTHON) $(HEAT_PLOT_SCRIPT)

plot-heat-long:
	$(PYTHON) $(HEAT_PLOT_SCRIPT) --trajectory ForgeFlowApps/heat_periodic/output/trajectory_long_t.csv --simulation-eval ForgeFlowApps/heat_periodic/output/eval_report_long_t.csv --out-dir ForgeFlowApps/heat_periodic/output/report_long_t

smoke: run-linear run-dem
