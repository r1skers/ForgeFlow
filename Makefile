PYTHON ?= python

LINEAR_CONFIG := ForgeFlowApps/linear_xy/config/run.json
DEM_CONFIG := ForgeFlowApps/dem_diffusion/config/run.json
POLY4_CONFIG := ForgeFlowApps/poly4_cubic/config/run.json
SOLAR_CONFIG := ForgeFlowApps/solar_terms/config/run.json
INK_CONFIG := ForgeFlowApps/ink_diffusion/config/run.json
INK_SAMPLE_SCRIPT := ForgeFlowApps/ink_diffusion/scripts/build_supervised_samples.py
INK_SURROGATE_DATA_SCRIPT := ForgeFlowApps/ink_diffusion/scripts/build_surrogate_datasets.py
INK_SURROGATE_CONFIG := ForgeFlowApps/ink_diffusion/config/surrogate_run.json
INK_REPORT_SCRIPT := ForgeFlowApps/ink_diffusion/scripts/plot_report.py

.PHONY: run-linear run-dem run-poly4 run-solar run-ink build-ink-samples build-ink-surrogate-data run-ink-surrogate plot-ink-report smoke

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

run-ink-surrogate:
	$(PYTHON) main.py --config $(INK_SURROGATE_CONFIG)

plot-ink-report:
	$(PYTHON) $(INK_REPORT_SCRIPT)

smoke: run-linear run-dem
