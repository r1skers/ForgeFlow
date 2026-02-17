PYTHON ?= python

LINEAR_CONFIG := ForgeFlowApps/linear_xy/config/run.json
DEM_CONFIG := ForgeFlowApps/dem_diffusion/config/run.json
POLY4_CONFIG := ForgeFlowApps/poly4_cubic/config/run.json
SOLAR_CONFIG := ForgeFlowApps/solar_terms/config/run.json

.PHONY: run-linear run-dem run-poly4 run-solar smoke

run-linear:
	$(PYTHON) main.py --config $(LINEAR_CONFIG)

run-dem:
	$(PYTHON) main.py --config $(DEM_CONFIG)

run-poly4:
	$(PYTHON) main.py --config $(POLY4_CONFIG)

run-solar:
	$(PYTHON) main.py --config $(SOLAR_CONFIG)

smoke: run-linear run-dem
