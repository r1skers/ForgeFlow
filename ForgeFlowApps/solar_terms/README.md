# Solar Terms Application Workspace

This folder is an external application workspace for ForgeFlow.

## Structure
- config/: runnable configs for this application
- data/raw/: raw source files
- data/processed/: training files
- data/infer/: inference files
- output/: runtime outputs
- tests/: smoke test cases and expected behavior

## Run
python D:/Github_Repos/ForgeFlow/main.py --config D:/Github_Repos/ForgeFlowApps/solar_terms/config/run.json

## Notes
Current workspace uses `solar_terms_v1` adapter.
Feature mapping:
- day_offset_to_mangzhong
- temp_max
- temp_min
- wind_level
- year
- day_of_year
Target:
- temp_mean

When a dedicated solar_terms adapter/model is ready, update only config + plugins.
