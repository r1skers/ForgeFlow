"""
Plugin registry stores lazy class references.

Using strings avoids importing optional app/model modules at framework startup.
"""

ADAPTER_REGISTRY: dict[str, str] = {
    "linear_xy": "forgeflow.plugins.adapters.linear_xy:LinearXYAdapter",
    "dem_grid": "forgeflow.plugins.adapters.dem_grid:DEMGridAdapter",
    "solar_terms_v1": "ForgeFlowApps.solar_terms.adapters.solar_terms_adapter:SolarTermsAdapter",
    "poly4_cubic_v1": "ForgeFlowApps.poly4_cubic.adapters.poly4_cubic_adapter:Poly4CubicAdapter",
}

MODEL_REGISTRY: dict[str, str] = {
    "linear_dynamics": "forgeflow.plugins.models.linear_regression:LinearDynamicsRegressor",
    "diffusion_explicit": "forgeflow.plugins.models.diffusion_explicit:DiffusionExplicitSimulator",
    "solar_terms_offset_mean_v1": (
        "ForgeFlowApps.solar_terms.models.offset_mean:SolarTermsOffsetMeanRegressor"
    ),
    "poly_linear_deg3_v1": "ForgeFlowApps.poly4_cubic.models.polynomial_linear:PolyLinearDeg3Regressor",
}
