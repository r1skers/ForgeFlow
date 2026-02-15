from forgeflow.plugins.adapters.linear_xy import LinearXYAdapter
from forgeflow.plugins.models.linear_regression import LinearDynamicsRegressor

ADAPTER_REGISTRY = {
    "linear_xy": LinearXYAdapter,
}

MODEL_REGISTRY = {
    "linear_dynamics": LinearDynamicsRegressor,
}

try:
    from ForgeFlowApps.solar_terms.adapters import SolarTermsAdapter

    ADAPTER_REGISTRY["solar_terms_v1"] = SolarTermsAdapter
except ImportError:
    # Keep framework usable even when optional app adapters are absent.
    pass

try:
    from ForgeFlowApps.solar_terms.models import SolarTermsOffsetMeanRegressor

    MODEL_REGISTRY["solar_terms_offset_mean_v1"] = SolarTermsOffsetMeanRegressor
except ImportError:
    # Keep framework usable even when optional app models are absent.
    pass

try:
    from ForgeFlowApps.poly4_cubic.adapters import Poly4CubicAdapter

    ADAPTER_REGISTRY["poly4_cubic_v1"] = Poly4CubicAdapter
except ImportError:
    # Keep framework usable even when optional app adapters are absent.
    pass

try:
    from ForgeFlowApps.poly4_cubic.models import PolyLinearDeg3Regressor

    MODEL_REGISTRY["poly_linear_deg3_v1"] = PolyLinearDeg3Regressor
except ImportError:
    # Keep framework usable even when optional app models are absent.
    pass
