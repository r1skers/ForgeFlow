from forgeflow.plugins.adapters.linear_xy import LinearXYAdapter
from forgeflow.plugins.models.linear_regression import LinearDynamicsRegressor

ADAPTER_REGISTRY = {
    "linear_xy": LinearXYAdapter,
}

MODEL_REGISTRY = {
    "linear_dynamics": LinearDynamicsRegressor,
}
