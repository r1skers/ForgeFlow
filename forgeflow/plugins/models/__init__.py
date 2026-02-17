"""Model plugin exports with lazy loading."""

from __future__ import annotations

from typing import Any

__all__ = ["LinearDynamicsRegressor", "DiffusionExplicitSimulator"]


def __getattr__(name: str) -> Any:
    if name == "LinearDynamicsRegressor":
        from forgeflow.plugins.models.linear_regression import LinearDynamicsRegressor

        return LinearDynamicsRegressor
    if name == "DiffusionExplicitSimulator":
        from forgeflow.plugins.models.diffusion_explicit import DiffusionExplicitSimulator

        return DiffusionExplicitSimulator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
