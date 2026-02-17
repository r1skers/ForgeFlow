"""Adapter plugin exports with lazy loading."""

from __future__ import annotations

from typing import Any

__all__ = ["LinearXYAdapter", "DEMGridAdapter"]


def __getattr__(name: str) -> Any:
    if name == "LinearXYAdapter":
        from forgeflow.plugins.adapters.linear_xy import LinearXYAdapter

        return LinearXYAdapter
    if name == "DEMGridAdapter":
        from forgeflow.plugins.adapters.dem_grid import DEMGridAdapter

        return DEMGridAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
