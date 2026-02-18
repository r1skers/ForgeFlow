from typing import Iterable

from forgeflow.interfaces import AdapterStats, Record, State, ensure_required_fields


class InkGridAdapter:
    """Build a 2D concentration grid from x,y,h rows for ink diffusion."""

    name = "ink_grid"
    required_fields = ("x", "y", "h")

    def build_initial_state(
        self, records: Iterable[Record], simulation: dict[str, object]
    ) -> tuple[State, AdapterStats]:
        parsed_cells: list[tuple[int, int, float]] = []
        stats: AdapterStats = {
            "total_records": 0,
            "valid_states": 0,
            "skipped_state_rows": 0,
        }

        max_x = -1
        max_y = -1
        for record in records:
            stats["total_records"] += 1
            try:
                ensure_required_fields(record, self.required_fields)
                x_idx = int(record["x"])
                y_idx = int(record["y"])
                h_val = float(record["h"])
                if x_idx < 0 or y_idx < 0:
                    raise ValueError("x and y must be >= 0")
            except (KeyError, TypeError, ValueError):
                stats["skipped_state_rows"] += 1
                continue

            parsed_cells.append((x_idx, y_idx, h_val))
            stats["valid_states"] += 1
            max_x = max(max_x, x_idx)
            max_y = max(max_y, y_idx)

        if not parsed_cells:
            raise ValueError("no valid ink cells were parsed from initial_csv")

        grid_nx_raw = simulation.get("grid_nx")
        grid_ny_raw = simulation.get("grid_ny")
        nx = int(grid_nx_raw) if grid_nx_raw is not None else (max_x + 1)
        ny = int(grid_ny_raw) if grid_ny_raw is not None else (max_y + 1)
        if nx <= 0 or ny <= 0:
            raise ValueError("resolved grid shape must be positive")

        grid = [[0.0 for _ in range(nx)] for _ in range(ny)]
        assigned_cells = 0
        clipped_cells = 0
        for x_idx, y_idx, h_val in parsed_cells:
            if x_idx >= nx or y_idx >= ny:
                clipped_cells += 1
                continue
            grid[y_idx][x_idx] = h_val
            assigned_cells += 1

        return (
            {
                "grid": grid,
                "nx": nx,
                "ny": ny,
                "assigned_cells": assigned_cells,
                "clipped_cells": clipped_cells,
            },
            stats,
        )
