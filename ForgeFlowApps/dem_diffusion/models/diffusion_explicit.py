from forgeflow.interfaces import State


class DiffusionExplicitSimulator:
    def summary(self) -> dict[str, str]:
        return {
            "solver": "explicit_diffusion_2d",
            "scheme": "forward_euler_five_point",
        }

    def simulate(self, initial_state: State, simulation: dict[str, object]) -> list[State]:
        raw_grid = initial_state.get("grid")
        if not isinstance(raw_grid, list) or not raw_grid:
            raise ValueError("initial_state.grid must be a non-empty 2D list")
        if not all(isinstance(row, list) and row for row in raw_grid):
            raise ValueError("initial_state.grid must be rectangular and non-empty")

        ny = len(raw_grid)
        nx = len(raw_grid[0])
        if any(len(row) != nx for row in raw_grid):
            raise ValueError("initial_state.grid rows must have the same width")

        steps = int(simulation["steps"])
        dt = float(simulation["dt"])
        dx = float(simulation["dx"])
        dy = float(simulation["dy"])
        kappa = float(simulation["kappa"])
        boundary = str(simulation["boundary"])
        strict_cfl = bool(simulation["strict_cfl"])

        cfl_limit = 1.0 / (2.0 * kappa * ((1.0 / (dx * dx)) + (1.0 / (dy * dy))))
        stable_cfl = dt <= (cfl_limit + 1e-12)
        if strict_cfl and not stable_cfl:
            raise ValueError(
                f"unstable dt for explicit diffusion: dt={dt:.6f} > cfl_limit={cfl_limit:.6f}"
            )

        grid = [[float(cell) for cell in row] for row in raw_grid]
        states: list[State] = []
        states.append(self._make_state(0, grid, cfl_limit, stable_cfl, boundary))

        for step in range(1, steps + 1):
            grid = self._step_once(grid, dt, dx, dy, kappa, boundary)
            states.append(self._make_state(step, grid, cfl_limit, stable_cfl, boundary))

        return states

    def _step_once(
        self,
        grid: list[list[float]],
        dt: float,
        dx: float,
        dy: float,
        kappa: float,
        boundary: str,
    ) -> list[list[float]]:
        ny = len(grid)
        nx = len(grid[0])
        next_grid = [[0.0 for _ in range(nx)] for _ in range(ny)]

        for y_idx in range(ny):
            for x_idx in range(nx):
                if boundary == "dirichlet0" and (
                    x_idx == 0 or y_idx == 0 or x_idx == nx - 1 or y_idx == ny - 1
                ):
                    next_grid[y_idx][x_idx] = 0.0
                    continue

                center = grid[y_idx][x_idx]
                left = self._neighbor(grid, x_idx - 1, y_idx, boundary)
                right = self._neighbor(grid, x_idx + 1, y_idx, boundary)
                up = self._neighbor(grid, x_idx, y_idx - 1, boundary)
                down = self._neighbor(grid, x_idx, y_idx + 1, boundary)

                laplacian = ((left - 2.0 * center + right) / (dx * dx)) + (
                    (up - 2.0 * center + down) / (dy * dy)
                )
                next_grid[y_idx][x_idx] = center + (dt * kappa * laplacian)

        return next_grid

    def _neighbor(self, grid: list[list[float]], x_idx: int, y_idx: int, boundary: str) -> float:
        ny = len(grid)
        nx = len(grid[0])

        if 0 <= x_idx < nx and 0 <= y_idx < ny:
            return grid[y_idx][x_idx]

        if boundary == "periodic":
            return grid[y_idx % ny][x_idx % nx]
        if boundary == "neumann":
            clamped_x = min(max(x_idx, 0), nx - 1)
            clamped_y = min(max(y_idx, 0), ny - 1)
            return grid[clamped_y][clamped_x]
        if boundary == "dirichlet0":
            return 0.0
        raise ValueError(f"unsupported boundary: {boundary}")

    def _make_state(
        self,
        step: int,
        grid: list[list[float]],
        cfl_limit: float,
        stable_cfl: bool,
        boundary: str,
    ) -> State:
        flat = [float(cell) for row in grid for cell in row]
        return {
            "step": step,
            "grid": [[float(cell) for cell in row] for row in grid],
            "mass": float(sum(flat)),
            "min_h": float(min(flat)),
            "max_h": float(max(flat)),
            "cfl_limit": float(cfl_limit),
            "stable_cfl": bool(stable_cfl),
            "boundary": boundary,
        }
