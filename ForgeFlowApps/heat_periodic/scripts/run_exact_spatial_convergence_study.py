import argparse
import csv
import importlib
import math
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from forgeflow.core.config.runtime import RuntimeConfig, load_runtime_config
from forgeflow.core.io.csv_reader import read_csv_records


def _load_class_from_ref(ref: str) -> type:
    if ":" in ref:
        module_name, class_name = ref.split(":", 1)
    else:
        module_name, _, class_name = ref.rpartition(".")
    if not module_name or not class_name:
        raise ValueError(f"invalid class reference: {ref}")

    module = importlib.import_module(module_name)
    cls = getattr(module, class_name, None)
    if cls is None or not isinstance(cls, type):
        raise ValueError(f"unable to resolve class reference: {ref}")
    return cls


def _resolve_adapter_class(runtime: RuntimeConfig) -> type:
    if runtime.adapter_ref is not None:
        return _load_class_from_ref(runtime.adapter_ref)
    if runtime.adapter is None:
        raise ValueError("adapter is not configured")
    from forgeflow.plugins.registry import ADAPTER_REGISTRY

    entry = ADAPTER_REGISTRY.get(runtime.adapter)
    if entry is None:
        raise ValueError(f"unknown adapter: {runtime.adapter}")
    if isinstance(entry, str):
        return _load_class_from_ref(entry)
    if isinstance(entry, type):
        return entry
    raise ValueError(f"invalid adapter registry entry: {type(entry).__name__}")


def _resolve_model_class(runtime: RuntimeConfig) -> type:
    if runtime.model_ref is not None:
        return _load_class_from_ref(runtime.model_ref)
    if runtime.model is None:
        raise ValueError("model is not configured")
    from forgeflow.plugins.registry import MODEL_REGISTRY

    entry = MODEL_REGISTRY.get(runtime.model)
    if entry is None:
        raise ValueError(f"unknown model: {runtime.model}")
    if isinstance(entry, str):
        return _load_class_from_ref(entry)
    if isinstance(entry, type):
        return entry
    raise ValueError(f"invalid model registry entry: {type(entry).__name__}")


def _build_base_grid(records: list[dict[str, Any]]) -> tuple[list[list[float]], int, int]:
    parsed_cells: list[tuple[int, int, float]] = []
    max_x = -1
    max_y = -1
    for row in records:
        x_idx = int(row["x"])
        y_idx = int(row["y"])
        h_val = float(row["h"])
        if x_idx < 0 or y_idx < 0:
            raise ValueError("x and y must be >= 0")
        parsed_cells.append((x_idx, y_idx, h_val))
        max_x = max(max_x, x_idx)
        max_y = max(max_y, y_idx)

    if not parsed_cells:
        raise ValueError("initial_csv has no valid cells")

    nx = max_x + 1
    ny = max_y + 1
    grid = [[0.0 for _ in range(nx)] for _ in range(ny)]
    for x_idx, y_idx, h_val in parsed_cells:
        grid[y_idx][x_idx] = h_val
    return grid, nx, ny


def _downsample_grid_by_stride(base_grid: list[list[float]], stride: int) -> list[list[float]]:
    ny = len(base_grid)
    nx = len(base_grid[0])
    if stride <= 0:
        raise ValueError("stride must be > 0")
    if nx % stride != 0 or ny % stride != 0:
        raise ValueError("grid shape must be divisible by stride")

    out: list[list[float]] = []
    for y_idx in range(0, ny, stride):
        row: list[float] = []
        for x_idx in range(0, nx, stride):
            row.append(float(base_grid[y_idx][x_idx]))
        out.append(row)
    return out


def _grid_to_records(grid: list[list[float]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for y_idx, row in enumerate(grid):
        for x_idx, value in enumerate(row):
            rows.append({"x": x_idx, "y": y_idx, "h": float(value)})
    return rows


def _build_exact_grid(
    nx: int,
    ny: int,
    dx: float,
    dy: float,
    t: float,
    kappa: float,
    amplitude: float,
    mode_x: int,
    mode_y: int,
) -> list[list[float]]:
    factor = math.exp(-4.0 * math.pi * math.pi * kappa * ((mode_x**2) + (mode_y**2)) * t)
    grid: list[list[float]] = []
    for y_idx in range(ny):
        y = y_idx * dy
        row: list[float] = []
        for x_idx in range(nx):
            x = x_idx * dx
            h = (
                amplitude
                * math.sin(2.0 * math.pi * mode_x * x)
                * math.sin(2.0 * math.pi * mode_y * y)
                * factor
            )
            row.append(h)
        grid.append(row)
    return grid


def _compute_grid_errors(grid_num: list[list[float]], grid_exact: list[list[float]]) -> tuple[float, float]:
    ny = len(grid_num)
    nx = len(grid_num[0])
    if len(grid_exact) != ny or len(grid_exact[0]) != nx:
        raise ValueError("numeric and exact grid shape mismatch")

    sq_sum = 0.0
    max_abs = 0.0
    count = nx * ny
    for y_idx in range(ny):
        for x_idx in range(nx):
            diff = float(grid_num[y_idx][x_idx]) - float(grid_exact[y_idx][x_idx])
            sq_sum += diff * diff
            max_abs = max(max_abs, abs(diff))

    rmse = math.sqrt(sq_sum / float(count))
    return rmse, max_abs


def _compute_observed_order(
    error_coarse: float,
    error_fine: float,
    h_coarse: float,
    h_fine: float,
) -> float | None:
    if error_coarse <= 0.0 or error_fine <= 0.0:
        return None
    ratio_err = error_coarse / error_fine
    ratio_h = h_coarse / h_fine
    if ratio_err <= 0.0 or ratio_h <= 1.0:
        return None
    return float(math.log(ratio_err) / math.log(ratio_h))


def run_exact_spatial_convergence_study(
    config_path: Path,
    strides: list[int],
    cfl_safety: float,
    total_time: float | None,
    output_csv: Path,
    amplitude: float,
    mode_x: int,
    mode_y: int,
) -> Path:
    if cfl_safety <= 0.0 or cfl_safety > 1.0:
        raise ValueError("cfl_safety must be in (0,1]")

    resolved_config = config_path if config_path.is_absolute() else PROJECT_ROOT / config_path
    runtime = load_runtime_config(resolved_config, PROJECT_ROOT)
    if runtime.mode != "simulation":
        raise ValueError("exact spatial convergence study requires simulation mode config")
    if runtime.paths.initial_csv is None:
        raise ValueError("simulation config must include paths.initial_csv")

    adapter_cls = _resolve_adapter_class(runtime)
    model_cls = _resolve_model_class(runtime)
    records, _ = read_csv_records(runtime.paths.initial_csv)
    if not records:
        raise ValueError("initial_csv contains no valid rows")

    base_grid, base_nx, base_ny = _build_base_grid(records)
    stride_list = sorted({int(s) for s in strides if int(s) > 0}, reverse=True)
    if len(stride_list) < 2:
        raise ValueError("at least two positive strides are required")
    for stride in stride_list:
        if base_nx % stride != 0 or base_ny % stride != 0:
            raise ValueError(
                f"stride={stride} incompatible with base grid {base_nx}x{base_ny}; both dimensions must divide"
            )

    base_total_time = float(runtime.simulation["steps"]) * float(runtime.simulation["dt"])
    target_total_time = base_total_time if total_time is None else float(total_time)
    if target_total_time <= 0.0:
        raise ValueError("total_time must be > 0")

    base_dx = float(runtime.simulation["dx"])
    base_dy = float(runtime.simulation["dy"])
    kappa = float(runtime.simulation["kappa"])

    rows: list[dict[str, Any]] = []
    for stride in stride_list:
        case_grid = _downsample_grid_by_stride(base_grid, stride)
        nx = len(case_grid[0])
        ny = len(case_grid)
        dx = base_dx * stride
        dy = base_dy * stride

        cfl_limit = 1.0 / (2.0 * kappa * ((1.0 / (dx * dx)) + (1.0 / (dy * dy))))
        dt_target = cfl_safety * cfl_limit
        steps = int(math.ceil(target_total_time / dt_target))
        if steps <= 0:
            raise ValueError("resolved steps must be > 0")
        dt = target_total_time / float(steps)

        simulation = dict(runtime.simulation)
        simulation["dx"] = dx
        simulation["dy"] = dy
        simulation["dt"] = dt
        simulation["steps"] = steps
        simulation["grid_nx"] = nx
        simulation["grid_ny"] = ny

        adapter = adapter_cls()
        build_initial_state_fn = getattr(adapter, "build_initial_state", None)
        if not callable(build_initial_state_fn):
            raise ValueError(f"adapter '{adapter_cls.__name__}' must implement build_initial_state()")
        initial_state, _ = build_initial_state_fn(_grid_to_records(case_grid), simulation)

        model = model_cls()
        simulate_fn = getattr(model, "simulate", None)
        if not callable(simulate_fn):
            raise ValueError(f"model '{model_cls.__name__}' must implement simulate()")
        states = simulate_fn(initial_state, simulation)
        if not states:
            raise ValueError("simulation produced no states")

        final_grid = states[-1]["grid"]
        resolved_t = dt * steps
        exact_grid = _build_exact_grid(
            nx=nx,
            ny=ny,
            dx=dx,
            dy=dy,
            t=resolved_t,
            kappa=kappa,
            amplitude=amplitude,
            mode_x=mode_x,
            mode_y=mode_y,
        )
        error_l2, error_linf = _compute_grid_errors(final_grid, exact_grid)
        stable_cfl = bool(states[-1].get("stable_cfl", True))
        row_status = "PASS" if stable_cfl else "FAIL"

        rows.append(
            {
                "case": f"stride_{stride}",
                "stride": stride,
                "nx": nx,
                "ny": ny,
                "dx": dx,
                "dy": dy,
                "dt": dt,
                "steps": steps,
                "total_time": resolved_t,
                "error_l2_exact": error_l2,
                "error_linf_exact": error_linf,
                "observed_order_l2_exact": None,
                "observed_order_linf_exact": None,
                "stable_cfl": stable_cfl,
                "cfl_limit": cfl_limit,
                "status": row_status,
            }
        )

    rows_sorted = sorted(rows, key=lambda item: int(item["stride"]), reverse=True)
    for idx in range(len(rows_sorted) - 1):
        coarse = rows_sorted[idx]
        fine = rows_sorted[idx + 1]
        fine["observed_order_l2_exact"] = _compute_observed_order(
            error_coarse=float(coarse["error_l2_exact"]),
            error_fine=float(fine["error_l2_exact"]),
            h_coarse=float(coarse["dx"]),
            h_fine=float(fine["dx"]),
        )
        fine["observed_order_linf_exact"] = _compute_observed_order(
            error_coarse=float(coarse["error_linf_exact"]),
            error_fine=float(fine["error_linf_exact"]),
            h_coarse=float(coarse["dx"]),
            h_fine=float(fine["dx"]),
        )

    output_path = output_csv if output_csv.is_absolute() else PROJECT_ROOT / output_csv
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "case",
                "stride",
                "nx",
                "ny",
                "dx",
                "dy",
                "dt",
                "steps",
                "total_time",
                "error_l2_exact",
                "error_linf_exact",
                "observed_order_l2_exact",
                "observed_order_linf_exact",
                "stable_cfl",
                "cfl_limit",
                "status",
            ],
        )
        writer.writeheader()
        for row in rows_sorted:
            writer.writerow(
                {
                    "case": row["case"],
                    "stride": int(row["stride"]),
                    "nx": int(row["nx"]),
                    "ny": int(row["ny"]),
                    "dx": f"{float(row['dx']):.10f}",
                    "dy": f"{float(row['dy']):.10f}",
                    "dt": f"{float(row['dt']):.10f}",
                    "steps": int(row["steps"]),
                    "total_time": f"{float(row['total_time']):.10f}",
                    "error_l2_exact": f"{float(row['error_l2_exact']):.12e}",
                    "error_linf_exact": f"{float(row['error_linf_exact']):.12e}",
                    "observed_order_l2_exact": (
                        ""
                        if row["observed_order_l2_exact"] is None
                        else f"{float(row['observed_order_l2_exact']):.6f}"
                    ),
                    "observed_order_linf_exact": (
                        ""
                        if row["observed_order_linf_exact"] is None
                        else f"{float(row['observed_order_linf_exact']):.6f}"
                    ),
                    "stable_cfl": str(bool(row["stable_cfl"])),
                    "cfl_limit": f"{float(row['cfl_limit']):.10f}",
                    "status": str(row["status"]),
                }
            )

    print(f"[heat-spatial] config={resolved_config}")
    print(f"[heat-spatial] base_grid={base_nx}x{base_ny}")
    print(f"[heat-spatial] total_time={target_total_time:.6f}")
    print(f"[heat-spatial] cfl_safety={cfl_safety:.4f}")
    print(f"[heat-spatial] amplitude={amplitude:.6f} mode_x={mode_x} mode_y={mode_y}")
    for row in rows_sorted:
        p_l2 = row.get("observed_order_l2_exact")
        p_linf = row.get("observed_order_linf_exact")
        order_suffix = ""
        if p_l2 is not None and p_linf is not None:
            order_suffix = f" p_l2={float(p_l2):.4f} p_linf={float(p_linf):.4f}"
        print(
            f"[heat-spatial:{row['case']}] grid={int(row['nx'])}x{int(row['ny'])} "
            f"dx={float(row['dx']):.6f} dt={float(row['dt']):.6f} "
            f"l2_exact={float(row['error_l2_exact']):.6e} "
            f"linf_exact={float(row['error_linf_exact']):.6e} status={row['status']}{order_suffix}"
        )
    print(f"[heat-spatial] report_file={output_path.resolve()}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run heat periodic spatial convergence against analytic exact solution."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("ForgeFlowApps/heat_periodic/config/run.json"),
        help="Simulation config path.",
    )
    parser.add_argument(
        "--strides",
        type=int,
        nargs="+",
        default=[4, 2, 1],
        help="Spatial strides over base grid (coarse -> fine).",
    )
    parser.add_argument(
        "--cfl-safety",
        type=float,
        default=0.8,
        help="dt = cfl_safety * cfl_limit before step rounding.",
    )
    parser.add_argument(
        "--total-time",
        type=float,
        default=None,
        help="Physical end time. Defaults to steps*dt from --config.",
    )
    parser.add_argument(
        "--amplitude",
        type=float,
        default=1.0,
        help="Initial mode amplitude in exact solution.",
    )
    parser.add_argument(
        "--mode-x",
        type=int,
        default=1,
        help="Sin mode index along x in exact solution.",
    )
    parser.add_argument(
        "--mode-y",
        type=int,
        default=1,
        help="Sin mode index along y in exact solution.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ForgeFlowApps/heat_periodic/output/exact_spatial_convergence_report.csv"),
        help="Output CSV path.",
    )
    args = parser.parse_args()

    run_exact_spatial_convergence_study(
        config_path=args.config,
        strides=list(args.strides),
        cfl_safety=float(args.cfl_safety),
        total_time=args.total_time,
        output_csv=args.output,
        amplitude=float(args.amplitude),
        mode_x=int(args.mode_x),
        mode_y=int(args.mode_y),
    )


if __name__ == "__main__":
    main()
