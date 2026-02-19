import argparse
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


def _compute_grid_mass(grid: list[list[float]]) -> float:
    return float(sum(sum(float(cell) for cell in row) for row in grid))


def _compute_grid_errors(
    candidate_grid: list[list[float]], reference_grid: list[list[float]]
) -> tuple[float, float]:
    if len(candidate_grid) != len(reference_grid):
        raise ValueError("candidate and reference grids must have same ny")
    if len(candidate_grid[0]) != len(reference_grid[0]):
        raise ValueError("candidate and reference grids must have same nx")

    sq_sum = 0.0
    max_abs = 0.0
    count = 0
    for y_idx, row in enumerate(candidate_grid):
        for x_idx, value in enumerate(row):
            diff = float(value) - float(reference_grid[y_idx][x_idx])
            abs_diff = abs(diff)
            sq_sum += diff * diff
            max_abs = max(max_abs, abs_diff)
            count += 1

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
    ratio_error = error_coarse / error_fine
    ratio_h = h_coarse / h_fine
    if ratio_error <= 0.0 or ratio_h <= 1.0:
        return None
    return float(math.log(ratio_error) / math.log(ratio_h))


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

    downsampled: list[list[float]] = []
    for y_idx in range(0, ny, stride):
        row: list[float] = []
        for x_idx in range(0, nx, stride):
            row.append(float(base_grid[y_idx][x_idx]))
        downsampled.append(row)
    return downsampled


def _sample_reference_to_stride(reference_grid: list[list[float]], stride: int) -> list[list[float]]:
    return _downsample_grid_by_stride(reference_grid, stride)


def _grid_to_records(grid: list[list[float]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for y_idx, row in enumerate(grid):
        for x_idx, cell in enumerate(row):
            records.append({"x": x_idx, "y": y_idx, "h": float(cell)})
    return records


def _run_case(
    adapter_cls: type,
    model_cls: type,
    records: list[dict[str, Any]],
    simulation: dict[str, Any],
    label: str,
    stride: int,
) -> dict[str, Any]:
    adapter = adapter_cls()
    build_initial_state_fn = getattr(adapter, "build_initial_state", None)
    if not callable(build_initial_state_fn):
        raise ValueError(f"adapter '{adapter_cls.__name__}' must implement build_initial_state()")

    initial_state, adapter_stats = build_initial_state_fn(records, simulation)
    model = model_cls()
    simulate_fn = getattr(model, "simulate", None)
    if not callable(simulate_fn):
        raise ValueError(f"model '{model_cls.__name__}' must implement simulate()")

    states = simulate_fn(initial_state, simulation)
    if not states:
        raise ValueError("simulation produced no states")

    initial_grid = states[0]["grid"]
    final_grid = states[-1]["grid"]
    mass_initial = _compute_grid_mass(initial_grid)
    mass_final = _compute_grid_mass(final_grid)
    mass_delta_abs = abs(mass_final - mass_initial)
    cfl_limit = float(states[-1].get("cfl_limit", float("nan")))
    stable_cfl = bool(states[-1].get("stable_cfl", True))
    boundary = str(simulation["boundary"])
    mass_tolerance = float(simulation["mass_tolerance"])
    should_check_mass = boundary in {"neumann", "periodic"}
    mass_ok = (mass_delta_abs <= mass_tolerance) if should_check_mass else True
    status = "PASS" if stable_cfl and mass_ok else "FAIL"

    return {
        "case": label,
        "stride": stride,
        "nx": len(final_grid[0]),
        "ny": len(final_grid),
        "dx": float(simulation["dx"]),
        "dy": float(simulation["dy"]),
        "dt": float(simulation["dt"]),
        "steps": int(simulation["steps"]),
        "total_time": float(simulation["dt"]) * int(simulation["steps"]),
        "stable_cfl": stable_cfl,
        "cfl_limit": cfl_limit,
        "mass_initial": mass_initial,
        "mass_final": mass_final,
        "mass_delta_abs": mass_delta_abs,
        "mass_tolerance": mass_tolerance,
        "mass_check": "checked" if should_check_mass else "skipped",
        "status": status,
        "valid_initial_rows": int(adapter_stats["valid_states"]),
        "final_grid": final_grid,
        "error_l2_vs_ref": 0.0,
        "error_linf_vs_ref": 0.0,
        "observed_order_l2": None,
        "observed_order_linf": None,
    }


def run_spatial_convergence_study(
    config_path: Path,
    output_csv: Path,
    strides: list[int],
    cfl_safety: float,
    total_time: float | None,
) -> None:
    if cfl_safety <= 0.0 or cfl_safety > 1.0:
        raise ValueError("cfl_safety must be in (0, 1]")

    resolved_config = config_path if config_path.is_absolute() else PROJECT_ROOT / config_path
    runtime = load_runtime_config(resolved_config, PROJECT_ROOT)
    if runtime.mode != "simulation":
        raise ValueError("spatial convergence study requires a simulation mode config")
    if runtime.paths.initial_csv is None:
        raise ValueError("simulation config must provide paths.initial_csv")

    adapter_cls = _resolve_adapter_class(runtime)
    model_cls = _resolve_model_class(runtime)

    records, stats = read_csv_records(runtime.paths.initial_csv)
    if not records:
        raise ValueError("initial_csv contains no valid rows")
    base_grid, base_nx, base_ny = _build_base_grid(records)

    unique_strides = sorted({int(s) for s in strides if int(s) > 0}, reverse=True)
    if len(unique_strides) < 2:
        raise ValueError("at least two positive strides are required")

    for stride in unique_strides:
        if base_nx % stride != 0 or base_ny % stride != 0:
            raise ValueError(
                f"stride={stride} is invalid for base grid {base_nx}x{base_ny}; "
                "both dimensions must be divisible by stride"
            )

    simulation_total_time = float(runtime.simulation["steps"]) * float(runtime.simulation["dt"])
    target_total_time = simulation_total_time if total_time is None else float(total_time)
    if target_total_time <= 0.0:
        raise ValueError("total_time must be > 0")

    base_dx = float(runtime.simulation["dx"])
    base_dy = float(runtime.simulation["dy"])
    kappa = float(runtime.simulation["kappa"])
    if kappa <= 0.0:
        raise ValueError("simulation.kappa must be > 0")

    cases: list[dict[str, Any]] = []
    for stride in unique_strides:
        grid_case = _downsample_grid_by_stride(base_grid, stride)
        nx = len(grid_case[0])
        ny = len(grid_case)
        dx = base_dx * stride
        dy = base_dy * stride
        cfl_limit = 1.0 / (2.0 * kappa * ((1.0 / (dx * dx)) + (1.0 / (dy * dy))))
        dt_target = cfl_safety * cfl_limit
        steps = int(math.ceil(target_total_time / dt_target))
        if steps <= 0:
            raise ValueError("resolved steps must be > 0")
        dt = target_total_time / float(steps)

        simulation = dict(runtime.simulation)
        simulation["grid_nx"] = nx
        simulation["grid_ny"] = ny
        simulation["dx"] = dx
        simulation["dy"] = dy
        simulation["dt"] = dt
        simulation["steps"] = steps

        case_records = _grid_to_records(grid_case)
        case_label = f"stride_{stride}"
        case = _run_case(
            adapter_cls=adapter_cls,
            model_cls=model_cls,
            records=case_records,
            simulation=simulation,
            label=case_label,
            stride=stride,
        )
        cases.append(case)

    # Finest (smallest stride) is the reference solution.
    reference_case = min(cases, key=lambda row: int(row["stride"]))
    reference_stride = int(reference_case["stride"])
    reference_grid = reference_case["final_grid"]
    reference_label = str(reference_case["case"])

    for case in cases:
        stride = int(case["stride"])
        if stride == reference_stride:
            continue
        reference_sampled = _sample_reference_to_stride(reference_grid, stride)
        rmse, linf = _compute_grid_errors(case["final_grid"], reference_sampled)
        case["error_l2_vs_ref"] = rmse
        case["error_linf_vs_ref"] = linf

    # Estimate spatial order p from adjacent grid levels.
    cases_by_stride = sorted(cases, key=lambda row: int(row["stride"]), reverse=True)
    for idx in range(len(cases_by_stride) - 1):
        coarse = cases_by_stride[idx]
        fine = cases_by_stride[idx + 1]
        fine["observed_order_l2"] = _compute_observed_order(
            error_coarse=float(coarse["error_l2_vs_ref"]),
            error_fine=float(fine["error_l2_vs_ref"]),
            h_coarse=float(coarse["dx"]),
            h_fine=float(fine["dx"]),
        )
        fine["observed_order_linf"] = _compute_observed_order(
            error_coarse=float(coarse["error_linf_vs_ref"]),
            error_fine=float(fine["error_linf_vs_ref"]),
            h_coarse=float(coarse["dx"]),
            h_fine=float(fine["dx"]),
        )

    output_path = output_csv if output_csv.is_absolute() else PROJECT_ROOT / output_csv
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as out_file:
        out_file.write(
            "case,stride,nx,ny,dx,dy,dt,steps,total_time,stable_cfl,cfl_limit,"
            "mass_initial,mass_final,mass_delta_abs,mass_tolerance,mass_check,status,"
            "error_l2_vs_ref,error_linf_vs_ref,observed_order_l2,observed_order_linf,"
            "reference_case,base_valid_rows\n"
        )
        for case in cases_by_stride:
            observed_order_l2 = (
                "" if case["observed_order_l2"] is None else f"{float(case['observed_order_l2']):.6f}"
            )
            observed_order_linf = (
                ""
                if case["observed_order_linf"] is None
                else f"{float(case['observed_order_linf']):.6f}"
            )
            out_file.write(
                (
                    f"{case['case']},{int(case['stride'])},{int(case['nx'])},{int(case['ny'])},"
                    f"{float(case['dx']):.8f},{float(case['dy']):.8f},{float(case['dt']):.8f},"
                    f"{int(case['steps'])},{float(case['total_time']):.8f},"
                    f"{bool(case['stable_cfl'])},{float(case['cfl_limit']):.8f},"
                    f"{float(case['mass_initial']):.12f},{float(case['mass_final']):.12f},"
                    f"{float(case['mass_delta_abs']):.12f},{float(case['mass_tolerance']):.12f},"
                    f"{case['mass_check']},{case['status']},"
                    f"{float(case['error_l2_vs_ref']):.12f},{float(case['error_linf_vs_ref']):.12f},"
                    f"{observed_order_l2},{observed_order_linf},"
                    f"{reference_label},{int(stats['valid_rows'])}\n"
                )
            )

    print(f"[spatial] config={resolved_config}")
    print(f"[spatial] base_grid={base_nx}x{base_ny}")
    print(f"[spatial] total_time={target_total_time:.6f}")
    print(f"[spatial] cfl_safety={cfl_safety:.4f}")
    print(f"[spatial] reference_case={reference_label}")
    for case in cases_by_stride:
        print(
            f"[spatial:{case['case']}] stride={int(case['stride'])} grid={int(case['nx'])}x{int(case['ny'])} "
            f"dx={float(case['dx']):.4f} dt={float(case['dt']):.6f} steps={int(case['steps'])} "
            f"l2={float(case['error_l2_vs_ref']):.8f} linf={float(case['error_linf_vs_ref']):.8f} "
            f"mass_delta={float(case['mass_delta_abs']):.8e} status={case['status']}"
        )
    print(f"[spatial] report_file={output_path.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run spatial convergence study for the ink diffusion app with nested grids. "
            "Coarser levels are sampled from the base initial grid; each case runs to the "
            "same physical end time with dt set from a CFL safety factor."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("ForgeFlowApps/ink_diffusion/config/run.json"),
        help="Simulation config path.",
    )
    parser.add_argument(
        "--strides",
        type=int,
        nargs="+",
        default=[4, 2, 1],
        help="Spatial strides against the base grid. Smallest stride is the reference case.",
    )
    parser.add_argument(
        "--cfl-safety",
        type=float,
        default=0.8,
        help="dt = cfl_safety * cfl_limit per level before step rounding.",
    )
    parser.add_argument(
        "--total-time",
        type=float,
        default=None,
        help="Physical end time. Defaults to steps*dt from --config.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ForgeFlowApps/ink_diffusion/output/spatial_convergence_report.csv"),
        help="Spatial convergence report CSV output path.",
    )
    args = parser.parse_args()

    run_spatial_convergence_study(
        config_path=args.config,
        output_csv=args.output,
        strides=list(args.strides),
        cfl_safety=float(args.cfl_safety),
        total_time=args.total_time,
    )


if __name__ == "__main__":
    main()
