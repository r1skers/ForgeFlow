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


def _build_semidiscrete_exact_grid(
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
    lambda_x = -4.0 * math.sin(math.pi * mode_x * dx) ** 2 / (dx * dx)
    lambda_y = -4.0 * math.sin(math.pi * mode_y * dy) ** 2 / (dy * dy)
    factor = math.exp(kappa * (lambda_x + lambda_y) * t)
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


def _compute_grid_errors(
    grid_num: list[list[float]], grid_exact: list[list[float]]
) -> tuple[float, float]:
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
    dt_coarse: float,
    dt_fine: float,
) -> float | None:
    if error_coarse <= 0.0 or error_fine <= 0.0:
        return None
    ratio_err = error_coarse / error_fine
    ratio_dt = dt_coarse / dt_fine
    if ratio_err <= 0.0 or ratio_dt <= 1.0:
        return None
    return float(math.log(ratio_err) / math.log(ratio_dt))


def run_exact_convergence_study(
    config_path: Path,
    dt_values: list[float],
    total_time: float | None,
    output_csv: Path,
    amplitude: float,
    mode_x: int,
    mode_y: int,
) -> Path:
    resolved_config = config_path if config_path.is_absolute() else PROJECT_ROOT / config_path
    runtime = load_runtime_config(resolved_config, PROJECT_ROOT)
    if runtime.mode != "simulation":
        raise ValueError("exact convergence study requires simulation mode config")
    if runtime.paths.initial_csv is None:
        raise ValueError("simulation config must include paths.initial_csv")

    adapter_cls = _resolve_adapter_class(runtime)
    model_cls = _resolve_model_class(runtime)
    records, _ = read_csv_records(runtime.paths.initial_csv)
    if not records:
        raise ValueError("initial_csv contains no valid rows")

    dt_list = sorted({float(dt) for dt in dt_values if float(dt) > 0.0}, reverse=True)
    if len(dt_list) < 2:
        raise ValueError("at least two positive dt values are required")

    base_total_time = float(runtime.simulation["steps"]) * float(runtime.simulation["dt"])
    target_total_time = base_total_time if total_time is None else float(total_time)
    if target_total_time <= 0.0:
        raise ValueError("total_time must be > 0")

    rows: list[dict[str, Any]] = []
    for dt in dt_list:
        steps = int(round(target_total_time / dt))
        if steps <= 0:
            raise ValueError("resolved steps must be > 0")
        resolved_t = steps * dt

        simulation = dict(runtime.simulation)
        simulation["dt"] = dt
        simulation["steps"] = steps

        adapter = adapter_cls()
        build_initial_state_fn = getattr(adapter, "build_initial_state", None)
        if not callable(build_initial_state_fn):
            raise ValueError(f"adapter '{adapter_cls.__name__}' must implement build_initial_state()")
        initial_state, _ = build_initial_state_fn(records, simulation)

        model = model_cls()
        simulate_fn = getattr(model, "simulate", None)
        if not callable(simulate_fn):
            raise ValueError(f"model '{model_cls.__name__}' must implement simulate()")
        states = simulate_fn(initial_state, simulation)
        if not states:
            raise ValueError("simulation produced no states")

        final_grid = states[-1]["grid"]
        ny = len(final_grid)
        nx = len(final_grid[0]) if ny > 0 else 0
        dx = float(simulation["dx"])
        dy = float(simulation["dy"])
        kappa = float(simulation["kappa"])

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
        semidiscrete_grid = _build_semidiscrete_exact_grid(
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
        error_l2_continuous, error_linf_continuous = _compute_grid_errors(final_grid, exact_grid)
        error_l2_temporal, error_linf_temporal = _compute_grid_errors(final_grid, semidiscrete_grid)
        stable_cfl = bool(states[-1].get("stable_cfl", True))
        cfl_limit = float(states[-1].get("cfl_limit", float("nan")))

        rows.append(
            {
                "case": f"dt_{dt:g}",
                "dt": dt,
                "steps": steps,
                "total_time": resolved_t,
                "error_l2_exact_continuous": error_l2_continuous,
                "error_linf_exact_continuous": error_linf_continuous,
                "error_l2_exact_semidiscrete": error_l2_temporal,
                "error_linf_exact_semidiscrete": error_linf_temporal,
                "observed_order_l2_exact_continuous": None,
                "observed_order_linf_exact_continuous": None,
                "observed_order_l2_exact_semidiscrete": None,
                "observed_order_linf_exact_semidiscrete": None,
                "stable_cfl": stable_cfl,
                "cfl_limit": cfl_limit,
                "status": "PASS" if stable_cfl else "FAIL",
            }
        )

    rows_sorted = sorted(rows, key=lambda item: float(item["dt"]), reverse=True)
    for idx in range(len(rows_sorted) - 1):
        coarse = rows_sorted[idx]
        fine = rows_sorted[idx + 1]
        fine["observed_order_l2_exact_continuous"] = _compute_observed_order(
            error_coarse=float(coarse["error_l2_exact_continuous"]),
            error_fine=float(fine["error_l2_exact_continuous"]),
            dt_coarse=float(coarse["dt"]),
            dt_fine=float(fine["dt"]),
        )
        fine["observed_order_linf_exact_continuous"] = _compute_observed_order(
            error_coarse=float(coarse["error_linf_exact_continuous"]),
            error_fine=float(fine["error_linf_exact_continuous"]),
            dt_coarse=float(coarse["dt"]),
            dt_fine=float(fine["dt"]),
        )
        fine["observed_order_l2_exact_semidiscrete"] = _compute_observed_order(
            error_coarse=float(coarse["error_l2_exact_semidiscrete"]),
            error_fine=float(fine["error_l2_exact_semidiscrete"]),
            dt_coarse=float(coarse["dt"]),
            dt_fine=float(fine["dt"]),
        )
        fine["observed_order_linf_exact_semidiscrete"] = _compute_observed_order(
            error_coarse=float(coarse["error_linf_exact_semidiscrete"]),
            error_fine=float(fine["error_linf_exact_semidiscrete"]),
            dt_coarse=float(coarse["dt"]),
            dt_fine=float(fine["dt"]),
        )

    output_path = output_csv if output_csv.is_absolute() else PROJECT_ROOT / output_csv
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "case",
                "dt",
                "steps",
                "total_time",
                "error_l2_exact_continuous",
                "error_linf_exact_continuous",
                "error_l2_exact_semidiscrete",
                "error_linf_exact_semidiscrete",
                "observed_order_l2_exact_continuous",
                "observed_order_linf_exact_continuous",
                "observed_order_l2_exact_semidiscrete",
                "observed_order_linf_exact_semidiscrete",
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
                    "dt": f"{float(row['dt']):.10f}",
                    "steps": int(row["steps"]),
                    "total_time": f"{float(row['total_time']):.10f}",
                    "error_l2_exact_continuous": f"{float(row['error_l2_exact_continuous']):.12e}",
                    "error_linf_exact_continuous": f"{float(row['error_linf_exact_continuous']):.12e}",
                    "error_l2_exact_semidiscrete": f"{float(row['error_l2_exact_semidiscrete']):.12e}",
                    "error_linf_exact_semidiscrete": f"{float(row['error_linf_exact_semidiscrete']):.12e}",
                    "observed_order_l2_exact_continuous": (
                        ""
                        if row["observed_order_l2_exact_continuous"] is None
                        else f"{float(row['observed_order_l2_exact_continuous']):.6f}"
                    ),
                    "observed_order_linf_exact_continuous": (
                        ""
                        if row["observed_order_linf_exact_continuous"] is None
                        else f"{float(row['observed_order_linf_exact_continuous']):.6f}"
                    ),
                    "observed_order_l2_exact_semidiscrete": (
                        ""
                        if row["observed_order_l2_exact_semidiscrete"] is None
                        else f"{float(row['observed_order_l2_exact_semidiscrete']):.6f}"
                    ),
                    "observed_order_linf_exact_semidiscrete": (
                        ""
                        if row["observed_order_linf_exact_semidiscrete"] is None
                        else f"{float(row['observed_order_linf_exact_semidiscrete']):.6f}"
                    ),
                    "stable_cfl": str(bool(row["stable_cfl"])),
                    "cfl_limit": f"{float(row['cfl_limit']):.10f}",
                    "status": str(row["status"]),
                }
            )

    print(f"[heat-exact] config={resolved_config}")
    print(f"[heat-exact] total_time={target_total_time:.6f}")
    print(f"[heat-exact] amplitude={amplitude:.6f} mode_x={mode_x} mode_y={mode_y}")
    print(
        "[heat-exact] note=use semidiscrete columns for clean temporal order; "
        "continuous columns include fixed spatial-discretization bias"
    )
    for row in rows_sorted:
        p_l2_semi = row.get("observed_order_l2_exact_semidiscrete")
        p_linf_semi = row.get("observed_order_linf_exact_semidiscrete")
        order_suffix = ""
        if p_l2_semi is not None and p_linf_semi is not None:
            order_suffix = (
                f" p_l2_semi={float(p_l2_semi):.4f} p_linf_semi={float(p_linf_semi):.4f}"
            )
        print(
            f"[heat-exact:{row['case']}] dt={float(row['dt']):.6f} steps={int(row['steps'])} "
            f"l2_cont={float(row['error_l2_exact_continuous']):.6e} "
            f"l2_semi={float(row['error_l2_exact_semidiscrete']):.6e} "
            f"linf_cont={float(row['error_linf_exact_continuous']):.6e} "
            f"linf_semi={float(row['error_linf_exact_semidiscrete']):.6e} "
            f"status={row['status']}{order_suffix}"
        )
    print(f"[heat-exact] report_file={output_path.resolve()}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run heat periodic time-step convergence against analytic exact solution."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("ForgeFlowApps/heat_periodic/config/run.json"),
        help="Simulation config path.",
    )
    parser.add_argument(
        "--dt-values",
        type=float,
        nargs="+",
        default=[0.0002, 0.0001, 0.00005],
        help="Time steps to evaluate (coarse -> fine).",
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
        default=Path("ForgeFlowApps/heat_periodic/output/exact_convergence_report.csv"),
        help="Output CSV path.",
    )
    args = parser.parse_args()

    run_exact_convergence_study(
        config_path=args.config,
        dt_values=list(args.dt_values),
        total_time=args.total_time,
        output_csv=args.output,
        amplitude=float(args.amplitude),
        mode_x=int(args.mode_x),
        mode_y=int(args.mode_y),
    )


if __name__ == "__main__":
    main()
