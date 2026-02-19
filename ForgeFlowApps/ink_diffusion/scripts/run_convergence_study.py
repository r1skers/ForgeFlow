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


def _run_case(
    records: list[dict[str, Any]],
    adapter_cls: type,
    model_cls: type,
    base_simulation: dict[str, Any],
    dt: float,
    total_time: float,
    label: str,
) -> dict[str, Any]:
    steps = int(round(total_time / dt))
    if steps <= 0:
        raise ValueError("resolved steps must be > 0")
    resolved_total_time = float(steps * dt)

    simulation = dict(base_simulation)
    simulation["dt"] = float(dt)
    simulation["steps"] = steps

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

    ny = len(final_grid)
    nx = len(final_grid[0]) if ny > 0 else 0

    return {
        "case": label,
        "dt": float(dt),
        "steps": steps,
        "total_time": resolved_total_time,
        "nx": nx,
        "ny": ny,
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


def _compute_observed_order(
    error_high: float,
    error_low: float,
    dt_high: float,
    dt_low: float,
) -> float | None:
    if error_high <= 0.0 or error_low <= 0.0:
        return None
    ratio_error = error_high / error_low
    ratio_dt = dt_high / dt_low
    if ratio_error <= 0.0 or ratio_dt <= 1.0:
        return None
    return float(math.log(ratio_error) / math.log(ratio_dt))


def _compute_richardson_order(
    error_coarse_mid: float,
    error_mid_fine: float,
    dt_coarse: float,
    dt_mid: float,
    dt_fine: float,
    ratio_tol: float = 1e-6,
) -> float | None:
    if error_coarse_mid <= 0.0 or error_mid_fine <= 0.0:
        return None

    ratio1 = dt_coarse / dt_mid
    ratio2 = dt_mid / dt_fine
    if ratio1 <= 1.0 or ratio2 <= 1.0:
        return None

    # Richardson-style order is valid only when refinement ratio is consistent.
    if abs(ratio1 - ratio2) > (ratio_tol * max(abs(ratio1), abs(ratio2))):
        return None

    ratio_error = error_coarse_mid / error_mid_fine
    if ratio_error <= 0.0:
        return None
    return float(math.log(ratio_error) / math.log(ratio1))


def run_convergence_study(
    config_path: Path,
    output_csv: Path,
    dt_values: list[float],
    total_time: float | None,
) -> None:
    resolved_config = config_path if config_path.is_absolute() else PROJECT_ROOT / config_path
    runtime = load_runtime_config(resolved_config, PROJECT_ROOT)
    if runtime.mode != "simulation":
        raise ValueError("convergence study requires a simulation mode config")
    if runtime.paths.initial_csv is None:
        raise ValueError("simulation config must provide paths.initial_csv")

    adapter_cls = _resolve_adapter_class(runtime)
    model_cls = _resolve_model_class(runtime)

    records, _ = read_csv_records(runtime.paths.initial_csv)
    if not records:
        raise ValueError("initial_csv contains no valid rows")

    dt_list = sorted({float(dt) for dt in dt_values if float(dt) > 0.0}, reverse=True)
    if len(dt_list) < 2:
        raise ValueError("at least two positive dt values are required")

    simulation_total_time = float(runtime.simulation["steps"]) * float(runtime.simulation["dt"])
    target_total_time = simulation_total_time if total_time is None else float(total_time)
    if target_total_time <= 0.0:
        raise ValueError("total_time must be > 0")

    cases: list[dict[str, Any]] = []
    for dt in dt_list:
        label = f"dt_{dt:g}"
        case = _run_case(
            records=records,
            adapter_cls=adapter_cls,
            model_cls=model_cls,
            base_simulation=runtime.simulation,
            dt=dt,
            total_time=target_total_time,
            label=label,
        )
        cases.append(case)

    # Use the smallest dt case as reference to estimate self-convergence.
    reference_case = min(cases, key=lambda row: float(row["dt"]))
    reference_grid = reference_case["final_grid"]
    reference_label = str(reference_case["case"])

    for case in cases:
        if case is reference_case:
            continue
        rmse, linf = _compute_grid_errors(case["final_grid"], reference_grid)
        case["error_l2_vs_ref"] = rmse
        case["error_linf_vs_ref"] = linf

    # Estimate order p from successive refined cases.
    cases_by_dt = sorted(cases, key=lambda row: float(row["dt"]), reverse=True)
    for idx in range(len(cases_by_dt) - 1):
        high = cases_by_dt[idx]
        low = cases_by_dt[idx + 1]
        low["observed_order_l2"] = _compute_observed_order(
            error_high=float(high["error_l2_vs_ref"]),
            error_low=float(low["error_l2_vs_ref"]),
            dt_high=float(high["dt"]),
            dt_low=float(low["dt"]),
        )
        low["observed_order_linf"] = _compute_observed_order(
            error_high=float(high["error_linf_vs_ref"]),
            error_low=float(low["error_linf_vs_ref"]),
            dt_high=float(high["dt"]),
            dt_low=float(low["dt"]),
        )

    # Richardson/triplet estimate: (dt, dt/r, dt/r^2) -> order attached to middle dt row.
    for idx in range(len(cases_by_dt) - 2):
        coarse = cases_by_dt[idx]
        mid = cases_by_dt[idx + 1]
        fine = cases_by_dt[idx + 2]

        error_l2_coarse_mid, error_linf_coarse_mid = _compute_grid_errors(
            coarse["final_grid"], mid["final_grid"]
        )
        error_l2_mid_fine, error_linf_mid_fine = _compute_grid_errors(
            mid["final_grid"], fine["final_grid"]
        )

        mid["observed_order_l2_richardson"] = _compute_richardson_order(
            error_coarse_mid=error_l2_coarse_mid,
            error_mid_fine=error_l2_mid_fine,
            dt_coarse=float(coarse["dt"]),
            dt_mid=float(mid["dt"]),
            dt_fine=float(fine["dt"]),
        )
        mid["observed_order_linf_richardson"] = _compute_richardson_order(
            error_coarse_mid=error_linf_coarse_mid,
            error_mid_fine=error_linf_mid_fine,
            dt_coarse=float(coarse["dt"]),
            dt_mid=float(mid["dt"]),
            dt_fine=float(fine["dt"]),
        )

    output_path = output_csv if output_csv.is_absolute() else PROJECT_ROOT / output_csv
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as out_file:
        writer = csv.DictWriter(
            out_file,
            fieldnames=[
                "case",
                "dt",
                "steps",
                "total_time",
                "nx",
                "ny",
                "stable_cfl",
                "cfl_limit",
                "mass_initial",
                "mass_final",
                "mass_delta_abs",
                "mass_tolerance",
                "mass_check",
                "status",
                "error_l2_vs_ref",
                "error_linf_vs_ref",
                "observed_order_l2",
                "observed_order_linf",
                "observed_order_l2_richardson",
                "observed_order_linf_richardson",
                "reference_case",
            ],
        )
        writer.writeheader()
        for case in cases_by_dt:
            writer.writerow(
                {
                    "case": case["case"],
                    "dt": f"{float(case['dt']):.8f}",
                    "steps": int(case["steps"]),
                    "total_time": f"{float(case['total_time']):.8f}",
                    "nx": int(case["nx"]),
                    "ny": int(case["ny"]),
                    "stable_cfl": str(bool(case["stable_cfl"])),
                    "cfl_limit": f"{float(case['cfl_limit']):.8f}",
                    "mass_initial": f"{float(case['mass_initial']):.12f}",
                    "mass_final": f"{float(case['mass_final']):.12f}",
                    "mass_delta_abs": f"{float(case['mass_delta_abs']):.12f}",
                    "mass_tolerance": f"{float(case['mass_tolerance']):.12f}",
                    "mass_check": str(case["mass_check"]),
                    "status": str(case["status"]),
                    "error_l2_vs_ref": f"{float(case['error_l2_vs_ref']):.12f}",
                    "error_linf_vs_ref": f"{float(case['error_linf_vs_ref']):.12f}",
                    "observed_order_l2": (
                        ""
                        if case["observed_order_l2"] is None
                        else f"{float(case['observed_order_l2']):.6f}"
                    ),
                    "observed_order_linf": (
                        ""
                        if case["observed_order_linf"] is None
                        else f"{float(case['observed_order_linf']):.6f}"
                    ),
                    "observed_order_l2_richardson": (
                        ""
                        if case.get("observed_order_l2_richardson") is None
                        else f"{float(case['observed_order_l2_richardson']):.6f}"
                    ),
                    "observed_order_linf_richardson": (
                        ""
                        if case.get("observed_order_linf_richardson") is None
                        else f"{float(case['observed_order_linf_richardson']):.6f}"
                    ),
                    "reference_case": reference_label,
                }
            )

    print(f"[convergence] config={resolved_config}")
    print(f"[convergence] total_time={target_total_time:.6f}")
    print(f"[convergence] reference_case={reference_label}")
    for case in cases_by_dt:
        rich_l2 = case.get("observed_order_l2_richardson")
        rich_linf = case.get("observed_order_linf_richardson")
        richardson_text = ""
        if rich_l2 is not None and rich_linf is not None:
            richardson_text = f" p_rich_l2={float(rich_l2):.4f} p_rich_linf={float(rich_linf):.4f}"
        print(
            f"[convergence:{case['case']}] dt={float(case['dt']):.6f} "
            f"steps={int(case['steps'])} l2={float(case['error_l2_vs_ref']):.8f} "
            f"linf={float(case['error_linf_vs_ref']):.8f} "
            f"mass_delta={float(case['mass_delta_abs']):.8e} status={case['status']}{richardson_text}"
        )
    print(f"[convergence] report_file={output_path.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a time-step refinement convergence study for the ink diffusion app. "
            "Each case simulates to the same physical end time and compares final grid "
            "errors against the finest dt case. When dt values form triplets "
            "(dt, dt/r, dt/r^2), Richardson-style observed order is also reported."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("ForgeFlowApps/ink_diffusion/config/run.json"),
        help="Simulation config path.",
    )
    parser.add_argument(
        "--dt-values",
        type=float,
        nargs="+",
        default=[0.04, 0.02, 0.01],
        help="Time steps to evaluate (coarse -> fine). Triplets enable Richardson order.",
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
        default=Path("ForgeFlowApps/ink_diffusion/output/convergence_report.csv"),
        help="Convergence report CSV output path.",
    )
    args = parser.parse_args()

    run_convergence_study(
        config_path=args.config,
        output_csv=args.output,
        dt_values=list(args.dt_values),
        total_time=args.total_time,
    )


if __name__ == "__main__":
    main()
