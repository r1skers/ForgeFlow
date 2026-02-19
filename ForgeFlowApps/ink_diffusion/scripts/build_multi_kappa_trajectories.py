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


def _write_trajectory(trajectory_csv: Path, states: list[dict[str, Any]], kappa: float) -> int:
    trajectory_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with trajectory_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["kappa", "step", "x", "y", "h"])
        for state in states:
            step = int(state["step"])
            grid = state["grid"]
            if not isinstance(grid, list):
                raise ValueError("simulation state must include 'grid' as a 2D list")
            for y_idx, row in enumerate(grid):
                if not isinstance(row, list):
                    raise ValueError("simulation grid rows must be lists")
                for x_idx, cell in enumerate(row):
                    writer.writerow([f"{kappa:.12f}", step, x_idx, y_idx, f"{float(cell):.12f}"])
                    rows += 1
    return rows


def _normalize_kappa_tag(kappa: float) -> str:
    return f"{kappa:g}".replace(".", "p")


def build_multi_kappa_trajectories(
    config_path: Path,
    out_dir: Path,
    kappas: list[float],
    cfl_safety: float,
    total_time: float | None,
) -> Path:
    if cfl_safety <= 0.0 or cfl_safety > 1.0:
        raise ValueError("cfl_safety must be in (0, 1]")

    resolved_config = config_path if config_path.is_absolute() else PROJECT_ROOT / config_path
    runtime = load_runtime_config(resolved_config, PROJECT_ROOT)
    if runtime.mode != "simulation":
        raise ValueError("multi-kappa generation requires simulation mode config")
    if runtime.paths.initial_csv is None:
        raise ValueError("simulation config must provide paths.initial_csv")

    adapter_cls = _resolve_adapter_class(runtime)
    model_cls = _resolve_model_class(runtime)

    records, csv_stats = read_csv_records(runtime.paths.initial_csv)
    if not records:
        raise ValueError("initial_csv contains no valid rows")

    out_path = out_dir if out_dir.is_absolute() else PROJECT_ROOT / out_dir
    out_path.mkdir(parents=True, exist_ok=True)
    manifest_csv = out_path / "manifest.csv"

    unique_kappas = sorted({float(k) for k in kappas if float(k) > 0.0})
    if not unique_kappas:
        raise ValueError("at least one positive kappa is required")

    base_total_time = float(runtime.simulation["steps"]) * float(runtime.simulation["dt"])
    target_total_time = base_total_time if total_time is None else float(total_time)
    if target_total_time <= 0.0:
        raise ValueError("total_time must be > 0")

    dx = float(runtime.simulation["dx"])
    dy = float(runtime.simulation["dy"])
    boundary = str(runtime.simulation["boundary"])
    mass_tolerance = float(runtime.simulation["mass_tolerance"])

    rows: list[dict[str, Any]] = []
    for kappa in unique_kappas:
        cfl_limit = 1.0 / (2.0 * kappa * ((1.0 / (dx * dx)) + (1.0 / (dy * dy))))
        dt_target = cfl_safety * cfl_limit
        steps = int(math.ceil(target_total_time / dt_target))
        if steps <= 0:
            raise ValueError("resolved steps must be > 0")
        dt = target_total_time / float(steps)

        simulation = dict(runtime.simulation)
        simulation["kappa"] = float(kappa)
        simulation["dt"] = float(dt)
        simulation["steps"] = int(steps)

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
        stable_cfl = bool(states[-1].get("stable_cfl", True))
        should_check_mass = boundary in {"neumann", "periodic"}
        mass_ok = (mass_delta_abs <= mass_tolerance) if should_check_mass else True
        status = "PASS" if stable_cfl and mass_ok else "FAIL"

        tag = _normalize_kappa_tag(kappa)
        trajectory_csv = out_path / f"trajectory_kappa_{tag}.csv"
        row_count = _write_trajectory(trajectory_csv, states, kappa)

        rows.append(
            {
                "kappa": f"{kappa:.12f}",
                "dt": f"{dt:.12f}",
                "steps": str(steps),
                "total_time": f"{target_total_time:.12f}",
                "dx": f"{dx:.12f}",
                "dy": f"{dy:.12f}",
                "cfl_limit": f"{cfl_limit:.12f}",
                "stable_cfl": str(stable_cfl),
                "mass_initial": f"{mass_initial:.12f}",
                "mass_final": f"{mass_final:.12f}",
                "mass_delta_abs": f"{mass_delta_abs:.12f}",
                "mass_tolerance": f"{mass_tolerance:.12f}",
                "mass_check": "checked" if should_check_mass else "skipped",
                "status": status,
                "valid_initial_rows": str(int(adapter_stats["valid_states"])),
                "trajectory_rows": str(row_count),
                "trajectory_csv": str(trajectory_csv.resolve()),
            }
        )

        print(
            f"[multi-kappa:{kappa:g}] dt={dt:.6f} steps={steps} "
            f"mass_delta={mass_delta_abs:.8e} status={status} rows={row_count}"
        )

    with manifest_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "kappa",
                "dt",
                "steps",
                "total_time",
                "dx",
                "dy",
                "cfl_limit",
                "stable_cfl",
                "mass_initial",
                "mass_final",
                "mass_delta_abs",
                "mass_tolerance",
                "mass_check",
                "status",
                "valid_initial_rows",
                "trajectory_rows",
                "trajectory_csv",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[multi-kappa] config={resolved_config}")
    print(f"[multi-kappa] input_valid_rows={csv_stats['valid_rows']}")
    print(f"[multi-kappa] kappas={','.join(f'{k:g}' for k in unique_kappas)}")
    print(f"[multi-kappa] manifest={manifest_csv.resolve()}")
    return manifest_csv


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run ink diffusion simulation for multiple kappa values and export one trajectory "
            "CSV per kappa plus a manifest file."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("ForgeFlowApps/ink_diffusion/config/run.json"),
        help="Simulation config path.",
    )
    parser.add_argument(
        "--kappas",
        type=float,
        nargs="+",
        default=[0.02, 0.05, 0.08, 0.10],
        help="Diffusion coefficients to simulate.",
    )
    parser.add_argument(
        "--cfl-safety",
        type=float,
        default=0.8,
        help="Per-kappa dt is selected by dt = cfl_safety * cfl_limit before step rounding.",
    )
    parser.add_argument(
        "--total-time",
        type=float,
        default=None,
        help="Physical horizon. Defaults to steps*dt from --config.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("ForgeFlowApps/ink_diffusion/output/multi_kappa"),
        help="Directory to write trajectory CSV files and manifest.csv.",
    )
    args = parser.parse_args()

    build_multi_kappa_trajectories(
        config_path=args.config,
        out_dir=args.out_dir,
        kappas=list(args.kappas),
        cfl_safety=float(args.cfl_safety),
        total_time=args.total_time,
    )


if __name__ == "__main__":
    main()
