from __future__ import annotations

import argparse
import csv
import copy
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ForgeFlowApps.heat1d_realistic_predict.scripts.build_dataset import load_config, simulate_truth


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run temporal convergence study for heat1d_realistic_predict dynamics."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("ForgeFlowApps/heat1d_realistic_predict/config/generate.json"),
        help="Path to generation config JSON.",
    )
    parser.add_argument(
        "--dt-values",
        type=float,
        nargs="+",
        default=[0.24, 0.12, 0.06],
        help="Time-step values to test (coarse -> fine).",
    )
    parser.add_argument(
        "--dt-ref",
        type=float,
        default=0.015,
        help="Reference fine time step used as pseudo-truth.",
    )
    parser.add_argument(
        "--total-time",
        type=float,
        default=None,
        help="Physical simulation time. Defaults to config steps*dt.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ForgeFlowApps/heat1d_realistic_predict/output/convergence_report.csv"),
        help="Output CSV report path.",
    )
    return parser


def _resolve_path(project_root: Path, raw_path: Path) -> Path:
    return raw_path if raw_path.is_absolute() else project_root / raw_path


def _prepare_deterministic_config(base_cfg: dict[str, object]) -> dict[str, object]:
    cfg = copy.deepcopy(base_cfg)
    dynamics = cfg.get("dynamics")
    if not isinstance(dynamics, dict):
        raise ValueError("config.dynamics must be an object")
    dynamics["process_noise_std"] = 0.0
    return cfg


def _compute_errors(u_num: np.ndarray, u_ref: np.ndarray) -> tuple[float, float]:
    if u_num.shape != u_ref.shape:
        raise ValueError("shape mismatch while computing convergence errors")
    diff = u_num - u_ref
    l2 = float(np.sqrt(np.mean(diff * diff)))
    linf = float(np.max(np.abs(diff)))
    return l2, linf


def _compute_order(err_coarse: float, err_fine: float, dt_coarse: float, dt_fine: float) -> float | None:
    if err_coarse <= 0.0 or err_fine <= 0.0:
        return None
    ratio_e = err_coarse / err_fine
    ratio_dt = dt_coarse / dt_fine
    if ratio_e <= 0.0 or ratio_dt <= 1.0:
        return None
    return float(math.log(ratio_e) / math.log(ratio_dt))


def run_convergence(
    config_path: Path,
    dt_values: list[float],
    dt_ref: float,
    total_time: float | None,
    output_path: Path,
) -> Path:
    resolved_config = _resolve_path(PROJECT_ROOT, config_path)
    cfg_raw = load_config(resolved_config)
    cfg = _prepare_deterministic_config(cfg_raw)

    time_cfg = cfg.get("time")
    if not isinstance(time_cfg, dict):
        raise ValueError("config.time must be an object")
    base_total_time = float(time_cfg["steps"]) * float(time_cfg["dt"])
    target_total_time = base_total_time if total_time is None else float(total_time)
    if target_total_time <= 0.0:
        raise ValueError("total_time must be > 0")

    dt_list = sorted({float(dt) for dt in dt_values if float(dt) > 0.0}, reverse=True)
    if len(dt_list) < 2:
        raise ValueError("need at least two positive dt values")
    if dt_ref <= 0.0:
        raise ValueError("dt_ref must be > 0")

    cfg_ref = copy.deepcopy(cfg)
    ref_steps = int(round(target_total_time / dt_ref))
    if ref_steps <= 0:
        raise ValueError("resolved ref steps must be > 0")
    cfg_ref["time"]["dt"] = dt_ref
    cfg_ref["time"]["steps"] = ref_steps

    _, _, u_ref, stability_ref = simulate_truth(cfg_ref, np.random.default_rng(0))
    u_ref_final = u_ref[-1, :]

    rows: list[dict[str, object]] = []
    for dt in dt_list:
        case_cfg = copy.deepcopy(cfg)
        steps = int(round(target_total_time / dt))
        if steps <= 0:
            raise ValueError("resolved steps must be > 0")
        resolved_t = steps * dt
        case_cfg["time"]["dt"] = dt
        case_cfg["time"]["steps"] = steps

        _, _, u_num, stability = simulate_truth(case_cfg, np.random.default_rng(0))
        u_num_final = u_num[-1, :]
        l2, linf = _compute_errors(u_num_final, u_ref_final)
        rows.append(
            {
                "case": f"dt_{dt:g}",
                "dt": dt,
                "steps": steps,
                "total_time": resolved_t,
                "error_l2": l2,
                "error_linf": linf,
                "observed_order_l2": None,
                "observed_order_linf": None,
                "cfl_limit_dt": float(stability["cfl_limit_dt"]),
                "stable_cfl": bool(stability["stable_cfl"]),
                "status": "PASS" if bool(stability["stable_cfl"]) else "FAIL",
            }
        )

    rows_sorted = sorted(rows, key=lambda item: float(item["dt"]), reverse=True)
    for idx in range(len(rows_sorted) - 1):
        coarse = rows_sorted[idx]
        fine = rows_sorted[idx + 1]
        fine["observed_order_l2"] = _compute_order(
            err_coarse=float(coarse["error_l2"]),
            err_fine=float(fine["error_l2"]),
            dt_coarse=float(coarse["dt"]),
            dt_fine=float(fine["dt"]),
        )
        fine["observed_order_linf"] = _compute_order(
            err_coarse=float(coarse["error_linf"]),
            err_fine=float(fine["error_linf"]),
            dt_coarse=float(coarse["dt"]),
            dt_fine=float(fine["dt"]),
        )

    resolved_out = _resolve_path(PROJECT_ROOT, output_path)
    resolved_out.parent.mkdir(parents=True, exist_ok=True)
    with resolved_out.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "case",
                "dt",
                "steps",
                "total_time",
                "error_l2",
                "error_linf",
                "observed_order_l2",
                "observed_order_linf",
                "cfl_limit_dt",
                "stable_cfl",
                "status",
                "reference_dt",
                "reference_steps",
                "reference_stable_cfl",
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
                    "error_l2": f"{float(row['error_l2']):.12e}",
                    "error_linf": f"{float(row['error_linf']):.12e}",
                    "observed_order_l2": (
                        ""
                        if row["observed_order_l2"] is None
                        else f"{float(row['observed_order_l2']):.6f}"
                    ),
                    "observed_order_linf": (
                        ""
                        if row["observed_order_linf"] is None
                        else f"{float(row['observed_order_linf']):.6f}"
                    ),
                    "cfl_limit_dt": f"{float(row['cfl_limit_dt']):.10f}",
                    "stable_cfl": str(bool(row["stable_cfl"])),
                    "status": str(row["status"]),
                    "reference_dt": f"{dt_ref:.10f}",
                    "reference_steps": ref_steps,
                    "reference_stable_cfl": str(bool(stability_ref["stable_cfl"])),
                }
            )

    print(f"[convergence] config={resolved_config}")
    print(f"[convergence] total_time={target_total_time:.6f}")
    print(f"[convergence] dt_ref={dt_ref:.6f} steps_ref={ref_steps}")
    for row in rows_sorted:
        p_l2 = row["observed_order_l2"]
        p_linf = row["observed_order_linf"]
        suffix = ""
        if p_l2 is not None and p_linf is not None:
            suffix = f" p_l2={float(p_l2):.4f} p_linf={float(p_linf):.4f}"
        print(
            f"[convergence:{row['case']}] dt={float(row['dt']):.6f} steps={int(row['steps'])} "
            f"l2={float(row['error_l2']):.8f} linf={float(row['error_linf']):.8f} "
            f"status={row['status']}{suffix}"
        )
    print(f"[convergence] report_file={resolved_out}")
    return resolved_out


def main() -> None:
    args = build_parser().parse_args()
    run_convergence(
        config_path=args.config,
        dt_values=list(args.dt_values),
        dt_ref=float(args.dt_ref),
        total_time=args.total_time,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()

