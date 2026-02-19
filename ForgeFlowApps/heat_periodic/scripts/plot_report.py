import argparse
import csv
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _scan_trajectory_shape(trajectory_csv: Path) -> tuple[int, int, int]:
    max_step = -1
    max_x = -1
    max_y = -1
    with trajectory_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            step = int(row["step"])
            x_idx = int(row["x"])
            y_idx = int(row["y"])
            max_step = max(max_step, step)
            max_x = max(max_x, x_idx)
            max_y = max(max_y, y_idx)

    if max_step < 0:
        raise ValueError("trajectory.csv has no data rows")
    return max_step, max_x + 1, max_y + 1


def _load_simulation_stats_and_snapshots(
    trajectory_csv: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    max_step, nx, ny = _scan_trajectory_shape(trajectory_csv)
    mid_step = max_step // 2
    snapshot_steps = [0, mid_step, max_step]

    mass = np.zeros(max_step + 1, dtype=float)
    max_abs = np.zeros(max_step + 1, dtype=float)
    min_val = np.full(max_step + 1, np.inf, dtype=float)
    snapshots = {step: np.zeros((ny, nx), dtype=float) for step in snapshot_steps}

    with trajectory_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            step = int(row["step"])
            x_idx = int(row["x"])
            y_idx = int(row["y"])
            h_val = float(row["h"])

            mass[step] += h_val
            max_abs[step] = max(max_abs[step], abs(h_val))
            min_val[step] = min(min_val[step], h_val)
            if step in snapshots:
                snapshots[step][y_idx, x_idx] = h_val

    return mass, max_abs, min_val, snapshots


def _read_eval_row(eval_csv: Path) -> dict[str, str]:
    with eval_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    return rows[0]


def _plot_simulation_report(
    trajectory_csv: Path,
    simulation_eval_csv: Path,
    output_png: Path,
    amplitude: float,
) -> None:
    mass, max_abs, min_val, snapshots = _load_simulation_stats_and_snapshots(trajectory_csv)
    steps = np.arange(len(mass))
    eval_row = _read_eval_row(simulation_eval_csv)

    dt = float(eval_row.get("dt", "1.0"))
    kappa = float(eval_row.get("kappa", "0.05"))
    exact_peak = amplitude * np.exp(-8.0 * math.pi * math.pi * kappa * steps * dt)

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.9])

    ordered_steps = sorted(snapshots.keys())
    # Use one shared color scale so amplitude decay is visible across snapshots.
    shared_abs_max = max(float(np.max(np.abs(snapshots[step]))) for step in ordered_steps)
    if shared_abs_max <= 0.0:
        shared_abs_max = 1.0

    for i, step in enumerate(ordered_steps):
        ax = fig.add_subplot(gs[0, i])
        step_abs_max = float(np.max(np.abs(snapshots[step])))
        im = ax.imshow(
            snapshots[step],
            cmap="coolwarm",
            origin="lower",
            vmin=-shared_abs_max,
            vmax=shared_abs_max,
        )
        ax.set_title(f"u @ step={step}\nmax|u|={step_abs_max:.3f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax_diag = fig.add_subplot(gs[1, :2])
    ax_diag.plot(steps, max_abs, label="max(|u|) numeric", color="#1f77b4")
    ax_diag.plot(steps, exact_peak, label="max(|u|) exact", color="#d62728", linestyle="--")
    ax_diag.plot(steps, np.abs(mass), label="|mass|", color="#2ca02c")
    ax_diag.plot(steps, np.abs(min_val), label="|min(u)|", color="#9467bd", alpha=0.6)
    ax_diag.set_title("Heat Simulation Diagnostics")
    ax_diag.set_xlabel("step")
    ax_diag.set_ylabel("value")
    ax_diag.grid(alpha=0.2)
    ax_diag.legend()

    ax_text = fig.add_subplot(gs[1, 2])
    ax_text.axis("off")
    status = eval_row.get("status", "-")
    cfl_limit = eval_row.get("cfl_limit", "-")
    stable_cfl = eval_row.get("stable_cfl", "-")
    mass_delta = eval_row.get("mass_delta_abs", "-")
    boundary = eval_row.get("boundary", "-")
    text = (
        "Simulation Summary\n"
        f"status: {status}\n"
        f"boundary: {boundary}\n"
        f"stable_cfl: {stable_cfl}\n"
        f"cfl_limit: {cfl_limit}\n"
        f"mass_delta_abs: {mass_delta}\n"
        f"steps: {len(steps) - 1}"
    )
    ax_text.text(0.0, 1.0, text, va="top", ha="left", fontsize=11)

    fig.suptitle("Heat Periodic Simulation Report", fontsize=14)
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=150)
    plt.close(fig)


def _load_surrogate_pairs(infer_csv: Path, predictions_csv: Path) -> tuple[np.ndarray, np.ndarray]:
    y_true: list[float] = []
    with infer_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            y_true.append(float(row["h_t1"]))

    y_pred: list[float] = []
    with predictions_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            y_pred.append(float(row["y_pred"]))

    if len(y_true) != len(y_pred):
        raise ValueError(f"infer/prediction length mismatch: {len(y_true)} vs {len(y_pred)}")
    return np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)


def _compute_basic_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    errors = y_true - y_pred
    mae = float(np.mean(np.abs(errors)))
    rmse = float(math.sqrt(float(np.mean(np.square(errors)))))
    maxae = float(np.max(np.abs(errors)))
    return {"mae": mae, "rmse": rmse, "maxae": maxae}


def _plot_surrogate_report(
    infer_csv: Path,
    predictions_csv: Path,
    surrogate_eval_csv: Path,
    output_png: Path,
    scatter_points: int,
) -> None:
    y_true, y_pred = _load_surrogate_pairs(infer_csv, predictions_csv)
    metrics = _compute_basic_metrics(y_true, y_pred)
    eval_row = _read_eval_row(surrogate_eval_csv)
    residuals = y_true - y_pred

    n = len(y_true)
    sample_n = min(max(scatter_points, 1), n)
    indices = list(range(n))
    random.Random(42).shuffle(indices)
    pick = indices[:sample_n]
    sample_true = y_true[pick]
    sample_pred = y_pred[pick]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].scatter(sample_true, sample_pred, s=6, alpha=0.5)
    min_v = float(min(sample_true.min(), sample_pred.min()))
    max_v = float(max(sample_true.max(), sample_pred.max()))
    axes[0].plot([min_v, max_v], [min_v, max_v], "r--", linewidth=1)
    axes[0].set_title("y_true vs y_pred")
    axes[0].set_xlabel("y_true")
    axes[0].set_ylabel("y_pred")
    axes[0].grid(alpha=0.2)

    axes[1].hist(residuals, bins=60, color="#1f77b4", alpha=0.85)
    axes[1].set_title("Residual Histogram (y_true - y_pred)")
    axes[1].set_xlabel("residual")
    axes[1].set_ylabel("count")
    axes[1].grid(alpha=0.2)

    axes[2].axis("off")
    status = eval_row.get("status", "-")
    text = (
        "Surrogate Summary\n"
        f"status: {status}\n"
        f"samples: {n}\n"
        f"mae: {metrics['mae']:.6e}\n"
        f"rmse: {metrics['rmse']:.6e}\n"
        f"maxae: {metrics['maxae']:.6e}\n"
        f"val_mae(csv): {eval_row.get('val_mae', '-')}\n"
        f"val_rmse(csv): {eval_row.get('val_rmse', '-')}\n"
        f"val_maxae(csv): {eval_row.get('val_maxae', '-')}"
    )
    axes[2].text(0.0, 1.0, text, va="top", ha="left", fontsize=11)

    fig.suptitle("Heat Periodic Surrogate Report", fontsize=14)
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=150)
    plt.close(fig)


def _load_rollout_rows(rollout_steps_csv: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with rollout_steps_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "step": float(row["step"]),
                    "mae": float(row["mae"]),
                    "rmse": float(row["rmse"]),
                    "maxae": float(row["maxae"]),
                    "mass_delta_abs": float(row["mass_delta_abs"]),
                }
            )
    return rows


def _plot_rollout_report(
    rollout_steps_csv: Path,
    rollout_summary_csv: Path,
    output_png: Path,
) -> None:
    rows = _load_rollout_rows(rollout_steps_csv)
    if not rows:
        raise ValueError("rollout step report is empty")

    summary = _read_eval_row(rollout_summary_csv)
    steps = np.asarray([row["step"] for row in rows], dtype=float)
    mae = np.asarray([row["mae"] for row in rows], dtype=float)
    rmse = np.asarray([row["rmse"] for row in rows], dtype=float)
    maxae = np.asarray([row["maxae"] for row in rows], dtype=float)
    mass_delta = np.asarray([row["mass_delta_abs"] for row in rows], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].plot(steps, mae, label="MAE", color="#1f77b4")
    axes[0].plot(steps, rmse, label="RMSE", color="#ff7f0e")
    axes[0].plot(steps, maxae, label="MaxAE", color="#d62728")
    axes[0].set_title("Rollout Error vs Step")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("error")
    axes[0].grid(alpha=0.2)
    axes[0].legend()

    axes[1].plot(steps, mass_delta, color="#2ca02c")
    axes[1].set_title("Rollout Mass Drift")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("abs mass delta")
    axes[1].grid(alpha=0.2)

    axes[2].axis("off")
    text = (
        "Rollout Summary\n"
        f"status: {summary.get('status', '-')}\n"
        f"rollout_steps: {summary.get('rollout_steps', '-')}\n"
        f"mean_mae: {summary.get('mean_mae', '-')}\n"
        f"mean_rmse: {summary.get('mean_rmse', '-')}\n"
        f"max_maxae: {summary.get('max_maxae', '-')}\n"
        f"last_mae: {summary.get('last_mae', '-')}\n"
        f"last_rmse: {summary.get('last_rmse', '-')}\n"
        f"last_maxae: {summary.get('last_maxae', '-')}"
    )
    axes[2].text(0.0, 1.0, text, va="top", ha="left", fontsize=11)

    fig.suptitle("Heat Periodic Surrogate Rollout Report", fontsize=14)
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build visual reports for heat periodic pipeline.")
    parser.add_argument(
        "--trajectory",
        type=Path,
        default=Path("ForgeFlowApps/heat_periodic/output/trajectory.csv"),
    )
    parser.add_argument(
        "--simulation-eval",
        type=Path,
        default=Path("ForgeFlowApps/heat_periodic/output/eval_report.csv"),
    )
    parser.add_argument(
        "--surrogate-infer",
        type=Path,
        default=Path("ForgeFlowApps/heat_periodic/data/processed/surrogate_infer.csv"),
    )
    parser.add_argument(
        "--surrogate-predictions",
        type=Path,
        default=Path("ForgeFlowApps/heat_periodic/output/surrogate_predictions.csv"),
    )
    parser.add_argument(
        "--surrogate-eval",
        type=Path,
        default=Path("ForgeFlowApps/heat_periodic/output/surrogate_eval_report.csv"),
    )
    parser.add_argument(
        "--rollout-steps-csv",
        type=Path,
        default=Path("ForgeFlowApps/heat_periodic/output/surrogate_rollout_steps.csv"),
    )
    parser.add_argument(
        "--rollout-summary-csv",
        type=Path,
        default=Path("ForgeFlowApps/heat_periodic/output/surrogate_rollout_summary.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("ForgeFlowApps/heat_periodic/output/report"),
    )
    parser.add_argument(
        "--amplitude",
        type=float,
        default=1.0,
        help="Initial amplitude used for exact max(|u|) decay reference.",
    )
    parser.add_argument(
        "--scatter-points",
        type=int,
        default=3000,
        help="Max sampled points for y_true vs y_pred scatter.",
    )
    args = parser.parse_args()

    simulation_png = args.out_dir / "simulation_report.png"
    surrogate_png = args.out_dir / "surrogate_report.png"
    rollout_png = args.out_dir / "rollout_report.png"

    _plot_simulation_report(
        trajectory_csv=args.trajectory,
        simulation_eval_csv=args.simulation_eval,
        output_png=simulation_png,
        amplitude=float(args.amplitude),
    )
    _plot_surrogate_report(
        infer_csv=args.surrogate_infer,
        predictions_csv=args.surrogate_predictions,
        surrogate_eval_csv=args.surrogate_eval,
        output_png=surrogate_png,
        scatter_points=args.scatter_points,
    )

    if args.rollout_steps_csv.exists() and args.rollout_summary_csv.exists():
        _plot_rollout_report(
            rollout_steps_csv=args.rollout_steps_csv,
            rollout_summary_csv=args.rollout_summary_csv,
            output_png=rollout_png,
        )
        print(f"[heat-plot] rollout_report={rollout_png.resolve()}")
    else:
        print("[heat-plot] rollout_report=skipped (missing rollout CSV files)")

    print(f"[heat-plot] simulation_report={simulation_png.resolve()}")
    print(f"[heat-plot] surrogate_report={surrogate_png.resolve()}")


if __name__ == "__main__":
    main()
