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
    peak = np.full(max_step + 1, -np.inf, dtype=float)
    valley = np.full(max_step + 1, np.inf, dtype=float)
    snapshots = {step: np.zeros((ny, nx), dtype=float) for step in snapshot_steps}

    with trajectory_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            step = int(row["step"])
            x_idx = int(row["x"])
            y_idx = int(row["y"])
            h_val = float(row["h"])

            mass[step] += h_val
            peak[step] = max(peak[step], h_val)
            valley[step] = min(valley[step], h_val)
            if step in snapshots:
                snapshots[step][y_idx, x_idx] = h_val

    return mass, peak, valley, snapshots


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
) -> None:
    mass, peak, valley, snapshots = _load_simulation_stats_and_snapshots(trajectory_csv)
    steps = np.arange(len(mass))
    eval_row = _read_eval_row(simulation_eval_csv)

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.9])

    ordered_steps = sorted(snapshots.keys())
    for i, step in enumerate(ordered_steps):
        ax = fig.add_subplot(gs[0, i])
        im = ax.imshow(snapshots[step], cmap="Blues", origin="lower")
        ax.set_title(f"Concentration @ step={step}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax_mass = fig.add_subplot(gs[1, :2])
    ax_mass.plot(steps, mass, label="mass(t)", color="#1f77b4")
    ax_mass.plot(steps, peak, label="max(h)", color="#d62728")
    ax_mass.plot(steps, valley, label="min(h)", color="#2ca02c")
    ax_mass.set_title("Simulation Diagnostics")
    ax_mass.set_xlabel("step")
    ax_mass.set_ylabel("value")
    ax_mass.grid(alpha=0.2)
    ax_mass.legend()

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

    fig.suptitle("Ink Diffusion Simulation Report", fontsize=14)
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=150)
    plt.close(fig)


def _load_surrogate_pairs(
    infer_csv: Path, predictions_csv: Path
) -> tuple[np.ndarray, np.ndarray]:
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
        raise ValueError(
            f"infer/prediction length mismatch: {len(y_true)} vs {len(y_pred)}"
        )
    return np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)


def _compute_basic_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    errors = y_true - y_pred
    mae = float(np.mean(np.abs(errors)))
    rmse = float(math.sqrt(float(np.mean(np.square(errors)))))
    maxae = float(np.max(np.abs(errors)))
    return {
        "mae": mae,
        "rmse": rmse,
        "maxae": maxae,
    }


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

    fig.suptitle("Ink Diffusion Surrogate Report", fontsize=14)
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build visual reports for ink diffusion pipeline.")
    parser.add_argument(
        "--trajectory",
        type=Path,
        default=Path("ForgeFlowApps/ink_diffusion/output/trajectory.csv"),
    )
    parser.add_argument(
        "--simulation-eval",
        type=Path,
        default=Path("ForgeFlowApps/ink_diffusion/output/eval_report.csv"),
    )
    parser.add_argument(
        "--surrogate-infer",
        type=Path,
        default=Path("ForgeFlowApps/ink_diffusion/data/processed/surrogate_infer.csv"),
    )
    parser.add_argument(
        "--surrogate-predictions",
        type=Path,
        default=Path("ForgeFlowApps/ink_diffusion/output/surrogate_predictions.csv"),
    )
    parser.add_argument(
        "--surrogate-eval",
        type=Path,
        default=Path("ForgeFlowApps/ink_diffusion/output/surrogate_eval_report.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("ForgeFlowApps/ink_diffusion/output/report"),
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

    _plot_simulation_report(
        trajectory_csv=args.trajectory,
        simulation_eval_csv=args.simulation_eval,
        output_png=simulation_png,
    )
    _plot_surrogate_report(
        infer_csv=args.surrogate_infer,
        predictions_csv=args.surrogate_predictions,
        surrogate_eval_csv=args.surrogate_eval,
        output_png=surrogate_png,
        scatter_points=args.scatter_points,
    )

    print(f"[plot] simulation_report={simulation_png.resolve()}")
    print(f"[plot] surrogate_report={surrogate_png.resolve()}")


if __name__ == "__main__":
    main()
