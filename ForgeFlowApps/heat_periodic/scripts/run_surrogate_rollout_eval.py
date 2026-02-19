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


def _load_trajectory_grids(trajectory_csv: Path) -> list[list[list[float]]]:
    max_step = -1
    max_x = -1
    max_y = -1
    rows: list[dict[str, str]] = []

    with trajectory_csv.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows.append(row)
            max_step = max(max_step, int(row["step"]))
            max_x = max(max_x, int(row["x"]))
            max_y = max(max_y, int(row["y"]))

    if max_step < 1:
        raise ValueError("trajectory must contain at least two steps")

    nx = max_x + 1
    ny = max_y + 1
    grids = [[[0.0 for _ in range(nx)] for _ in range(ny)] for _ in range(max_step + 1)]
    counts = [0 for _ in range(max_step + 1)]
    for row in rows:
        step = int(row["step"])
        x_idx = int(row["x"])
        y_idx = int(row["y"])
        h_val = float(row["h"])
        grids[step][y_idx][x_idx] = h_val
        counts[step] += 1

    expected = nx * ny
    for step, count in enumerate(counts):
        if count != expected:
            raise ValueError(f"trajectory step {step} has {count} cells, expected {expected}")

    return grids


def _compute_mass(grid: list[list[float]]) -> float:
    return float(sum(sum(float(cell) for cell in row) for row in grid))


def _build_feature_matrix_from_grid(
    grid: list[list[float]], feature_names: tuple[str, ...]
) -> list[list[float]]:
    ny = len(grid)
    nx = len(grid[0])
    features: list[list[float]] = []

    for y_idx in range(ny):
        up_idx = (y_idx - 1) % ny
        down_idx = (y_idx + 1) % ny
        for x_idx in range(nx):
            left_idx = (x_idx - 1) % nx
            right_idx = (x_idx + 1) % nx

            value_map: dict[str, float] = {
                "h_t": float(grid[y_idx][x_idx]),
                "h_up": float(grid[up_idx][x_idx]),
                "h_down": float(grid[down_idx][x_idx]),
                "h_left": float(grid[y_idx][left_idx]),
                "h_right": float(grid[y_idx][right_idx]),
            }

            feature_row: list[float] = []
            for name in feature_names:
                if name not in value_map:
                    raise ValueError(f"unsupported feature name in rollout: {name}")
                feature_row.append(value_map[name])
            features.append(feature_row)

    return features


def _reshape_predictions(predictions: list[list[float]], ny: int, nx: int) -> list[list[float]]:
    if len(predictions) != ny * nx:
        raise ValueError("prediction row count does not match grid size")
    grid = [[0.0 for _ in range(nx)] for _ in range(ny)]
    idx = 0
    for y_idx in range(ny):
        for x_idx in range(nx):
            row = predictions[idx]
            if not row:
                raise ValueError("prediction target row is empty")
            grid[y_idx][x_idx] = float(row[0])
            idx += 1
    return grid


def _compute_grid_metrics(
    true_grid: list[list[float]], pred_grid: list[list[float]]
) -> tuple[float, float, float]:
    ny = len(true_grid)
    nx = len(true_grid[0])
    if len(pred_grid) != ny or len(pred_grid[0]) != nx:
        raise ValueError("true and pred grid shape mismatch")

    abs_sum = 0.0
    sq_sum = 0.0
    max_abs = 0.0
    count = nx * ny
    for y_idx in range(ny):
        for x_idx in range(nx):
            diff = float(true_grid[y_idx][x_idx]) - float(pred_grid[y_idx][x_idx])
            abs_diff = abs(diff)
            abs_sum += abs_diff
            sq_sum += diff * diff
            max_abs = max(max_abs, abs_diff)

    mae = abs_sum / float(count)
    rmse = math.sqrt(sq_sum / float(count))
    return mae, rmse, max_abs


def run_rollout_eval(
    config_path: Path,
    trajectory_csv: Path,
    rollout_steps: int | None,
    output_steps_csv: Path,
    output_summary_csv: Path,
) -> tuple[Path, Path]:
    resolved_config = config_path if config_path.is_absolute() else PROJECT_ROOT / config_path
    runtime = load_runtime_config(resolved_config, PROJECT_ROOT)
    if runtime.mode != "supervised":
        raise ValueError("rollout eval requires a supervised config")
    if runtime.paths.train_csv is None:
        raise ValueError("supervised config must include paths.train_csv")

    adapter_cls = _resolve_adapter_class(runtime)
    model_cls = _resolve_model_class(runtime)
    adapter = adapter_cls()
    feature_names = tuple(getattr(adapter, "feature_names", ()))
    if not feature_names:
        raise ValueError("adapter must expose feature_names")

    train_records, csv_stats = read_csv_records(runtime.paths.train_csv)
    states, adapter_stats = adapter.adapt_records(train_records)
    x_train, feature_stats = adapter.build_feature_matrix(states)
    y_train, target_stats = adapter.build_target_matrix(states)
    if not x_train or not y_train:
        raise ValueError("training set is empty after adapter processing")

    model = model_cls()
    model.fit(x_train, y_train)

    resolved_trajectory = trajectory_csv if trajectory_csv.is_absolute() else PROJECT_ROOT / trajectory_csv
    true_grids = _load_trajectory_grids(resolved_trajectory)
    max_steps = len(true_grids) - 1
    if rollout_steps is None:
        steps = max_steps
    else:
        if rollout_steps <= 0:
            raise ValueError("rollout_steps must be > 0")
        steps = min(int(rollout_steps), max_steps)

    pred_grid = [[float(cell) for cell in row] for row in true_grids[0]]
    ny = len(pred_grid)
    nx = len(pred_grid[0])
    step_rows: list[dict[str, Any]] = []

    for step in range(1, steps + 1):
        features = _build_feature_matrix_from_grid(pred_grid, feature_names)
        predictions = model.predict(features)
        pred_grid = _reshape_predictions(predictions, ny=ny, nx=nx)

        true_grid = true_grids[step]
        mae, rmse, maxae = _compute_grid_metrics(true_grid, pred_grid)
        true_mass = _compute_mass(true_grid)
        pred_mass = _compute_mass(pred_grid)
        step_rows.append(
            {
                "step": step,
                "mae": mae,
                "rmse": rmse,
                "maxae": maxae,
                "true_mass": true_mass,
                "pred_mass": pred_mass,
                "mass_delta_abs": abs(true_mass - pred_mass),
            }
        )

    mean_mae = float(sum(row["mae"] for row in step_rows) / len(step_rows))
    mean_rmse = float(sum(row["rmse"] for row in step_rows) / len(step_rows))
    max_maxae = float(max(row["maxae"] for row in step_rows))
    last_row = step_rows[-1]
    status = "PASS" if all(math.isfinite(row["rmse"]) for row in step_rows) else "FAIL"

    resolved_steps_csv = (
        output_steps_csv if output_steps_csv.is_absolute() else PROJECT_ROOT / output_steps_csv
    )
    resolved_summary_csv = (
        output_summary_csv if output_summary_csv.is_absolute() else PROJECT_ROOT / output_summary_csv
    )
    resolved_steps_csv.parent.mkdir(parents=True, exist_ok=True)
    resolved_summary_csv.parent.mkdir(parents=True, exist_ok=True)

    with resolved_steps_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "step",
                "mae",
                "rmse",
                "maxae",
                "true_mass",
                "pred_mass",
                "mass_delta_abs",
            ],
        )
        writer.writeheader()
        for row in step_rows:
            writer.writerow(
                {
                    "step": row["step"],
                    "mae": f"{float(row['mae']):.12f}",
                    "rmse": f"{float(row['rmse']):.12f}",
                    "maxae": f"{float(row['maxae']):.12f}",
                    "true_mass": f"{float(row['true_mass']):.12f}",
                    "pred_mass": f"{float(row['pred_mass']):.12f}",
                    "mass_delta_abs": f"{float(row['mass_delta_abs']):.12f}",
                }
            )

    with resolved_summary_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "task",
                "config",
                "trajectory_csv",
                "feature_names",
                "train_samples",
                "rollout_steps",
                "mean_mae",
                "mean_rmse",
                "max_maxae",
                "last_mae",
                "last_rmse",
                "last_maxae",
                "last_mass_delta_abs",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "task": runtime.task,
                "config": str(resolved_config),
                "trajectory_csv": str(resolved_trajectory),
                "feature_names": ",".join(feature_names),
                "train_samples": len(x_train),
                "rollout_steps": steps,
                "mean_mae": f"{mean_mae:.12f}",
                "mean_rmse": f"{mean_rmse:.12f}",
                "max_maxae": f"{max_maxae:.12f}",
                "last_mae": f"{float(last_row['mae']):.12f}",
                "last_rmse": f"{float(last_row['rmse']):.12f}",
                "last_maxae": f"{float(last_row['maxae']):.12f}",
                "last_mass_delta_abs": f"{float(last_row['mass_delta_abs']):.12f}",
                "status": status,
            }
        )

    print(f"[heat-rollout] config={resolved_config}")
    print(f"[heat-rollout] trajectory={resolved_trajectory}")
    print(f"[heat-rollout] train_rows={csv_stats['valid_rows']}")
    print(f"[heat-rollout] train_states={adapter_stats['valid_states']}")
    print(f"[heat-rollout] train_vectors={feature_stats['valid_vectors']}")
    print(f"[heat-rollout] target_vectors={target_stats['valid_vectors']}")
    print(f"[heat-rollout] feature_names={','.join(feature_names)}")
    print(f"[heat-rollout] steps={steps}")
    print(f"[heat-rollout] mean_mae={mean_mae:.8f}")
    print(f"[heat-rollout] mean_rmse={mean_rmse:.8f}")
    print(f"[heat-rollout] max_maxae={max_maxae:.8f}")
    print(f"[heat-rollout] last_mae={float(last_row['mae']):.8f}")
    print(f"[heat-rollout] last_rmse={float(last_row['rmse']):.8f}")
    print(f"[heat-rollout] last_maxae={float(last_row['maxae']):.8f}")
    print(f"[heat-rollout] status={status}")
    print(f"[heat-rollout] step_report={resolved_steps_csv.resolve()}")
    print(f"[heat-rollout] summary_report={resolved_summary_csv.resolve()}")
    return resolved_steps_csv, resolved_summary_csv


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train heat surrogate from config.train_csv and evaluate multi-step rollout "
            "against heat trajectory truth."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("ForgeFlowApps/heat_periodic/config/surrogate_run.json"),
        help="Supervised config path.",
    )
    parser.add_argument(
        "--trajectory",
        type=Path,
        default=Path("ForgeFlowApps/heat_periodic/output/trajectory.csv"),
        help="Trajectory CSV path (step,x,y,h).",
    )
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=50,
        help="Rollout step count (clipped by trajectory length).",
    )
    parser.add_argument(
        "--steps-out",
        type=Path,
        default=Path("ForgeFlowApps/heat_periodic/output/surrogate_rollout_steps.csv"),
        help="Per-step rollout CSV output.",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=Path("ForgeFlowApps/heat_periodic/output/surrogate_rollout_summary.csv"),
        help="Rollout summary CSV output.",
    )
    args = parser.parse_args()

    run_rollout_eval(
        config_path=args.config,
        trajectory_csv=args.trajectory,
        rollout_steps=args.rollout_steps,
        output_steps_csv=args.steps_out,
        output_summary_csv=args.summary_out,
    )


if __name__ == "__main__":
    main()
