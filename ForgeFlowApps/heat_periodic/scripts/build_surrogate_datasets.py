import argparse
import csv
from pathlib import Path


def _scan_trajectory_shape(trajectory_csv: Path) -> tuple[int, int, int]:
    max_step = -1
    max_x = -1
    max_y = -1

    with trajectory_csv.open("r", encoding="utf-8", newline="") as in_file:
        reader = csv.DictReader(in_file)
        for row in reader:
            step = int(row["step"])
            x_idx = int(row["x"])
            y_idx = int(row["y"])
            max_step = max(max_step, step)
            max_x = max(max_x, x_idx)
            max_y = max(max_y, y_idx)

    if max_step < 1:
        raise ValueError("trajectory must contain at least two steps (step 0 and step 1)")
    return max_step, (max_x + 1), (max_y + 1)


def _load_grids(trajectory_csv: Path, max_step: int, nx: int, ny: int) -> list[list[list[float]]]:
    grids = [[[0.0 for _ in range(nx)] for _ in range(ny)] for _ in range(max_step + 1)]
    cell_counts = [0 for _ in range(max_step + 1)]

    with trajectory_csv.open("r", encoding="utf-8", newline="") as in_file:
        reader = csv.DictReader(in_file)
        for row in reader:
            step = int(row["step"])
            x_idx = int(row["x"])
            y_idx = int(row["y"])
            h_val = float(row["h"])
            grids[step][y_idx][x_idx] = h_val
            cell_counts[step] += 1

    expected_cells = nx * ny
    for step, count in enumerate(cell_counts):
        if count != expected_cells:
            raise ValueError(
                f"trajectory step {step} has {count} cells, expected {expected_cells}"
            )

    return grids


def build_surrogate_datasets(
    trajectory_csv: Path,
    train_csv: Path,
    infer_csv: Path,
    train_ratio: float,
    spatial_stride: int,
) -> tuple[int, int, int, int]:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if spatial_stride <= 0:
        raise ValueError("spatial_stride must be > 0")

    max_step, nx, ny = _scan_trajectory_shape(trajectory_csv)
    grids = _load_grids(trajectory_csv, max_step=max_step, nx=nx, ny=ny)

    pair_count = max_step
    train_pair_count = int(pair_count * train_ratio)
    train_pair_count = min(max(train_pair_count, 1), pair_count - 1)

    train_csv.parent.mkdir(parents=True, exist_ok=True)
    infer_csv.parent.mkdir(parents=True, exist_ok=True)

    row_header = ["step", "x_idx", "y_idx", "h_t", "h_up", "h_down", "h_left", "h_right", "h_t1"]
    train_rows = 0
    infer_rows = 0

    with train_csv.open("w", encoding="utf-8", newline="") as train_file:
        with infer_csv.open("w", encoding="utf-8", newline="") as infer_file:
            train_writer = csv.writer(train_file)
            infer_writer = csv.writer(infer_file)
            train_writer.writerow(row_header)
            infer_writer.writerow(row_header)

            for step in range(pair_count):
                current = grids[step]
                nxt = grids[step + 1]
                writer = train_writer if step < train_pair_count else infer_writer

                for y_idx in range(0, ny, spatial_stride):
                    up_idx = (y_idx - 1) % ny
                    down_idx = (y_idx + 1) % ny
                    for x_idx in range(0, nx, spatial_stride):
                        left_idx = (x_idx - 1) % nx
                        right_idx = (x_idx + 1) % nx

                        h_t = current[y_idx][x_idx]
                        h_up = current[up_idx][x_idx]
                        h_down = current[down_idx][x_idx]
                        h_left = current[y_idx][left_idx]
                        h_right = current[y_idx][right_idx]
                        h_t1 = nxt[y_idx][x_idx]

                        writer.writerow(
                            [
                                step,
                                x_idx,
                                y_idx,
                                f"{h_t:.12f}",
                                f"{h_up:.12f}",
                                f"{h_down:.12f}",
                                f"{h_left:.12f}",
                                f"{h_right:.12f}",
                                f"{h_t1:.12f}",
                            ]
                        )
                        if step < train_pair_count:
                            train_rows += 1
                        else:
                            infer_rows += 1

    return nx, ny, train_rows, infer_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build supervised train/infer datasets from heat periodic trajectory. "
            "Features are local 5-point neighborhood values at time t, target is h_t1."
        )
    )
    parser.add_argument(
        "--trajectory",
        type=Path,
        default=Path("ForgeFlowApps/heat_periodic/output/trajectory.csv"),
        help="Input trajectory CSV from simulation pipeline.",
    )
    parser.add_argument(
        "--train-out",
        type=Path,
        default=Path("ForgeFlowApps/heat_periodic/data/processed/surrogate_train.csv"),
        help="Output supervised train CSV path.",
    )
    parser.add_argument(
        "--infer-out",
        type=Path,
        default=Path("ForgeFlowApps/heat_periodic/data/processed/surrogate_infer.csv"),
        help="Output supervised infer CSV path.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Step-pair ratio assigned to train split.",
    )
    parser.add_argument(
        "--spatial-stride",
        type=int,
        default=4,
        help="Sub-sampling stride over spatial grid to control dataset size.",
    )
    args = parser.parse_args()

    nx, ny, train_rows, infer_rows = build_surrogate_datasets(
        trajectory_csv=args.trajectory,
        train_csv=args.train_out,
        infer_csv=args.infer_out,
        train_ratio=args.train_ratio,
        spatial_stride=args.spatial_stride,
    )
    print(f"[heat-surrogate-data] grid={nx}x{ny}")
    print(f"[heat-surrogate-data] train_rows={train_rows}")
    print(f"[heat-surrogate-data] infer_rows={infer_rows}")
    print(f"[heat-surrogate-data] train_file={args.train_out.resolve()}")
    print(f"[heat-surrogate-data] infer_file={args.infer_out.resolve()}")


if __name__ == "__main__":
    main()
