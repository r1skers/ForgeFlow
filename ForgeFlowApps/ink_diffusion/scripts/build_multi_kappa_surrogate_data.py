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
        raise ValueError(f"trajectory {trajectory_csv} must contain at least two steps")
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
                f"trajectory {trajectory_csv} step {step} has {count} cells, expected {expected_cells}"
            )

    return grids


def _parse_manifest(manifest_csv: Path) -> list[dict[str, str]]:
    with manifest_csv.open("r", encoding="utf-8", newline="") as in_file:
        reader = csv.DictReader(in_file)
        rows = list(reader)
    if not rows:
        raise ValueError("manifest is empty")
    required_fields = {"kappa", "trajectory_csv", "status"}
    if any(not required_fields.issubset(set(row.keys())) for row in rows):
        raise ValueError("manifest must include fields: kappa, trajectory_csv, status")
    return rows


def _resolve_kappa_splits(
    kappas: list[float],
    train_kappas: list[float] | None,
    ood_kappas: list[float] | None,
) -> tuple[set[float], set[float]]:
    available = sorted({float(k) for k in kappas})
    available_set = set(available)

    if train_kappas is None:
        if len(available) < 2:
            raise ValueError("need at least 2 kappas to split train and ood")
        inferred_train = set(available[:-1])
    else:
        inferred_train = {float(k) for k in train_kappas}

    if ood_kappas is None:
        inferred_ood = available_set - inferred_train
    else:
        inferred_ood = {float(k) for k in ood_kappas}

    if not inferred_train:
        raise ValueError("train_kappas resolved to empty set")
    if not inferred_ood:
        raise ValueError("ood_kappas resolved to empty set")
    if inferred_train & inferred_ood:
        overlap = sorted(inferred_train & inferred_ood)
        raise ValueError(f"train/ood kappa overlap is not allowed: {overlap}")
    if not inferred_train.issubset(available_set):
        raise ValueError("train_kappas contains value not in manifest")
    if not inferred_ood.issubset(available_set):
        raise ValueError("ood_kappas contains value not in manifest")

    return inferred_train, inferred_ood


def build_multi_kappa_surrogate_data(
    manifest_csv: Path,
    train_out: Path,
    infer_id_out: Path,
    infer_ood_out: Path,
    spatial_stride: int,
    train_ratio: float,
    train_kappas: list[float] | None,
    ood_kappas: list[float] | None,
) -> tuple[int, int, int]:
    if spatial_stride <= 0:
        raise ValueError("spatial_stride must be > 0")
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")

    entries = _parse_manifest(manifest_csv)
    pass_entries = [row for row in entries if str(row["status"]).strip() == "PASS"]
    if not pass_entries:
        raise ValueError("manifest has no PASS trajectories")

    kappas = [float(row["kappa"]) for row in pass_entries]
    train_set, ood_set = _resolve_kappa_splits(
        kappas=kappas, train_kappas=train_kappas, ood_kappas=ood_kappas
    )

    train_out.parent.mkdir(parents=True, exist_ok=True)
    infer_id_out.parent.mkdir(parents=True, exist_ok=True)
    infer_ood_out.parent.mkdir(parents=True, exist_ok=True)

    row_header = [
        "kappa",
        "step",
        "x_idx",
        "y_idx",
        "h_t",
        "h_up",
        "h_down",
        "h_left",
        "h_right",
        "h_t1",
        "y",
        "split",
    ]
    train_rows = 0
    infer_id_rows = 0
    infer_ood_rows = 0

    with train_out.open("w", encoding="utf-8", newline="") as train_file:
        with infer_id_out.open("w", encoding="utf-8", newline="") as infer_id_file:
            with infer_ood_out.open("w", encoding="utf-8", newline="") as infer_ood_file:
                train_writer = csv.writer(train_file)
                infer_id_writer = csv.writer(infer_id_file)
                infer_ood_writer = csv.writer(infer_ood_file)
                train_writer.writerow(row_header)
                infer_id_writer.writerow(row_header)
                infer_ood_writer.writerow(row_header)

                for row in pass_entries:
                    kappa = float(row["kappa"])
                    trajectory_csv = Path(row["trajectory_csv"])
                    if not trajectory_csv.is_absolute():
                        trajectory_csv = (manifest_csv.parent / trajectory_csv).resolve()

                    max_step, nx, ny = _scan_trajectory_shape(trajectory_csv)
                    grids = _load_grids(trajectory_csv=trajectory_csv, max_step=max_step, nx=nx, ny=ny)

                    pair_count = max_step
                    train_pair_count = int(pair_count * train_ratio)
                    train_pair_count = min(max(train_pair_count, 1), pair_count - 1)

                    if kappa in train_set:
                        split_name = "train_id"
                    elif kappa in ood_set:
                        split_name = "ood"
                    else:
                        continue

                    for step in range(pair_count):
                        current = grids[step]
                        nxt = grids[step + 1]
                        if split_name == "train_id":
                            writer = train_writer if step < train_pair_count else infer_id_writer
                            row_counter = "train" if step < train_pair_count else "infer_id"
                        else:
                            writer = infer_ood_writer
                            row_counter = "infer_ood"

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
                                        f"{kappa:.12f}",
                                        step,
                                        x_idx,
                                        y_idx,
                                        f"{h_t:.12f}",
                                        f"{h_up:.12f}",
                                        f"{h_down:.12f}",
                                        f"{h_left:.12f}",
                                        f"{h_right:.12f}",
                                        f"{h_t1:.12f}",
                                        f"{h_t1:.12f}",
                                        row_counter,
                                    ]
                                )
                                if row_counter == "train":
                                    train_rows += 1
                                elif row_counter == "infer_id":
                                    infer_id_rows += 1
                                else:
                                    infer_ood_rows += 1

                    print(
                        f"[multi-kappa-data:{kappa:g}] split={split_name} "
                        f"pairs={pair_count} grid={nx}x{ny}"
                    )

    return train_rows, infer_id_rows, infer_ood_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build surrogate datasets from multi-kappa trajectories. "
            "Outputs train/infer_id/infer_ood CSV files with kappa-aware local features."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("ForgeFlowApps/ink_diffusion/output/multi_kappa/manifest.csv"),
        help="Manifest generated by build_multi_kappa_trajectories.py.",
    )
    parser.add_argument(
        "--train-out",
        type=Path,
        default=Path("ForgeFlowApps/ink_diffusion/data/processed/surrogate_kappa_train.csv"),
        help="Output train CSV.",
    )
    parser.add_argument(
        "--infer-id-out",
        type=Path,
        default=Path("ForgeFlowApps/ink_diffusion/data/processed/surrogate_kappa_infer_id.csv"),
        help="Output in-domain infer CSV (seen kappa, held-out time pairs).",
    )
    parser.add_argument(
        "--infer-ood-out",
        type=Path,
        default=Path("ForgeFlowApps/ink_diffusion/data/processed/surrogate_kappa_infer_ood.csv"),
        help="Output out-of-domain infer CSV (held-out kappa set).",
    )
    parser.add_argument(
        "--spatial-stride",
        type=int,
        default=4,
        help="Sub-sampling stride over spatial grid to control dataset size.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Step-pair ratio assigned to train for train_kappa trajectories.",
    )
    parser.add_argument(
        "--train-kappas",
        type=float,
        nargs="+",
        default=None,
        help="Optional explicit train kappa set. Default: all except max kappa in manifest.",
    )
    parser.add_argument(
        "--ood-kappas",
        type=float,
        nargs="+",
        default=None,
        help="Optional explicit OOD kappa set. Default: manifest kappas not in train set.",
    )
    args = parser.parse_args()

    train_rows, infer_id_rows, infer_ood_rows = build_multi_kappa_surrogate_data(
        manifest_csv=args.manifest,
        train_out=args.train_out,
        infer_id_out=args.infer_id_out,
        infer_ood_out=args.infer_ood_out,
        spatial_stride=int(args.spatial_stride),
        train_ratio=float(args.train_ratio),
        train_kappas=list(args.train_kappas) if args.train_kappas is not None else None,
        ood_kappas=list(args.ood_kappas) if args.ood_kappas is not None else None,
    )
    print(f"[multi-kappa-data] train_rows={train_rows}")
    print(f"[multi-kappa-data] infer_id_rows={infer_id_rows}")
    print(f"[multi-kappa-data] infer_ood_rows={infer_ood_rows}")
    print(f"[multi-kappa-data] train_file={args.train_out.resolve()}")
    print(f"[multi-kappa-data] infer_id_file={args.infer_id_out.resolve()}")
    print(f"[multi-kappa-data] infer_ood_file={args.infer_ood_out.resolve()}")


if __name__ == "__main__":
    main()
