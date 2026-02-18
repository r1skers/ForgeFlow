import argparse
import csv
from pathlib import Path


def build_samples(trajectory_csv: Path, output_csv: Path) -> tuple[int, int]:
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    pair_rows = 0
    step_pairs = 0

    with trajectory_csv.open("r", encoding="utf-8", newline="") as in_file:
        reader = csv.DictReader(in_file)

        with output_csv.open("w", encoding="utf-8", newline="") as out_file:
            writer = csv.writer(out_file)
            writer.writerow(["step", "x", "y", "h_t", "h_t1"])

            current_step: int | None = None
            current_cells: list[tuple[int, int, float]] = []

            prev_step: int | None = None
            prev_cells: list[tuple[int, int, float]] | None = None

            for row in reader:
                step = int(row["step"])
                x = int(row["x"])
                y = int(row["y"])
                h = float(row["h"])

                if current_step is None:
                    current_step = step

                if step != current_step:
                    step_cells = current_cells

                    if prev_cells is not None:
                        if prev_step is None or current_step != prev_step + 1:
                            raise ValueError("trajectory steps must be contiguous for supervised export")
                        if len(prev_cells) != len(step_cells):
                            raise ValueError("grid cell count mismatch between adjacent steps")

                        for (x0, y0, h0), (x1, y1, h1) in zip(prev_cells, step_cells):
                            if x0 != x1 or y0 != y1:
                                raise ValueError("grid ordering mismatch between adjacent steps")
                            writer.writerow([prev_step, x0, y0, f"{h0:.12f}", f"{h1:.12f}"])
                            pair_rows += 1
                        step_pairs += 1

                    prev_cells = step_cells
                    prev_step = current_step
                    current_step = step
                    current_cells = []

                current_cells.append((x, y, h))

            if current_step is not None and prev_cells is not None:
                if len(prev_cells) != len(current_cells):
                    raise ValueError("grid cell count mismatch at final step transition")
                for (x0, y0, h0), (x1, y1, h1) in zip(prev_cells, current_cells):
                    if x0 != x1 or y0 != y1:
                        raise ValueError("grid ordering mismatch at final step transition")
                    writer.writerow([prev_step, x0, y0, f"{h0:.12f}", f"{h1:.12f}"])
                    pair_rows += 1
                step_pairs += 1

    return step_pairs, pair_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert simulation trajectory.csv into supervised samples (h_t -> h_t1)."
    )
    parser.add_argument(
        "--trajectory",
        type=Path,
        default=Path("ForgeFlowApps/ink_diffusion/output/trajectory.csv"),
        help="Input trajectory CSV path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ForgeFlowApps/ink_diffusion/data/processed/supervised_samples.csv"),
        help="Output supervised samples CSV path.",
    )
    args = parser.parse_args()

    step_pairs, pair_rows = build_samples(args.trajectory, args.output)
    print(f"[samples] step_pairs={step_pairs}")
    print(f"[samples] rows={pair_rows}")
    print(f"[samples] output={args.output.resolve()}")


if __name__ == "__main__":
    main()
