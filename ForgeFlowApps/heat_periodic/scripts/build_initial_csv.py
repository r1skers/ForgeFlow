import argparse
import csv
import math
from pathlib import Path


def build_initial_csv(
    output_csv: Path,
    nx: int,
    ny: int,
    amplitude: float,
    mode_x: int,
    mode_y: int,
) -> tuple[float, float]:
    if nx <= 0 or ny <= 0:
        raise ValueError("nx and ny must be > 0")
    if mode_x <= 0 or mode_y <= 0:
        raise ValueError("mode_x and mode_y must be > 0")

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    min_h = float("inf")
    max_h = float("-inf")

    with output_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["x", "y", "h"])
        for y_idx in range(ny):
            y = y_idx / float(ny)
            for x_idx in range(nx):
                x = x_idx / float(nx)
                h = amplitude * math.sin(2.0 * math.pi * mode_x * x) * math.sin(
                    2.0 * math.pi * mode_y * y
                )
                min_h = min(min_h, h)
                max_h = max(max_h, h)
                writer.writerow([x_idx, y_idx, f"{h:.12f}"])

    return min_h, max_h


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build periodic 2D heat benchmark initial condition CSV."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ForgeFlowApps/heat_periodic/data/processed/initial.csv"),
        help="Output CSV path.",
    )
    parser.add_argument("--nx", type=int, default=100, help="Grid width.")
    parser.add_argument("--ny", type=int, default=100, help="Grid height.")
    parser.add_argument("--amplitude", type=float, default=1.0, help="Initial amplitude.")
    parser.add_argument("--mode-x", type=int, default=1, help="Wave mode along x.")
    parser.add_argument("--mode-y", type=int, default=1, help="Wave mode along y.")
    args = parser.parse_args()

    min_h, max_h = build_initial_csv(
        output_csv=args.output,
        nx=int(args.nx),
        ny=int(args.ny),
        amplitude=float(args.amplitude),
        mode_x=int(args.mode_x),
        mode_y=int(args.mode_y),
    )

    print(f"[heat-initial] grid={int(args.nx)}x{int(args.ny)}")
    print(f"[heat-initial] amplitude={float(args.amplitude):.6f}")
    print(f"[heat-initial] mode_x={int(args.mode_x)} mode_y={int(args.mode_y)}")
    print(f"[heat-initial] min_h={min_h:.6f}")
    print(f"[heat-initial] max_h={max_h:.6f}")
    print(f"[heat-initial] output={args.output.resolve()}")


if __name__ == "__main__":
    main()
