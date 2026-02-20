from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare USGS temp-only vs full-feature runs.")
    parser.add_argument(
        "--temp-only-predictions",
        type=Path,
        default=Path("ForgeFlowApps/usgs_water_temp/output/predictions_temp_only.csv"),
        help="Path to predictions CSV from temp-only config.",
    )
    parser.add_argument(
        "--full-predictions",
        type=Path,
        default=Path("ForgeFlowApps/usgs_water_temp/output/predictions_full_features.csv"),
        help="Path to predictions CSV from full-feature config.",
    )
    parser.add_argument(
        "--out-report",
        type=Path,
        default=Path("ForgeFlowApps/usgs_water_temp/output/ab_feature_compare.csv"),
        help="Output CSV report path.",
    )
    return parser


def compute_metrics(predictions_csv: Path) -> dict[str, float]:
    residuals: list[float] = []
    with predictions_csv.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            residual_cell = str(row.get("residual", "")).strip()
            if residual_cell == "":
                continue
            residuals.append(float(residual_cell))

    if not residuals:
        raise ValueError(f"no scored residuals found in {predictions_csv}")

    n = float(len(residuals))
    abs_errors = [abs(value) for value in residuals]
    squared_errors = [value * value for value in residuals]
    mae = sum(abs_errors) / n
    mse = sum(squared_errors) / n
    rmse = math.sqrt(mse)
    maxae = max(abs_errors)
    return {
        "n": n,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "maxae": maxae,
    }


def main() -> None:
    args = build_parser().parse_args()
    temp_metrics = compute_metrics(args.temp_only_predictions)
    full_metrics = compute_metrics(args.full_predictions)

    mae_gain = temp_metrics["mae"] - full_metrics["mae"]
    rmse_gain = temp_metrics["rmse"] - full_metrics["rmse"]
    mae_gain_ratio = mae_gain / temp_metrics["mae"] if temp_metrics["mae"] > 0 else 0.0
    rmse_gain_ratio = rmse_gain / temp_metrics["rmse"] if temp_metrics["rmse"] > 0 else 0.0

    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    with args.out_report.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "scenario",
                "n",
                "mae",
                "mse",
                "rmse",
                "maxae",
                "mae_gain_vs_temp_only",
                "rmse_gain_vs_temp_only",
                "mae_gain_ratio_vs_temp_only",
                "rmse_gain_ratio_vs_temp_only",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "scenario": "temp_only",
                "n": f"{temp_metrics['n']:.0f}",
                "mae": f"{temp_metrics['mae']:.6f}",
                "mse": f"{temp_metrics['mse']:.6f}",
                "rmse": f"{temp_metrics['rmse']:.6f}",
                "maxae": f"{temp_metrics['maxae']:.6f}",
                "mae_gain_vs_temp_only": "0.000000",
                "rmse_gain_vs_temp_only": "0.000000",
                "mae_gain_ratio_vs_temp_only": "0.000000",
                "rmse_gain_ratio_vs_temp_only": "0.000000",
            }
        )
        writer.writerow(
            {
                "scenario": "full_features",
                "n": f"{full_metrics['n']:.0f}",
                "mae": f"{full_metrics['mae']:.6f}",
                "mse": f"{full_metrics['mse']:.6f}",
                "rmse": f"{full_metrics['rmse']:.6f}",
                "maxae": f"{full_metrics['maxae']:.6f}",
                "mae_gain_vs_temp_only": f"{mae_gain:.6f}",
                "rmse_gain_vs_temp_only": f"{rmse_gain:.6f}",
                "mae_gain_ratio_vs_temp_only": f"{mae_gain_ratio:.6f}",
                "rmse_gain_ratio_vs_temp_only": f"{rmse_gain_ratio:.6f}",
            }
        )

    print(f"[ab] temp_only_mae={temp_metrics['mae']:.6f}")
    print(f"[ab] full_features_mae={full_metrics['mae']:.6f}")
    print(f"[ab] mae_gain={mae_gain:.6f}")
    print(f"[ab] mae_gain_ratio={mae_gain_ratio:.6f}")
    print(f"[ab] report_file={args.out_report}")


if __name__ == "__main__":
    main()

