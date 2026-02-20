from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run full-feature USGS model across multiple site CSV pairs."
    )
    parser.add_argument(
        "--site-ids",
        nargs="+",
        default=["USGS-01491000", "USGS-13192200", "USGS-02450250"],
        help="Site IDs that already have train/infer CSV files generated.",
    )
    parser.add_argument(
        "--template-config",
        type=Path,
        default=Path("ForgeFlowApps/usgs_water_temp/config/run_full_features.json"),
        help="Template runtime config used for each site run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ForgeFlowApps/usgs_water_temp/output/multi_site"),
        help="Output directory for per-site artifacts and summary report.",
    )
    parser.add_argument(
        "--summary-file",
        type=str,
        default="site_metrics.csv",
        help="Summary CSV filename in output-dir.",
    )
    return parser


def compute_infer_metrics(predictions_csv: Path) -> dict[str, float]:
    residuals: list[float] = []
    with predictions_csv.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            residual_cell = str(row.get("residual", "")).strip()
            if residual_cell == "":
                continue
            residuals.append(float(residual_cell))

    if not residuals:
        raise RuntimeError(f"no scored residuals found in {predictions_csv}")

    n = float(len(residuals))
    abs_errors = [abs(value) for value in residuals]
    squared_errors = [value * value for value in residuals]
    mae = sum(abs_errors) / n
    mse = sum(squared_errors) / n
    rmse = math.sqrt(mse)
    maxae = max(abs_errors)
    return {
        "infer_n": n,
        "infer_mae": mae,
        "infer_mse": mse,
        "infer_rmse": rmse,
        "infer_maxae": maxae,
    }


def load_eval_metrics(eval_csv: Path) -> dict[str, str]:
    with eval_csv.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        row = next(reader, None)
    if row is None:
        raise RuntimeError(f"empty eval report: {eval_csv}")
    return row


def main() -> None:
    args = build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    template_config_path = args.template_config
    if not template_config_path.is_absolute():
        template_config_path = repo_root / template_config_path

    with template_config_path.open("r", encoding="utf-8") as file:
        template = json.load(file)
    if not isinstance(template, dict):
        raise RuntimeError("template config must be a JSON object")

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []

    for site_id in args.site_ids:
        train_csv = repo_root / "ForgeFlowApps" / "usgs_water_temp" / "data" / "processed" / (
            f"train_{site_id}.csv"
        )
        infer_csv = repo_root / "ForgeFlowApps" / "usgs_water_temp" / "data" / "infer" / (
            f"infer_{site_id}.csv"
        )
        if not train_csv.exists():
            raise FileNotFoundError(f"missing train csv for site {site_id}: {train_csv}")
        if not infer_csv.exists():
            raise FileNotFoundError(f"missing infer csv for site {site_id}: {infer_csv}")

        predictions_csv = output_dir / f"predictions_{site_id}.csv"
        eval_csv = output_dir / f"eval_report_{site_id}.csv"
        tmp_config = output_dir / f"_tmp_run_{site_id}.json"

        config_payload = dict(template)
        config_payload["task"] = f"usgs_water_temp_{site_id}"
        paths_payload = dict(config_payload.get("paths", {}))
        paths_payload["train_csv"] = str(train_csv)
        paths_payload["infer_csv"] = str(infer_csv)
        paths_payload["predictions_csv"] = str(predictions_csv)
        paths_payload["eval_report_csv"] = str(eval_csv)
        config_payload["paths"] = paths_payload

        with tmp_config.open("w", encoding="utf-8") as file:
            json.dump(config_payload, file, ensure_ascii=True, indent=2)

        subprocess.run(
            [sys.executable, "main.py", "--config", str(tmp_config)],
            cwd=repo_root,
            check=True,
        )

        infer_metrics = compute_infer_metrics(predictions_csv)
        eval_metrics = load_eval_metrics(eval_csv)

        row = {
            "site_id": site_id,
            "infer_n": f"{infer_metrics['infer_n']:.0f}",
            "infer_mae": f"{infer_metrics['infer_mae']:.6f}",
            "infer_mse": f"{infer_metrics['infer_mse']:.6f}",
            "infer_rmse": f"{infer_metrics['infer_rmse']:.6f}",
            "infer_maxae": f"{infer_metrics['infer_maxae']:.6f}",
            "val_mae": str(eval_metrics.get("val_mae", "")),
            "val_rmse": str(eval_metrics.get("val_rmse", "")),
            "val_maxae": str(eval_metrics.get("val_maxae", "")),
            "status": str(eval_metrics.get("status", "")),
        }
        rows.append(row)
        print(
            (
                f"[site:{site_id}] infer_mae={row['infer_mae']} infer_rmse={row['infer_rmse']} "
                f"infer_maxae={row['infer_maxae']} status={row['status']}"
            )
        )

    summary_csv = output_dir / args.summary_file
    with summary_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "site_id",
                "infer_n",
                "infer_mae",
                "infer_mse",
                "infer_rmse",
                "infer_maxae",
                "val_mae",
                "val_rmse",
                "val_maxae",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[multi-site] summary_file={summary_csv}")


if __name__ == "__main__":
    main()

