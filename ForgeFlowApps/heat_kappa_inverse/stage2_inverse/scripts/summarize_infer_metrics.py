import argparse
import csv
import math
from pathlib import Path
from statistics import mean, pstdev


def _read_residuals_and_anomalies(predictions_csv: Path) -> tuple[list[float], int, int]:
    residuals: list[float] = []
    total_rows = 0
    anomaly_rows = 0
    with predictions_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1
            residual_cell = str(row.get("residual", "")).strip()
            if residual_cell != "":
                residuals.append(float(residual_cell))
            anomaly_cell = str(row.get("anomaly_flag", "")).strip()
            if anomaly_cell == "1":
                anomaly_rows += 1
    return residuals, total_rows, anomaly_rows


def _parse_noise_std(split: str) -> float:
    marker = "_noise_"
    if marker not in split:
        return 0.0
    tag = split.split(marker, 1)[1]
    return float(tag.replace("p", "."))


def _split_group(split: str) -> str:
    if split.startswith("infer_id"):
        return "infer_id"
    if split.startswith("infer_ood"):
        return "infer_ood"
    return "other"


def _build_metrics(split: str, predictions_csv: Path) -> dict[str, str]:
    residuals, total_rows, anomaly_rows = _read_residuals_and_anomalies(predictions_csv)
    scored_rows = len(residuals)
    abs_res = [abs(x) for x in residuals]
    sq_res = [x * x for x in residuals]

    mae = float(mean(abs_res)) if abs_res else float("nan")
    rmse = float(math.sqrt(mean(sq_res))) if sq_res else float("nan")
    maxae = float(max(abs_res)) if abs_res else float("nan")
    res_mu = float(mean(residuals)) if residuals else float("nan")
    res_sigma = float(pstdev(residuals)) if len(residuals) >= 2 else 0.0
    anomaly_ratio = (anomaly_rows / scored_rows) if scored_rows > 0 else float("nan")
    noise_std = _parse_noise_std(split)

    return {
        "split": split,
        "split_group": _split_group(split),
        "noise_std": f"{noise_std:.6f}",
        "file": str(predictions_csv),
        "total_rows": str(total_rows),
        "scored_rows": str(scored_rows),
        "anomaly_rows": str(anomaly_rows),
        "anomaly_ratio": f"{anomaly_ratio:.6f}",
        "mae": f"{mae:.6f}",
        "rmse": f"{rmse:.6f}",
        "maxae": f"{maxae:.6f}",
        "residual_mean": f"{res_mu:.6f}",
        "residual_sigma": f"{res_sigma:.6f}",
    }


def _default_items() -> list[tuple[str, Path]]:
    return [
        ("infer_id", Path("ForgeFlowApps/heat_kappa_inverse/output/predictions_id.csv")),
        ("infer_id_noise_0p01", Path("ForgeFlowApps/heat_kappa_inverse/output/predictions_id_noise_0p01.csv")),
        ("infer_id_noise_0p03", Path("ForgeFlowApps/heat_kappa_inverse/output/predictions_id_noise_0p03.csv")),
        ("infer_ood", Path("ForgeFlowApps/heat_kappa_inverse/output/predictions_ood.csv")),
        ("infer_ood_noise_0p01", Path("ForgeFlowApps/heat_kappa_inverse/output/predictions_ood_noise_0p01.csv")),
        ("infer_ood_noise_0p03", Path("ForgeFlowApps/heat_kappa_inverse/output/predictions_ood_noise_0p03.csv")),
    ]


def _parse_items(raw_items: list[str] | None) -> list[tuple[str, Path]]:
    if not raw_items:
        return _default_items()
    items: list[tuple[str, Path]] = []
    for raw in raw_items:
        if "=" not in raw:
            raise ValueError(f"invalid --item '{raw}', expected split=path")
        split, raw_path = raw.split("=", 1)
        split = split.strip()
        raw_path = raw_path.strip()
        if not split or not raw_path:
            raise ValueError(f"invalid --item '{raw}', expected split=path")
        items.append((split, Path(raw_path)))
    return items


def _write_report(out_csv: Path, rows: list[dict[str, str]]) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "split_group",
        "noise_std",
        "file",
        "total_rows",
        "scored_rows",
        "anomaly_rows",
        "anomaly_ratio",
        "mae",
        "rmse",
        "maxae",
        "residual_mean",
        "residual_sigma",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize ID/OOD infer metrics for heat kappa inverse.")
    parser.add_argument(
        "--item",
        action="append",
        default=None,
        help="Override metric items using split=predictions_csv (repeatable).",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        default=False,
        help="Skip missing prediction files instead of failing.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("ForgeFlowApps/heat_kappa_inverse/output/infer_metrics_report.csv"),
    )
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    for split, predictions_csv in _parse_items(args.item):
        if not predictions_csv.exists():
            if args.skip_missing:
                print(f"[kappa-infer:{split}] skipped missing file={predictions_csv}")
                continue
            raise FileNotFoundError(f"missing predictions file: {predictions_csv}")
        row = _build_metrics(split, predictions_csv)
        rows.append(row)
        print(
            f"[kappa-infer:{row['split']}] "
            f"mae={row['mae']} rmse={row['rmse']} maxae={row['maxae']} "
            f"anomaly_ratio={row['anomaly_ratio']}"
        )

    if not rows:
        raise ValueError("no infer metrics rows were generated")

    _write_report(args.out_csv, rows)
    print(f"[kappa-infer] report_csv={args.out_csv.resolve()}")


if __name__ == "__main__":
    main()

