import argparse
import csv
from pathlib import Path


def _load_rows(report_csv: Path) -> list[dict[str, str]]:
    with report_csv.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _row_map(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["split"]: row for row in rows if "split" in row}


def _as_float(row: dict[str, str], key: str) -> float:
    return float(row.get(key, "nan"))


def _fmt(x: float) -> str:
    return f"{x:.6f}"


def _safe_ratio(a: float, b: float) -> float:
    if b == 0.0:
        return float("inf")
    return a / b


def _build_markdown(rows: list[dict[str, str]]) -> str:
    by_split = _row_map(rows)

    required = ["infer_id", "infer_ood"]
    missing = [name for name in required if name not in by_split]
    if missing:
        raise ValueError(f"missing required rows in infer metrics: {', '.join(missing)}")

    id_clean = by_split["infer_id"]
    ood_clean = by_split["infer_ood"]

    id_clean_mae = _as_float(id_clean, "mae")
    ood_clean_mae = _as_float(ood_clean, "mae")
    ood_id_ratio = _safe_ratio(ood_clean_mae, id_clean_mae)

    id_noise_rows = [
        row for row in rows if row.get("split", "").startswith("infer_id_noise_")
    ]
    ood_noise_rows = [
        row for row in rows if row.get("split", "").startswith("infer_ood_noise_")
    ]

    id_noise_rows.sort(key=lambda row: _as_float(row, "noise_std"))
    ood_noise_rows.sort(key=lambda row: _as_float(row, "noise_std"))

    lines: list[str] = []
    lines.append("# Heat Kappa Inverse Summary")
    lines.append("")
    lines.append("## Scope")
    lines.append("- Goal: estimate `kappa` from compact observation features.")
    lines.append("- Evaluation slices: ID, OOD, and noisy-ID/OOD.")
    lines.append("- Source: `infer_metrics_report.csv`.")
    lines.append("")
    lines.append("## Method (Sigma Rule)")
    lines.append("- Residual: `r = y_true - y_pred` (here `y` is `kappa`).")
    lines.append("- Validation baseline: `mu = mean(r_val)`, `sigma = std(r_val)`.")
    lines.append("- Anomaly threshold: `T = sigma_k * sigma`.")
    lines.append("- Flag condition: `abs(r - mu) > T`.")
    lines.append("- Notes: this rule uses residual statistics, not raw PDE state variables directly.")
    lines.append("")
    lines.append("## Key Results")
    lines.append(
        f"- Clean ID MAE: `{_fmt(id_clean_mae)}`; clean OOD MAE: `{_fmt(ood_clean_mae)}` "
        f"(OOD/ID ratio: `{_fmt(ood_id_ratio)}`x)."
    )

    if id_noise_rows:
        worst_id = max(id_noise_rows, key=lambda row: _as_float(row, "noise_std"))
        lines.append(
            f"- ID noise degradation (max tested): "
            f"`{worst_id.get('split', '-')}` MAE=`{worst_id.get('mae', '-')}`."
        )
    if ood_noise_rows:
        worst_ood = max(ood_noise_rows, key=lambda row: _as_float(row, "noise_std"))
        lines.append(
            f"- OOD noise degradation (max tested): "
            f"`{worst_ood.get('split', '-')}` MAE=`{worst_ood.get('mae', '-')}`."
        )

    lines.append("")
    lines.append("## Infer Metrics Table")
    lines.append("| split | noise_std | mae | rmse | maxae | anomaly_ratio |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            f"| {row.get('split', '-')} | {row.get('noise_std', '-')} | {row.get('mae', '-')} | "
            f"{row.get('rmse', '-')} | {row.get('maxae', '-')} | {row.get('anomaly_ratio', '-')} |"
        )

    lines.append("")
    lines.append("## Interpretation")
    lines.append("- The inverse regressor is accurate on clean ID data.")
    lines.append("- Error rises under distribution shift and observation noise, as expected.")
    lines.append("- Current sigma-rule anomaly flagging is sensitive and may over-flag noisy ID samples.")
    lines.append("")
    lines.append("## Next Actions")
    lines.append("- Calibrate anomaly threshold (`sigma_k`) to reduce noisy-ID false positives.")
    lines.append("- Keep OOD alerting high while improving ID robustness.")
    lines.append("- For real unlabeled deployment, add feature-space OOD detection.")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate markdown summary from infer metrics report.")
    parser.add_argument(
        "--report-csv",
        type=Path,
        default=Path("ForgeFlowApps/heat_kappa_inverse/output/infer_metrics_report.csv"),
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("ForgeFlowApps/heat_kappa_inverse/output/summary.md"),
    )
    args = parser.parse_args()

    rows = _load_rows(args.report_csv)
    if not rows:
        raise ValueError("infer metrics report is empty")
    content = _build_markdown(rows)

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(content, encoding="utf-8")
    print(f"[kappa-summary] rows={len(rows)}")
    print(f"[kappa-summary] summary_md={args.out_md.resolve()}")


if __name__ == "__main__":
    main()
