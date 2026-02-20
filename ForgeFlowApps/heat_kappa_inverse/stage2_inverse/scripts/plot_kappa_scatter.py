import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_pairs(predictions_csv: Path) -> tuple[np.ndarray, np.ndarray]:
    y_true: list[float] = []
    y_pred: list[float] = []
    with predictions_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pred = float(row["y_pred"])
            residual = str(row.get("residual", "")).strip()
            if residual == "":
                continue
            y_pred.append(pred)
            y_true.append(pred + float(residual))
    return np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)


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


def _split_group(split: str) -> str:
    if split.startswith("infer_id"):
        return "infer_id"
    if split.startswith("infer_ood"):
        return "infer_ood"
    return "other"


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot kappa true-vs-pred scatter for ID/OOD infer sets.")
    parser.add_argument(
        "--item",
        action="append",
        default=None,
        help="split=predictions_csv (repeatable).",
    )
    parser.add_argument(
        "--out-png",
        type=Path,
        default=Path("ForgeFlowApps/heat_kappa_inverse/output/kappa_scatter_report.png"),
    )
    args = parser.parse_args()

    items = _parse_items(args.item)
    groups = {"infer_id": [], "infer_ood": []}
    for split, path in items:
        if not path.exists():
            print(f"[kappa-scatter:{split}] skipped missing file={path}")
            continue
        group = _split_group(split)
        if group in groups:
            groups[group].append((split, path))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, group in zip(axes, ["infer_id", "infer_ood"]):
        all_true: list[np.ndarray] = []
        all_pred: list[np.ndarray] = []
        for split, path in groups[group]:
            y_true, y_pred = _load_pairs(path)
            if y_true.size == 0:
                continue
            all_true.append(y_true)
            all_pred.append(y_pred)
            ax.scatter(y_true, y_pred, s=8, alpha=0.6, label=split)

        if all_true and all_pred:
            stacked_true = np.concatenate(all_true)
            stacked_pred = np.concatenate(all_pred)
            lo = float(min(np.min(stacked_true), np.min(stacked_pred)))
            hi = float(max(np.max(stacked_true), np.max(stacked_pred)))
            ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
        ax.set_title(group)
        ax.set_xlabel("kappa_true")
        ax.set_ylabel("kappa_pred")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)

    fig.suptitle("Heat Kappa Inverse: True vs Pred")
    fig.tight_layout()
    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=150)
    plt.close(fig)
    print(f"[kappa-scatter] output={args.out_png.resolve()}")


if __name__ == "__main__":
    main()

