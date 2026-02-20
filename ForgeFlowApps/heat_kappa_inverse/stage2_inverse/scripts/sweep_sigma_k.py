import argparse
import csv
import importlib
import math
import statistics
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from forgeflow.core.config.runtime import RuntimeConfig, load_runtime_config
from forgeflow.core.data.split import split_train_val
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


def _parse_items(raw_items: list[str] | None) -> list[tuple[str, Path]]:
    if not raw_items:
        return [
            ("infer_id", PROJECT_ROOT / "ForgeFlowApps/heat_kappa_inverse/data/processed/infer_id.csv"),
            (
                "infer_id_noise_0p01",
                PROJECT_ROOT / "ForgeFlowApps/heat_kappa_inverse/data/processed/infer_id_noise_0p01.csv",
            ),
            (
                "infer_id_noise_0p03",
                PROJECT_ROOT / "ForgeFlowApps/heat_kappa_inverse/data/processed/infer_id_noise_0p03.csv",
            ),
            ("infer_ood", PROJECT_ROOT / "ForgeFlowApps/heat_kappa_inverse/data/processed/infer_ood.csv"),
            (
                "infer_ood_noise_0p01",
                PROJECT_ROOT / "ForgeFlowApps/heat_kappa_inverse/data/processed/infer_ood_noise_0p01.csv",
            ),
            (
                "infer_ood_noise_0p03",
                PROJECT_ROOT / "ForgeFlowApps/heat_kappa_inverse/data/processed/infer_ood_noise_0p03.csv",
            ),
        ]

    items: list[tuple[str, Path]] = []
    for raw in raw_items:
        if "=" not in raw:
            raise ValueError(f"invalid --item '{raw}', expected split=path")
        split, path_str = raw.split("=", 1)
        split = split.strip()
        path_str = path_str.strip()
        if not split or not path_str:
            raise ValueError(f"invalid --item '{raw}', expected split=path")
        path = Path(path_str)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        items.append((split, path))
    return items


def _parse_sigma_values(raw_values: str) -> list[float]:
    values = [float(x.strip()) for x in raw_values.split(",") if x.strip()]
    if not values:
        raise ValueError("at least one sigma value is required")
    for value in values:
        if value <= 0:
            raise ValueError("sigma values must be > 0")
    return values


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


def _compute_basic_metrics(residuals: list[float]) -> tuple[float, float, float]:
    if not residuals:
        return float("nan"), float("nan"), float("nan")
    abs_vals = [abs(x) for x in residuals]
    mae = float(statistics.mean(abs_vals))
    rmse = float(math.sqrt(statistics.mean([x * x for x in residuals])))
    maxae = float(max(abs_vals))
    return mae, rmse, maxae


def _fit_model_and_val_stats(runtime: RuntimeConfig) -> tuple[Any, Any, float, float]:
    if runtime.paths.train_csv is None:
        raise ValueError("supervised config requires paths.train_csv")

    adapter_cls = _resolve_adapter_class(runtime)
    model_cls = _resolve_model_class(runtime)

    records, _ = read_csv_records(runtime.paths.train_csv)
    adapter = adapter_cls()
    states, _ = adapter.adapt_records(records)
    x, _ = adapter.build_feature_matrix(states)
    y, _ = adapter.build_target_matrix(states)
    x_train, y_train, x_val, y_val, _ = split_train_val(
        x,
        y,
        train_ratio=runtime.train_ratio,
        shuffle=runtime.split_shuffle,
        seed=runtime.split_seed,
    )
    model = model_cls()
    model.fit(x_train, y_train)
    val_pred = model.predict(x_val)
    val_residuals = [float(t[0] - p[0]) for t, p in zip(y_val, val_pred)]
    mu = float(statistics.mean(val_residuals))
    sigma = float(statistics.pstdev(val_residuals))
    sigma = max(sigma, 1e-9)
    return adapter, model, mu, sigma


def _infer_residuals(adapter: Any, model: Any, infer_csv: Path) -> tuple[list[float], int]:
    records, _ = read_csv_records(infer_csv)
    to_infer = getattr(adapter, "to_infer_feature_vector", None)
    if not callable(to_infer):
        raise ValueError("adapter must implement to_infer_feature_vector")

    x: list[list[float]] = []
    y_true: list[float] = []
    for record in records:
        try:
            feature_vector = to_infer(record)
            y_cell = str(record.get("y", "")).strip()
            if y_cell == "":
                continue
            x.append([float(v) for v in feature_vector])
            y_true.append(float(y_cell))
        except (KeyError, TypeError, ValueError):
            continue

    if not x:
        return [], 0

    y_pred = model.predict(x)
    residuals = [float(t - p[0]) for t, p in zip(y_true, y_pred)]
    return residuals, len(records)


def _recommend_sigma(rows: list[dict[str, str]], ood_floor: float = 0.95) -> float:
    grouped: dict[float, list[dict[str, str]]] = {}
    for row in rows:
        sigma_k = float(row["sigma_k"])
        grouped.setdefault(sigma_k, []).append(row)

    feasible: list[tuple[float, float, float]] = []
    fallback: list[tuple[float, float, float]] = []
    for sigma_k, sigma_rows in grouped.items():
        id_noisy = [
            float(r["anomaly_ratio"])
            for r in sigma_rows
            if r["split"].startswith("infer_id_noise_")
        ]
        ood_all = [
            float(r["anomaly_ratio"])
            for r in sigma_rows
            if r["split"].startswith("infer_ood")
        ]
        id_noisy_mean = statistics.mean(id_noisy) if id_noisy else float("nan")
        ood_mean = statistics.mean(ood_all) if ood_all else float("nan")
        item = (sigma_k, id_noisy_mean, ood_mean)
        fallback.append(item)
        if ood_mean >= ood_floor:
            feasible.append(item)

    candidates = feasible if feasible else fallback
    candidates.sort(key=lambda item: (item[1], -item[2]))
    return candidates[0][0]


def _write_csv(out_csv: Path, rows: list[dict[str, str]]) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "split_group",
        "noise_std",
        "sigma_k",
        "infer_file",
        "total_rows",
        "scored_rows",
        "anomaly_rows",
        "anomaly_ratio",
        "mae",
        "rmse",
        "maxae",
        "val_mu",
        "val_sigma",
        "threshold",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_recommendation_md(
    out_md: Path,
    base_config: Path,
    sigma_values: list[float],
    recommended_sigma: float,
    rows: list[dict[str, str]],
) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("# Sigma-K Calibration")
    lines.append("")
    lines.append(f"- Base config: `{base_config}`")
    lines.append(f"- Candidate sigma_k: `{', '.join(f'{v:.2f}' for v in sigma_values)}`")
    lines.append(f"- Recommended sigma_k: `{recommended_sigma:.2f}`")
    lines.append("")
    lines.append("| sigma_k | id_noisy_anomaly_mean | ood_anomaly_mean |")
    lines.append("|---:|---:|---:|")

    for sigma_k in sigma_values:
        sigma_rows = [r for r in rows if float(r["sigma_k"]) == sigma_k]
        id_noisy = [
            float(r["anomaly_ratio"]) for r in sigma_rows if r["split"].startswith("infer_id_noise_")
        ]
        ood_all = [float(r["anomaly_ratio"]) for r in sigma_rows if r["split"].startswith("infer_ood")]
        id_noisy_mean = statistics.mean(id_noisy) if id_noisy else float("nan")
        ood_mean = statistics.mean(ood_all) if ood_all else float("nan")
        lines.append(f"| {sigma_k:.2f} | {id_noisy_mean:.6f} | {ood_mean:.6f} |")

    lines.append("")
    lines.append("Selection rule: prefer sigma values with OOD anomaly mean >= 0.95, then minimize noisy-ID anomaly mean.")
    lines.append("")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep sigma_k anomaly threshold for heat kappa inverse.")
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("ForgeFlowApps/heat_kappa_inverse/stage2_inverse/config/run_id.json"),
    )
    parser.add_argument(
        "--sigma-values",
        type=str,
        default="3,4,5",
        help="Comma-separated sigma_k values, e.g. 3,4,5",
    )
    parser.add_argument(
        "--item",
        action="append",
        default=None,
        help="split=infer_csv (repeatable). If omitted, use default ID/OOD clean+noise infer files.",
    )
    parser.add_argument("--skip-missing", action="store_true", default=True)
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("ForgeFlowApps/heat_kappa_inverse/output/sigma_k_sweep_report.csv"),
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("ForgeFlowApps/heat_kappa_inverse/output/sigma_k_recommendation.md"),
    )
    args = parser.parse_args()

    base_config = args.base_config if args.base_config.is_absolute() else PROJECT_ROOT / args.base_config
    runtime = load_runtime_config(base_config, PROJECT_ROOT)
    if runtime.mode != "supervised":
        raise ValueError("sigma sweep requires a supervised config")

    sigma_values = _parse_sigma_values(args.sigma_values)
    items = _parse_items(args.item)

    adapter, model, val_mu, val_sigma = _fit_model_and_val_stats(runtime)
    rows: list[dict[str, str]] = []

    for split, infer_csv in items:
        if not infer_csv.exists():
            if args.skip_missing:
                print(f"[sigma-sweep:{split}] skipped missing file={infer_csv}")
                continue
            raise FileNotFoundError(f"missing infer file: {infer_csv}")

        residuals, total_rows = _infer_residuals(adapter, model, infer_csv)
        scored_rows = len(residuals)
        mae, rmse, maxae = _compute_basic_metrics(residuals)

        for sigma_k in sigma_values:
            threshold = sigma_k * val_sigma
            anomaly_rows = sum(1 for r in residuals if abs(r - val_mu) > threshold)
            anomaly_ratio = (anomaly_rows / scored_rows) if scored_rows > 0 else float("nan")
            row = {
                "split": split,
                "split_group": _split_group(split),
                "noise_std": f"{_parse_noise_std(split):.6f}",
                "sigma_k": f"{sigma_k:.6f}",
                "infer_file": str(infer_csv),
                "total_rows": str(total_rows),
                "scored_rows": str(scored_rows),
                "anomaly_rows": str(anomaly_rows),
                "anomaly_ratio": f"{anomaly_ratio:.6f}",
                "mae": f"{mae:.6f}",
                "rmse": f"{rmse:.6f}",
                "maxae": f"{maxae:.6f}",
                "val_mu": f"{val_mu:.6f}",
                "val_sigma": f"{val_sigma:.6f}",
                "threshold": f"{threshold:.6f}",
            }
            rows.append(row)
            print(
                f"[sigma-sweep:{split}] sigma_k={sigma_k:.2f} "
                f"anomaly_ratio={row['anomaly_ratio']} mae={row['mae']}"
            )

    if not rows:
        raise ValueError("no sigma-sweep rows generated")

    recommended_sigma = _recommend_sigma(rows)
    out_csv = args.out_csv if args.out_csv.is_absolute() else PROJECT_ROOT / args.out_csv
    out_md = args.out_md if args.out_md.is_absolute() else PROJECT_ROOT / args.out_md
    _write_csv(out_csv, rows)
    _write_recommendation_md(out_md, base_config, sigma_values, recommended_sigma, rows)
    print(f"[sigma-sweep] recommended_sigma_k={recommended_sigma:.2f}")
    print(f"[sigma-sweep] report_csv={out_csv.resolve()}")
    print(f"[sigma-sweep] recommendation_md={out_md.resolve()}")


if __name__ == "__main__":
    main()
