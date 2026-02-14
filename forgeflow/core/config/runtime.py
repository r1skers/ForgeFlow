import json
from dataclasses import dataclass
from pathlib import Path

from forgeflow.interfaces import EvalPolicy


@dataclass(frozen=True)
class PathConfig:
    train_csv: Path
    infer_csv: Path
    predictions_csv: Path
    eval_report_csv: Path


@dataclass(frozen=True)
class RuntimeConfig:
    task: str
    adapter: str
    model: str
    paths: PathConfig
    train_ratio: float
    anomaly_sigma_k: float
    eval_policy_override: EvalPolicy


def _resolve_path(project_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return project_root / path


def load_runtime_config(config_path: Path, project_root: Path) -> RuntimeConfig:
    with config_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    paths_payload = payload["paths"]
    split_payload = payload.get("split", {})
    anomaly_payload = payload.get("anomaly", {})
    eval_policy_payload = payload.get("eval_policy", {})

    train_ratio = float(split_payload.get("train_ratio", 0.8))
    if not 0 < train_ratio < 1:
        raise ValueError("split.train_ratio must be between 0 and 1")

    anomaly_sigma_k = float(anomaly_payload.get("sigma_k", 3.0))
    if anomaly_sigma_k <= 0:
        raise ValueError("anomaly.sigma_k must be > 0")

    eval_policy_override: EvalPolicy = {
        "val_mae_max": float(eval_policy_payload.get("val_mae_max", 0.1)),
        "val_rmse_max": float(eval_policy_payload.get("val_rmse_max", 0.1)),
        "val_maxae_max": float(eval_policy_payload.get("val_maxae_max", 0.2)),
    }

    return RuntimeConfig(
        task=str(payload["task"]),
        adapter=str(payload["adapter"]),
        model=str(payload["model"]),
        paths=PathConfig(
            train_csv=_resolve_path(project_root, str(paths_payload["train_csv"])),
            infer_csv=_resolve_path(project_root, str(paths_payload["infer_csv"])),
            predictions_csv=_resolve_path(project_root, str(paths_payload["predictions_csv"])),
            eval_report_csv=_resolve_path(project_root, str(paths_payload["eval_report_csv"])),
        ),
        train_ratio=train_ratio,
        anomaly_sigma_k=anomaly_sigma_k,
        eval_policy_override=eval_policy_override,
    )
