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
    adapter: str | None
    adapter_ref: str | None
    model: str | None
    model_ref: str | None
    paths: PathConfig
    train_ratio: float
    split_shuffle: bool
    split_seed: int | None
    anomaly_sigma_k: float
    infer_chunk_size: int
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
    infer_payload = payload.get("infer", {})
    eval_policy_payload = payload.get("eval_policy", {})

    train_ratio = float(split_payload.get("train_ratio", 0.8))
    if not 0 < train_ratio < 1:
        raise ValueError("split.train_ratio must be between 0 and 1")

    raw_split_shuffle = split_payload.get("shuffle", False)
    if not isinstance(raw_split_shuffle, bool):
        raise ValueError("split.shuffle must be a boolean")
    split_shuffle = raw_split_shuffle

    raw_split_seed = split_payload.get("seed", 42)
    if raw_split_seed is None:
        split_seed: int | None = None
    else:
        split_seed = int(raw_split_seed)

    anomaly_sigma_k = float(anomaly_payload.get("sigma_k", 3.0))
    if anomaly_sigma_k <= 0:
        raise ValueError("anomaly.sigma_k must be > 0")

    infer_chunk_size = int(infer_payload.get("chunk_size", 10000))
    if infer_chunk_size <= 0:
        raise ValueError("infer.chunk_size must be > 0")

    eval_policy_override: EvalPolicy = {
        "val_mae_max": float(eval_policy_payload.get("val_mae_max", 0.1)),
        "val_rmse_max": float(eval_policy_payload.get("val_rmse_max", 0.1)),
        "val_maxae_max": float(eval_policy_payload.get("val_maxae_max", 0.2)),
    }

    raw_adapter = payload.get("adapter")
    raw_adapter_ref = payload.get("adapter_ref")
    adapter = str(raw_adapter) if raw_adapter is not None else None
    adapter_ref = str(raw_adapter_ref) if raw_adapter_ref is not None else None
    if adapter is None and adapter_ref is None:
        raise ValueError("config must provide either 'adapter' or 'adapter_ref'")

    raw_model = payload.get("model")
    raw_model_ref = payload.get("model_ref")
    model = str(raw_model) if raw_model is not None else None
    model_ref = str(raw_model_ref) if raw_model_ref is not None else None
    if model is None and model_ref is None:
        raise ValueError("config must provide either 'model' or 'model_ref'")

    return RuntimeConfig(
        task=str(payload["task"]),
        adapter=adapter,
        adapter_ref=adapter_ref,
        model=model,
        model_ref=model_ref,
        paths=PathConfig(
            train_csv=_resolve_path(project_root, str(paths_payload["train_csv"])),
            infer_csv=_resolve_path(project_root, str(paths_payload["infer_csv"])),
            predictions_csv=_resolve_path(project_root, str(paths_payload["predictions_csv"])),
            eval_report_csv=_resolve_path(project_root, str(paths_payload["eval_report_csv"])),
        ),
        train_ratio=train_ratio,
        split_shuffle=split_shuffle,
        split_seed=split_seed,
        anomaly_sigma_k=anomaly_sigma_k,
        infer_chunk_size=infer_chunk_size,
        eval_policy_override=eval_policy_override,
    )
