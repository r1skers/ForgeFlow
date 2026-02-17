import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from forgeflow.interfaces import EvalPolicy


@dataclass(frozen=True)
class PathConfig:
    train_csv: Path | None
    infer_csv: Path | None
    initial_csv: Path | None
    predictions_csv: Path | None
    trajectory_csv: Path | None
    eval_report_csv: Path | None


@dataclass(frozen=True)
class RuntimeConfig:
    mode: str
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
    simulation: dict[str, Any]


def _resolve_path(project_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return project_root / path


def _resolve_required_path(
    project_root: Path, paths_payload: dict[str, Any], key: str, mode: str
) -> Path:
    raw_path = paths_payload.get(key)
    if raw_path is None:
        raise ValueError(f"paths.{key} is required when mode='{mode}'")
    return _resolve_path(project_root, str(raw_path))


def _parse_simulation_config(payload: dict[str, Any]) -> dict[str, Any]:
    simulation_payload = payload.get("simulation", {})
    if not isinstance(simulation_payload, dict):
        raise ValueError("simulation must be an object")

    steps = int(simulation_payload.get("steps", 20))
    if steps <= 0:
        raise ValueError("simulation.steps must be > 0")

    dt = float(simulation_payload.get("dt", 0.1))
    if dt <= 0:
        raise ValueError("simulation.dt must be > 0")

    dx = float(simulation_payload.get("dx", 1.0))
    if dx <= 0:
        raise ValueError("simulation.dx must be > 0")

    dy = float(simulation_payload.get("dy", dx))
    if dy <= 0:
        raise ValueError("simulation.dy must be > 0")

    kappa = float(simulation_payload.get("kappa", 1.0))
    if kappa <= 0:
        raise ValueError("simulation.kappa must be > 0")

    boundary = str(simulation_payload.get("boundary", "neumann")).strip().lower()
    if boundary not in {"neumann", "periodic", "dirichlet0"}:
        raise ValueError("simulation.boundary must be one of: neumann, periodic, dirichlet0")

    raw_strict_cfl = simulation_payload.get("strict_cfl", True)
    if not isinstance(raw_strict_cfl, bool):
        raise ValueError("simulation.strict_cfl must be a boolean")

    mass_tolerance = float(simulation_payload.get("mass_tolerance", 1e-6))
    if mass_tolerance < 0:
        raise ValueError("simulation.mass_tolerance must be >= 0")

    grid_nx_raw = simulation_payload.get("grid_nx")
    grid_ny_raw = simulation_payload.get("grid_ny")
    grid_nx = None if grid_nx_raw is None else int(grid_nx_raw)
    grid_ny = None if grid_ny_raw is None else int(grid_ny_raw)
    if grid_nx is not None and grid_nx <= 0:
        raise ValueError("simulation.grid_nx must be > 0 when provided")
    if grid_ny is not None and grid_ny <= 0:
        raise ValueError("simulation.grid_ny must be > 0 when provided")

    return {
        "steps": steps,
        "dt": dt,
        "dx": dx,
        "dy": dy,
        "kappa": kappa,
        "boundary": boundary,
        "strict_cfl": raw_strict_cfl,
        "mass_tolerance": mass_tolerance,
        "grid_nx": grid_nx,
        "grid_ny": grid_ny,
    }


def load_runtime_config(config_path: Path, project_root: Path) -> RuntimeConfig:
    with config_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    mode = str(payload.get("mode", "supervised")).strip().lower()
    if mode not in {"supervised", "simulation"}:
        raise ValueError("mode must be either 'supervised' or 'simulation'")

    paths_payload = payload["paths"]
    if not isinstance(paths_payload, dict):
        raise ValueError("paths must be an object")

    split_payload = payload.get("split", {})
    anomaly_payload = payload.get("anomaly", {})
    infer_payload = payload.get("infer", {})
    eval_policy_payload = payload.get("eval_policy", {})
    simulation_config = _parse_simulation_config(payload)

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

    if mode == "supervised":
        train_csv = _resolve_required_path(project_root, paths_payload, "train_csv", mode)
        infer_csv = _resolve_required_path(project_root, paths_payload, "infer_csv", mode)
        predictions_csv = _resolve_required_path(project_root, paths_payload, "predictions_csv", mode)
        trajectory_csv = None
        initial_csv = None
    else:
        train_csv = None
        infer_csv = None
        predictions_csv = None
        trajectory_csv = _resolve_required_path(project_root, paths_payload, "trajectory_csv", mode)
        initial_csv = _resolve_required_path(project_root, paths_payload, "initial_csv", mode)

    eval_report_csv = _resolve_required_path(project_root, paths_payload, "eval_report_csv", mode)

    return RuntimeConfig(
        mode=mode,
        task=str(payload["task"]),
        adapter=adapter,
        adapter_ref=adapter_ref,
        model=model,
        model_ref=model_ref,
        paths=PathConfig(
            train_csv=train_csv,
            infer_csv=infer_csv,
            initial_csv=initial_csv,
            predictions_csv=predictions_csv,
            trajectory_csv=trajectory_csv,
            eval_report_csv=eval_report_csv,
        ),
        train_ratio=train_ratio,
        split_shuffle=split_shuffle,
        split_seed=split_seed,
        anomaly_sigma_k=anomaly_sigma_k,
        infer_chunk_size=infer_chunk_size,
        eval_policy_override=eval_policy_override,
        simulation=simulation_config,
    )
