import csv
import importlib
import logging
import time
from pathlib import Path

from forgeflow.core.config.runtime import RuntimeConfig, load_runtime_config
from forgeflow.core.data.split import split_train_val
from forgeflow.core.io.csv_reader import read_csv_records, read_csv_records_in_chunks

logger = logging.getLogger(__name__)


def _load_class_from_ref(ref: str) -> type:
    if ":" in ref:
        module_name, class_name = ref.split(":", 1)
    else:
        module_name, _, class_name = ref.rpartition(".")
    if not module_name or not class_name:
        raise ValueError(f"invalid class reference: {ref}")

    module = importlib.import_module(module_name)
    cls = getattr(module, class_name, None)
    if cls is None:
        raise ValueError(f"class '{class_name}' not found in module '{module_name}'")
    if not isinstance(cls, type):
        raise ValueError(f"reference '{ref}' does not point to a class")
    return cls


def _resolve_adapter_class(runtime: RuntimeConfig) -> type:
    if runtime.adapter_ref is not None:
        return _load_class_from_ref(runtime.adapter_ref)
    if runtime.adapter is None:
        raise ValueError("adapter is not configured")
    from forgeflow.plugins.registry import ADAPTER_REGISTRY

    adapter_entry = ADAPTER_REGISTRY.get(runtime.adapter)
    if adapter_entry is None:
        raise ValueError(f"unknown adapter: {runtime.adapter}")
    if isinstance(adapter_entry, str):
        return _load_class_from_ref(adapter_entry)
    if isinstance(adapter_entry, type):
        return adapter_entry
    raise ValueError(f"invalid adapter registry entry type: {type(adapter_entry).__name__}")


def _resolve_model_class(runtime: RuntimeConfig) -> type:
    if runtime.model_ref is not None:
        return _load_class_from_ref(runtime.model_ref)
    if runtime.model is None:
        raise ValueError("model is not configured")
    from forgeflow.plugins.registry import MODEL_REGISTRY

    model_entry = MODEL_REGISTRY.get(runtime.model)
    if model_entry is None:
        raise ValueError(f"unknown model: {runtime.model}")
    if isinstance(model_entry, str):
        return _load_class_from_ref(model_entry)
    if isinstance(model_entry, type):
        return model_entry
    raise ValueError(f"invalid model registry entry type: {type(model_entry).__name__}")


def _format_model_summary(model: object) -> dict[str, str]:
    summary: dict[str, str] = {}
    summary_fn = getattr(model, "summary", None)
    if callable(summary_fn):
        raw_summary = summary_fn()
        if isinstance(raw_summary, dict):
            for key, value in raw_summary.items():
                summary[str(key)] = str(value)
            if summary:
                return summary

    coefficients_fn = getattr(model, "coefficients", None)
    if callable(coefficients_fn):
        coefficients = coefficients_fn()
        rows = len(coefficients)
        cols = len(coefficients[0]) if rows > 0 else 0
        summary["coeff_shape"] = f"{rows}x{cols}"
        if rows > 0 and cols > 0:
            preview_count = min(rows, 3)
            preview = ",".join(f"{float(coefficients[i][0]):.6f}" for i in range(preview_count))
            summary["coeff_preview_col0"] = preview
        return summary

    summary["details"] = "unavailable"
    return summary


def _require_path(path_value: Path | None, key_name: str) -> Path:
    if path_value is None:
        raise ValueError(f"paths.{key_name} is required for selected mode")
    return path_value


def _compute_grid_mass(grid: list[list[float]]) -> float:
    return float(sum(sum(float(cell) for cell in row) for row in grid))


def _run_supervised(
    runtime: RuntimeConfig, adapter_cls: type, model_cls: type, timings: dict[str, float]
) -> None:
    from forgeflow.core.evaluation.anomaly import ResidualSigmaRule
    from forgeflow.core.evaluation.metrics import compute_regression_metrics
    from forgeflow.core.evaluation.policy import evaluate_pass_fail, get_eval_policy

    stage_start = time.perf_counter()
    train_csv_path = _require_path(runtime.paths.train_csv, "train_csv")
    infer_csv_path = _require_path(runtime.paths.infer_csv, "infer_csv")
    predictions_csv_path = _require_path(runtime.paths.predictions_csv, "predictions_csv")
    eval_report_csv_path = _require_path(runtime.paths.eval_report_csv, "eval_report_csv")

    records, stats = read_csv_records(train_csv_path)
    adapter = adapter_cls()
    states, adapter_stats = adapter.adapt_records(records)
    feature_matrix, feature_stats = adapter.build_feature_matrix(states)
    target_matrix, target_stats = adapter.build_target_matrix(states)
    timings["train_data_ms"] = (time.perf_counter() - stage_start) * 1000.0

    if not feature_matrix or not target_matrix:
        logger.info("[demo:%s] samples=0", adapter.name)
        logger.info("[demo:%s] train_mae=nan", adapter.name)
        logger.info("[demo:%s] val_mae=nan", adapter.name)
        logger.info("[csv] valid_rows=%s", stats["valid_rows"])
        logger.info("[adapter:%s] valid_states=%s", adapter.name, adapter_stats["valid_states"])
        logger.info("[features:%s] valid_vectors=%s", adapter.name, feature_stats["valid_vectors"])
        logger.info("[targets:%s] valid_vectors=%s", adapter.name, target_stats["valid_vectors"])
        return

    stage_start = time.perf_counter()
    x_train, y_train, x_val, y_val, split_stats = split_train_val(
        feature_matrix,
        target_matrix,
        train_ratio=runtime.train_ratio,
        shuffle=runtime.split_shuffle,
        seed=runtime.split_seed,
    )

    model = model_cls()
    model.fit(x_train, y_train)

    train_predictions = model.predict(x_train)
    val_predictions = model.predict(x_val)
    train_metrics = compute_regression_metrics(y_train, train_predictions)
    val_metrics = compute_regression_metrics(y_val, val_predictions)
    val_residuals = [float(true[0] - pred[0]) for true, pred in zip(y_val, val_predictions)]
    anomaly_detector = ResidualSigmaRule(sigma_k=runtime.anomaly_sigma_k)
    anomaly_stats = anomaly_detector.fit(val_residuals)
    eval_policy = get_eval_policy(runtime.task)
    eval_policy.update(runtime.eval_policy_override)
    eval_result = evaluate_pass_fail(val_metrics, eval_policy)
    timings["train_eval_ms"] = (time.perf_counter() - stage_start) * 1000.0

    model_summary = _format_model_summary(model)

    logger.info("[demo:%s] samples=%s", adapter.name, len(feature_matrix))
    for key, value in model_summary.items():
        logger.info("[model:%s] %s=%s", runtime.task, key, value)
    logger.info("[demo:%s] train_mae=%.6f", adapter.name, train_metrics["mae"])
    logger.info("[demo:%s] val_mae=%.6f", adapter.name, val_metrics["mae"])
    logger.info("[demo:%s] val_rmse=%.6f", adapter.name, val_metrics["rmse"])
    logger.info("[demo:%s] val_maxae=%.6f", adapter.name, val_metrics["maxae"])
    logger.info("[eval] status=%s", eval_result["status"])
    logger.info("[anomaly] mu=%.6f", anomaly_stats["mu"])
    logger.info("[anomaly] sigma=%.6f", anomaly_stats["sigma"])
    logger.info("[anomaly] threshold=%.6f", anomaly_stats["threshold"])
    logger.info("[split] total_samples=%s", split_stats["total_samples"])
    logger.info("[split] train_samples=%s", split_stats["train_samples"])
    logger.info("[split] val_samples=%s", split_stats["val_samples"])
    logger.info("[split] shuffle=%s", runtime.split_shuffle)
    logger.info("[split] seed=%s", runtime.split_seed)
    logger.info(
        "%s",
        {
            "train_example": {
                "x": x_train[0],
                "y_true": y_train[0],
                "y_pred": train_predictions[0],
            },
            "val_example": {
                "x": x_val[0],
                "y_true": y_val[0],
                "y_pred": val_predictions[0],
            },
        },
    )

    stage_start = time.perf_counter()
    infer_csv_stats = {
        "total_data_rows": 0,
        "valid_rows": 0,
        "skipped_bad_rows": 0,
    }
    infer_stats = {
        "total_states": 0,
        "valid_vectors": 0,
        "skipped_feature_rows": 0,
        "n_features": len(getattr(adapter, "feature_names", ())),
    }
    infer_scored_rows = 0
    infer_anomaly_rows = 0

    predictions_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with predictions_csv_path.open("w", encoding="utf-8", newline="") as out_file:
        writer = csv.writer(out_file)
        writer.writerow(["x", "y_pred", "residual", "anomaly_flag"])
        for infer_records_chunk, chunk_csv_stats in read_csv_records_in_chunks(
            infer_csv_path, runtime.infer_chunk_size
        ):
            infer_csv_stats = chunk_csv_stats
            infer_matrix, chunk_infer_stats = adapter.build_infer_feature_matrix(infer_records_chunk)
            infer_stats["total_states"] += chunk_infer_stats["total_states"]
            infer_stats["valid_vectors"] += chunk_infer_stats["valid_vectors"]
            infer_stats["skipped_feature_rows"] += chunk_infer_stats["skipped_feature_rows"]
            infer_stats["n_features"] = chunk_infer_stats["n_features"]

            infer_predictions = model.predict(infer_matrix) if infer_matrix else []
            for record, feature_vector, prediction_vector in zip(
                infer_records_chunk, infer_matrix, infer_predictions
            ):
                y_pred = float(prediction_vector[0])
                residual_cell = ""
                anomaly_cell = ""
                if "y" in record and record["y"] != "":
                    residual = float(record["y"]) - y_pred
                    is_anomaly = anomaly_detector.is_anomaly(residual)
                    residual_cell = f"{residual:.6f}"
                    anomaly_cell = "1" if is_anomaly else "0"
                    infer_scored_rows += 1
                    if is_anomaly:
                        infer_anomaly_rows += 1

                writer.writerow([feature_vector[0], y_pred, residual_cell, anomaly_cell])
    timings["infer_ms"] = (time.perf_counter() - stage_start) * 1000.0

    logger.info("[infer] input_rows=%s", infer_csv_stats["valid_rows"])
    logger.info("[infer] valid_vectors=%s", infer_stats["valid_vectors"])
    logger.info("[infer] scored_rows=%s", infer_scored_rows)
    logger.info("[infer] anomaly_rows=%s", infer_anomaly_rows)
    logger.info("[infer] output_file=%s", predictions_csv_path)

    stage_start = time.perf_counter()
    eval_report_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with eval_report_csv_path.open("w", encoding="utf-8", newline="") as eval_file:
        writer = csv.DictWriter(
            eval_file,
            fieldnames=[
                "task",
                "mode",
                "split_shuffle",
                "split_seed",
                "train_samples",
                "val_samples",
                "train_mae",
                "val_mae",
                "val_rmse",
                "val_maxae",
                "val_mae_max",
                "val_rmse_max",
                "val_maxae_max",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "task": runtime.task,
                "mode": runtime.mode,
                "split_shuffle": str(runtime.split_shuffle),
                "split_seed": "" if runtime.split_seed is None else str(runtime.split_seed),
                "train_samples": split_stats["train_samples"],
                "val_samples": split_stats["val_samples"],
                "train_mae": f"{train_metrics['mae']:.6f}",
                "val_mae": f"{val_metrics['mae']:.6f}",
                "val_rmse": f"{val_metrics['rmse']:.6f}",
                "val_maxae": f"{val_metrics['maxae']:.6f}",
                "val_mae_max": f"{eval_policy['val_mae_max']:.6f}",
                "val_rmse_max": f"{eval_policy['val_rmse_max']:.6f}",
                "val_maxae_max": f"{eval_policy['val_maxae_max']:.6f}",
                "status": str(eval_result["status"]),
            }
        )
    timings["eval_report_ms"] = (time.perf_counter() - stage_start) * 1000.0
    logger.info("[eval] report_file=%s", eval_report_csv_path)

    logger.info("[csv] valid_rows=%s", stats["valid_rows"])
    logger.info("[adapter:%s] valid_states=%s", adapter.name, adapter_stats["valid_states"])
    logger.info("[features:%s] valid_vectors=%s", adapter.name, feature_stats["valid_vectors"])
    logger.info("[targets:%s] valid_vectors=%s", adapter.name, target_stats["valid_vectors"])


def _run_simulation(
    runtime: RuntimeConfig, adapter_cls: type, model_cls: type, timings: dict[str, float]
) -> None:
    stage_start = time.perf_counter()
    initial_csv_path = _require_path(runtime.paths.initial_csv, "initial_csv")
    trajectory_csv_path = _require_path(runtime.paths.trajectory_csv, "trajectory_csv")
    eval_report_csv_path = _require_path(runtime.paths.eval_report_csv, "eval_report_csv")

    records, csv_stats = read_csv_records(initial_csv_path)
    adapter = adapter_cls()
    build_initial_state_fn = getattr(adapter, "build_initial_state", None)
    if not callable(build_initial_state_fn):
        raise ValueError(f"adapter '{adapter_cls.__name__}' must implement build_initial_state()")
    initial_state, adapter_stats = build_initial_state_fn(records, runtime.simulation)
    timings["initial_data_ms"] = (time.perf_counter() - stage_start) * 1000.0

    stage_start = time.perf_counter()
    model = model_cls()
    simulate_fn = getattr(model, "simulate", None)
    if not callable(simulate_fn):
        raise ValueError(f"model '{model_cls.__name__}' must implement simulate()")
    model_summary = _format_model_summary(model)
    for key, value in model_summary.items():
        logger.info("[model:%s] %s=%s", runtime.task, key, value)
    trajectory_states = simulate_fn(initial_state, runtime.simulation)
    if not trajectory_states:
        raise ValueError("simulation returned no states")
    timings["simulate_ms"] = (time.perf_counter() - stage_start) * 1000.0

    stage_start = time.perf_counter()
    trajectory_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with trajectory_csv_path.open("w", encoding="utf-8", newline="") as trajectory_file:
        writer = csv.writer(trajectory_file)
        writer.writerow(["step", "x", "y", "h"])
        for state in trajectory_states:
            step = int(state.get("step", 0))
            grid = state.get("grid")
            if not isinstance(grid, list):
                raise ValueError("simulation state must include 'grid' as a 2D list")
            for y_idx, row in enumerate(grid):
                if not isinstance(row, list):
                    raise ValueError("simulation grid rows must be lists")
                for x_idx, cell in enumerate(row):
                    writer.writerow([step, x_idx, y_idx, float(cell)])
    timings["trajectory_ms"] = (time.perf_counter() - stage_start) * 1000.0

    initial_grid = trajectory_states[0].get("grid")
    final_grid = trajectory_states[-1].get("grid")
    if not isinstance(initial_grid, list) or not isinstance(final_grid, list):
        raise ValueError("simulation states must expose initial and final grid values")

    initial_mass = _compute_grid_mass(initial_grid)
    final_mass = _compute_grid_mass(final_grid)
    mass_delta_abs = abs(final_mass - initial_mass)

    cfl_limit = float(trajectory_states[-1].get("cfl_limit", float("nan")))
    stable_cfl = bool(trajectory_states[-1].get("stable_cfl", True))
    boundary = str(runtime.simulation["boundary"])
    mass_tolerance = float(runtime.simulation["mass_tolerance"])
    should_check_mass = boundary in {"neumann", "periodic"}
    mass_check = "checked" if should_check_mass else "skipped"
    mass_ok = (mass_delta_abs <= mass_tolerance) if should_check_mass else True
    status = "PASS" if stable_cfl and mass_ok else "FAIL"

    logger.info("[simulation:%s] initial_cells=%s", runtime.task, csv_stats["valid_rows"])
    logger.info("[simulation:%s] steps=%s", runtime.task, runtime.simulation["steps"])
    logger.info("[simulation:%s] boundary=%s", runtime.task, boundary)
    logger.info("[simulation:%s] stable_cfl=%s", runtime.task, stable_cfl)
    logger.info("[simulation:%s] cfl_limit=%.6f", runtime.task, cfl_limit)
    logger.info("[simulation:%s] mass_initial=%.6f", runtime.task, initial_mass)
    logger.info("[simulation:%s] mass_final=%.6f", runtime.task, final_mass)
    logger.info("[simulation:%s] mass_delta_abs=%.6f", runtime.task, mass_delta_abs)
    logger.info("[simulation:%s] mass_check=%s", runtime.task, mass_check)
    logger.info("[simulation:%s] status=%s", runtime.task, status)
    logger.info("[simulation] trajectory_file=%s", trajectory_csv_path)
    logger.info("[csv] valid_rows=%s", csv_stats["valid_rows"])
    logger.info("[adapter:%s] valid_states=%s", adapter.name, adapter_stats["valid_states"])

    stage_start = time.perf_counter()
    eval_report_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with eval_report_csv_path.open("w", encoding="utf-8", newline="") as eval_file:
        writer = csv.DictWriter(
            eval_file,
            fieldnames=[
                "task",
                "mode",
                "steps_requested",
                "states_recorded",
                "boundary",
                "dt",
                "dx",
                "dy",
                "kappa",
                "strict_cfl",
                "cfl_limit",
                "stable_cfl",
                "mass_initial",
                "mass_final",
                "mass_delta_abs",
                "mass_tolerance",
                "mass_check",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "task": runtime.task,
                "mode": runtime.mode,
                "steps_requested": runtime.simulation["steps"],
                "states_recorded": len(trajectory_states),
                "boundary": boundary,
                "dt": f"{float(runtime.simulation['dt']):.6f}",
                "dx": f"{float(runtime.simulation['dx']):.6f}",
                "dy": f"{float(runtime.simulation['dy']):.6f}",
                "kappa": f"{float(runtime.simulation['kappa']):.6f}",
                "strict_cfl": str(bool(runtime.simulation["strict_cfl"])),
                "cfl_limit": f"{cfl_limit:.6f}",
                "stable_cfl": str(stable_cfl),
                "mass_initial": f"{initial_mass:.6f}",
                "mass_final": f"{final_mass:.6f}",
                "mass_delta_abs": f"{mass_delta_abs:.6f}",
                "mass_tolerance": f"{mass_tolerance:.6f}",
                "mass_check": mass_check,
                "status": status,
            }
        )
    timings["eval_report_ms"] = (time.perf_counter() - stage_start) * 1000.0
    logger.info("[eval] report_file=%s", eval_report_csv_path)


def run_pipeline(config_path: Path, project_root: Path) -> None:
    timings: dict[str, float] = {}
    pipeline_start = time.perf_counter()
    stage_start = pipeline_start

    runtime = load_runtime_config(config_path, project_root)
    timings["config_ms"] = (time.perf_counter() - stage_start) * 1000.0
    logger.info("[config] file=%s", config_path)
    logger.info("[config] mode=%s", runtime.mode)
    logger.info("[config] adapter=%s", runtime.adapter if runtime.adapter is not None else "-")
    logger.info(
        "[config] adapter_ref=%s",
        runtime.adapter_ref if runtime.adapter_ref is not None else "-",
    )
    logger.info("[config] model=%s", runtime.model if runtime.model is not None else "-")
    logger.info(
        "[config] model_ref=%s",
        runtime.model_ref if runtime.model_ref is not None else "-",
    )

    stage_start = time.perf_counter()
    adapter_cls = _resolve_adapter_class(runtime)
    model_cls = _resolve_model_class(runtime)
    timings["registry_ms"] = (time.perf_counter() - stage_start) * 1000.0

    if runtime.mode == "supervised":
        _run_supervised(runtime, adapter_cls, model_cls, timings)
    elif runtime.mode == "simulation":
        _run_simulation(runtime, adapter_cls, model_cls, timings)
    else:
        raise ValueError(f"unsupported mode: {runtime.mode}")

    timings["total_ms"] = (time.perf_counter() - pipeline_start) * 1000.0
    logger.info(
        (
            "[timing] config_ms=%.2f registry_ms=%.2f train_data_ms=%.2f "
            "train_eval_ms=%.2f infer_ms=%.2f initial_data_ms=%.2f "
            "simulate_ms=%.2f trajectory_ms=%.2f eval_report_ms=%.2f total_ms=%.2f"
        ),
        timings.get("config_ms", 0.0),
        timings.get("registry_ms", 0.0),
        timings.get("train_data_ms", 0.0),
        timings.get("train_eval_ms", 0.0),
        timings.get("infer_ms", 0.0),
        timings.get("initial_data_ms", 0.0),
        timings.get("simulate_ms", 0.0),
        timings.get("trajectory_ms", 0.0),
        timings.get("eval_report_ms", 0.0),
        timings["total_ms"],
    )
