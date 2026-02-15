import csv
import importlib
import logging
import time
from pathlib import Path

from forgeflow.core.config.runtime import RuntimeConfig, load_runtime_config
from forgeflow.core.data.split import split_train_val
from forgeflow.core.evaluation.anomaly import ResidualSigmaRule
from forgeflow.core.evaluation.metrics import compute_regression_metrics
from forgeflow.core.evaluation.policy import evaluate_pass_fail, get_eval_policy
from forgeflow.core.io.csv_reader import read_csv_records, read_csv_records_in_chunks
from forgeflow.plugins.registry import ADAPTER_REGISTRY, MODEL_REGISTRY

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
    adapter_cls = ADAPTER_REGISTRY.get(runtime.adapter)
    if adapter_cls is None:
        raise ValueError(f"unknown adapter: {runtime.adapter}")
    return adapter_cls


def _resolve_model_class(runtime: RuntimeConfig) -> type:
    if runtime.model_ref is not None:
        return _load_class_from_ref(runtime.model_ref)
    if runtime.model is None:
        raise ValueError("model is not configured")
    model_cls = MODEL_REGISTRY.get(runtime.model)
    if model_cls is None:
        raise ValueError(f"unknown model: {runtime.model}")
    return model_cls


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


def run_pipeline(config_path: Path, project_root: Path) -> None:
    timings: dict[str, float] = {}
    pipeline_start = time.perf_counter()
    stage_start = pipeline_start

    runtime = load_runtime_config(config_path, project_root)
    timings["config_ms"] = (time.perf_counter() - stage_start) * 1000.0
    logger.info("[config] file=%s", config_path)
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

    stage_start = time.perf_counter()
    records, stats = read_csv_records(runtime.paths.train_csv)
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
        timings["total_ms"] = (time.perf_counter() - pipeline_start) * 1000.0
        logger.info(
            "[timing] config_ms=%.2f registry_ms=%.2f train_data_ms=%.2f total_ms=%.2f",
            timings["config_ms"],
            timings["registry_ms"],
            timings["train_data_ms"],
            timings["total_ms"],
        )
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

    runtime.paths.predictions_csv.parent.mkdir(parents=True, exist_ok=True)
    with runtime.paths.predictions_csv.open("w", encoding="utf-8", newline="") as out_file:
        writer = csv.writer(out_file)
        writer.writerow(["x", "y_pred", "residual", "anomaly_flag"])
        for infer_records_chunk, chunk_csv_stats in read_csv_records_in_chunks(
            runtime.paths.infer_csv, runtime.infer_chunk_size
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
    logger.info("[infer] output_file=%s", runtime.paths.predictions_csv)

    stage_start = time.perf_counter()
    runtime.paths.eval_report_csv.parent.mkdir(parents=True, exist_ok=True)
    with runtime.paths.eval_report_csv.open("w", encoding="utf-8", newline="") as eval_file:
        writer = csv.DictWriter(
            eval_file,
            fieldnames=[
                "task",
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
    logger.info("[eval] report_file=%s", runtime.paths.eval_report_csv)

    logger.info("[csv] valid_rows=%s", stats["valid_rows"])
    logger.info("[adapter:%s] valid_states=%s", adapter.name, adapter_stats["valid_states"])
    logger.info("[features:%s] valid_vectors=%s", adapter.name, feature_stats["valid_vectors"])
    logger.info("[targets:%s] valid_vectors=%s", adapter.name, target_stats["valid_vectors"])
    timings["total_ms"] = (time.perf_counter() - pipeline_start) * 1000.0
    logger.info(
        (
            "[timing] config_ms=%.2f registry_ms=%.2f train_data_ms=%.2f "
            "train_eval_ms=%.2f infer_ms=%.2f eval_report_ms=%.2f total_ms=%.2f"
        ),
        timings["config_ms"],
        timings["registry_ms"],
        timings["train_data_ms"],
        timings["train_eval_ms"],
        timings["infer_ms"],
        timings["eval_report_ms"],
        timings["total_ms"],
    )
