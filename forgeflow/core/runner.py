import csv
from pathlib import Path

from forgeflow.core.config.runtime import RuntimeConfig, load_runtime_config
from forgeflow.core.data.split import split_train_val
from forgeflow.core.evaluation.anomaly import ResidualSigmaRule
from forgeflow.core.evaluation.metrics import compute_regression_metrics
from forgeflow.core.evaluation.policy import evaluate_pass_fail, get_eval_policy
from forgeflow.core.io.csv_reader import read_csv_records
from forgeflow.plugins.registry import ADAPTER_REGISTRY, MODEL_REGISTRY


def run_pipeline(config_path: Path, project_root: Path) -> None:
    runtime = load_runtime_config(config_path, project_root)
    print(f"[config] file={config_path}")

    adapter_cls = ADAPTER_REGISTRY.get(runtime.adapter)
    if adapter_cls is None:
        raise ValueError(f"unknown adapter: {runtime.adapter}")
    model_cls = MODEL_REGISTRY.get(runtime.model)
    if model_cls is None:
        raise ValueError(f"unknown model: {runtime.model}")

    records, stats = read_csv_records(runtime.paths.train_csv)
    adapter = adapter_cls()
    states, adapter_stats = adapter.adapt_records(records)
    feature_matrix, feature_stats = adapter.build_feature_matrix(states)
    target_matrix, target_stats = adapter.build_target_matrix(states)

    if not feature_matrix or not target_matrix:
        print(f"[demo:{adapter.name}] samples=0")
        print(f"[demo:{adapter.name}] slope=nan")
        print(f"[demo:{adapter.name}] intercept=nan")
        print(f"[demo:{adapter.name}] train_mae=nan")
        print(f"[demo:{adapter.name}] val_mae=nan")
        print(f"[csv] valid_rows={stats['valid_rows']}")
        print(f"[adapter:{adapter.name}] valid_states={adapter_stats['valid_states']}")
        print(f"[features:{adapter.name}] valid_vectors={feature_stats['valid_vectors']}")
        print(f"[targets:{adapter.name}] valid_vectors={target_stats['valid_vectors']}")
        return

    x_train, y_train, x_val, y_val, split_stats = split_train_val(
        feature_matrix, target_matrix, train_ratio=runtime.train_ratio
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

    coefficients = model.coefficients()
    slope = coefficients[0][0]
    intercept = coefficients[-1][0]

    print(f"[demo:{adapter.name}] samples={len(feature_matrix)}")
    print(f"[demo:{adapter.name}] slope={slope:.6f}")
    print(f"[demo:{adapter.name}] intercept={intercept:.6f}")
    print(f"[demo:{adapter.name}] train_mae={train_metrics['mae']:.6f}")
    print(f"[demo:{adapter.name}] val_mae={val_metrics['mae']:.6f}")
    print(f"[demo:{adapter.name}] val_rmse={val_metrics['rmse']:.6f}")
    print(f"[demo:{adapter.name}] val_maxae={val_metrics['maxae']:.6f}")
    print(f"[eval] status={eval_result['status']}")
    print(f"[anomaly] mu={anomaly_stats['mu']:.6f}")
    print(f"[anomaly] sigma={anomaly_stats['sigma']:.6f}")
    print(f"[anomaly] threshold={anomaly_stats['threshold']:.6f}")
    print(f"[split] total_samples={split_stats['total_samples']}")
    print(f"[split] train_samples={split_stats['train_samples']}")
    print(f"[split] val_samples={split_stats['val_samples']}")
    print(
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
        }
    )

    infer_records, infer_csv_stats = read_csv_records(runtime.paths.infer_csv)
    infer_matrix, infer_stats = adapter.build_infer_feature_matrix(infer_records)
    infer_predictions = model.predict(infer_matrix) if infer_matrix else []
    infer_scored_rows = 0
    infer_anomaly_rows = 0

    runtime.paths.predictions_csv.parent.mkdir(parents=True, exist_ok=True)
    with runtime.paths.predictions_csv.open("w", encoding="utf-8", newline="") as out_file:
        writer = csv.writer(out_file)
        writer.writerow(["x", "y_pred", "residual", "anomaly_flag"])
        for record, feature_vector, prediction_vector in zip(
            infer_records, infer_matrix, infer_predictions
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

    print(f"[infer] input_rows={infer_csv_stats['valid_rows']}")
    print(f"[infer] valid_vectors={infer_stats['valid_vectors']}")
    print(f"[infer] scored_rows={infer_scored_rows}")
    print(f"[infer] anomaly_rows={infer_anomaly_rows}")
    print(f"[infer] output_file={runtime.paths.predictions_csv}")

    runtime.paths.eval_report_csv.parent.mkdir(parents=True, exist_ok=True)
    with runtime.paths.eval_report_csv.open("w", encoding="utf-8", newline="") as eval_file:
        writer = csv.DictWriter(
            eval_file,
            fieldnames=[
                "task",
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
    print(f"[eval] report_file={runtime.paths.eval_report_csv}")

    print(f"[csv] valid_rows={stats['valid_rows']}")
    print(f"[adapter:{adapter.name}] valid_states={adapter_stats['valid_states']}")
    print(f"[features:{adapter.name}] valid_vectors={feature_stats['valid_vectors']}")
    print(f"[targets:{adapter.name}] valid_vectors={target_stats['valid_vectors']}")
