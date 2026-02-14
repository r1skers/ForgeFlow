from forgeflow.interfaces import EvalPolicy, RegressionMetrics

DEFAULT_EVAL_POLICY: EvalPolicy = {
    "val_mae_max": 0.1,
    "val_rmse_max": 0.1,
    "val_maxae_max": 0.2,
}

EVAL_POLICY_BY_TASK: dict[str, EvalPolicy] = {
    "linear_xy": {
        "val_mae_max": 0.01,
        "val_rmse_max": 0.01,
        "val_maxae_max": 0.02,
    },
    "linear_xy_noisy": {
        "val_mae_max": 0.5,
        "val_rmse_max": 0.6,
        "val_maxae_max": 0.8,
    },
}


def get_eval_policy(task_name: str) -> EvalPolicy:
    policy = EVAL_POLICY_BY_TASK.get(task_name, DEFAULT_EVAL_POLICY)
    return dict(policy)


def evaluate_pass_fail(val_metrics: RegressionMetrics, policy: EvalPolicy) -> dict[str, object]:
    mae_pass = val_metrics["mae"] <= policy["val_mae_max"]
    rmse_pass = val_metrics["rmse"] <= policy["val_rmse_max"]
    maxae_pass = val_metrics["maxae"] <= policy["val_maxae_max"]
    status = "PASS" if mae_pass and rmse_pass and maxae_pass else "FAIL"
    return {
        "mae_pass": mae_pass,
        "rmse_pass": rmse_pass,
        "maxae_pass": maxae_pass,
        "status": status,
    }
