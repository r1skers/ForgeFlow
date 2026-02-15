import numpy as np

from forgeflow.interfaces import FeatureMatrix, RegressionMetrics


def compute_regression_metrics(
    y_true: FeatureMatrix, y_pred: FeatureMatrix
) -> RegressionMetrics:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same number of samples")
    if not y_true:
        raise ValueError("y_true and y_pred must contain at least one sample")

    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    if y_true_arr.ndim != 2 or y_pred_arr.ndim != 2:
        raise ValueError("y_true and y_pred must be 2D arrays")
    if y_true_arr.shape[1] != y_pred_arr.shape[1]:
        raise ValueError("target row and prediction row must have same width")

    errors = y_true_arr - y_pred_arr
    abs_errors = np.abs(errors)

    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    maxae = float(np.max(abs_errors))

    return {
        "mae": mae,
        "rmse": rmse,
        "maxae": maxae,
    }
