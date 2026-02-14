import math

from forgeflow.interfaces import FeatureMatrix, RegressionMetrics


def compute_regression_metrics(
    y_true: FeatureMatrix, y_pred: FeatureMatrix
) -> RegressionMetrics:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same number of samples")
    if not y_true:
        raise ValueError("y_true and y_pred must contain at least one sample")

    abs_errors: list[float] = []
    squared_errors: list[float] = []

    for true_row, pred_row in zip(y_true, y_pred):
        if len(true_row) != len(pred_row):
            raise ValueError("target row and prediction row must have same width")

        for true_value, pred_value in zip(true_row, pred_row):
            error = float(true_value) - float(pred_value)
            abs_errors.append(abs(error))
            squared_errors.append(error * error)

    n_values = len(abs_errors)
    mae = sum(abs_errors) / n_values
    mse = sum(squared_errors) / n_values
    rmse = math.sqrt(mse)
    maxae = max(abs_errors)

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "maxae": float(maxae),
    }
