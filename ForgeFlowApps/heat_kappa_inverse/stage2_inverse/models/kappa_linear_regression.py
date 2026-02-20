import numpy as np
from numpy.typing import NDArray

from forgeflow.interfaces import FeatureMatrix


class KappaLinearRegressor:
    """Least-squares baseline for kappa inversion."""

    def __init__(self) -> None:
        self._weights: NDArray[np.float64] | None = None

    def fit(self, x: FeatureMatrix, y: FeatureMatrix) -> None:
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        if x_arr.ndim != 2 or y_arr.ndim != 2:
            raise ValueError("x and y must be 2D arrays")
        if x_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("x and y must have the same number of samples")
        if x_arr.shape[0] == 0:
            raise ValueError("x and y must contain at least one sample")

        x_design = np.hstack([x_arr, np.ones((x_arr.shape[0], 1), dtype=float)])
        weights, *_ = np.linalg.lstsq(x_design, y_arr, rcond=None)
        self._weights = weights

    def predict(self, x: FeatureMatrix) -> FeatureMatrix:
        if self._weights is None:
            raise RuntimeError("model must be fitted before predict")
        x_arr = np.asarray(x, dtype=float)
        if x_arr.ndim != 2:
            raise ValueError("x must be a 2D array")
        x_design = np.hstack([x_arr, np.ones((x_arr.shape[0], 1), dtype=float)])
        predictions = x_design @ self._weights
        return predictions.tolist()

    def coefficients(self) -> list[list[float]]:
        if self._weights is None:
            raise RuntimeError("model must be fitted before coefficients")
        return self._weights.tolist()

    def summary(self) -> dict[str, int]:
        if self._weights is None:
            raise RuntimeError("model must be fitted before summary")
        return {
            "n_features": int(self._weights.shape[0] - 1),
            "n_targets": int(self._weights.shape[1]),
        }

