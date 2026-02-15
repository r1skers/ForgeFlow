from itertools import combinations_with_replacement

import numpy as np
from numpy.typing import NDArray

from forgeflow.interfaces import FeatureMatrix


class PolyLinearDeg3Regressor:
    """General m-variable polynomial linear regressor with degree=3."""

    def __init__(self) -> None:
        self._degree = 3
        self._weights: NDArray[np.float64] | None = None
        self._n_raw_features: int | None = None
        self._monomial_index: tuple[tuple[int, ...], ...] = ()

    def _build_monomial_index(self, n_features: int) -> tuple[tuple[int, ...], ...]:
        return tuple(
            combo
            for d in range(1, self._degree + 1)
            for combo in combinations_with_replacement(range(n_features), d)
        )

    def _expand(self, x_arr: NDArray[np.float64]) -> NDArray[np.float64]:
        if x_arr.ndim != 2:
            raise ValueError("x must be a 2D array")
        if self._n_raw_features is None:
            raise RuntimeError("model must be fitted before feature expansion")
        if x_arr.shape[1] != self._n_raw_features:
            raise ValueError("x feature width does not match fitted model")

        if not self._monomial_index:
            return np.empty((x_arr.shape[0], 0), dtype=float)

        columns: list[NDArray[np.float64]] = []
        for combo in self._monomial_index:
            col = np.prod(x_arr[:, combo], axis=1)
            columns.append(col)
        return np.column_stack(columns)

    def fit(self, x: FeatureMatrix, y: FeatureMatrix) -> None:
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        if x_arr.ndim != 2 or y_arr.ndim != 2:
            raise ValueError("x and y must be 2D arrays")
        if x_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("x and y must have the same number of samples")
        if x_arr.shape[0] == 0:
            raise ValueError("x and y must contain at least one sample")

        self._n_raw_features = int(x_arr.shape[1])
        self._monomial_index = self._build_monomial_index(self._n_raw_features)

        x_poly = self._expand(x_arr)
        x_design = np.hstack([x_poly, np.ones((x_arr.shape[0], 1), dtype=float)])
        weights, *_ = np.linalg.lstsq(x_design, y_arr, rcond=None)
        self._weights = weights

    def predict(self, x: FeatureMatrix) -> FeatureMatrix:
        if self._weights is None:
            raise RuntimeError("model must be fitted before predict")

        x_arr = np.asarray(x, dtype=float)
        x_poly = self._expand(x_arr)
        x_design = np.hstack([x_poly, np.ones((x_arr.shape[0], 1), dtype=float)])
        predictions = x_design @ self._weights
        return predictions.tolist()

    def coefficients(self) -> list[list[float]]:
        if self._weights is None:
            raise RuntimeError("model must be fitted before coefficients")
        return self._weights.tolist()

    def summary(self) -> dict[str, int]:
        if self._weights is None or self._n_raw_features is None:
            raise RuntimeError("model must be fitted before summary")
        return {
            "degree": int(self._degree),
            "n_raw_features": int(self._n_raw_features),
            "n_poly_terms": int(len(self._monomial_index)),
            "n_targets": int(self._weights.shape[1]),
        }
