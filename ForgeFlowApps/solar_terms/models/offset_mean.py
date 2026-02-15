from collections import defaultdict

from forgeflow.interfaces import FeatureMatrix


class SolarTermsOffsetMeanRegressor:
    """Predict temp_mean by historical mean on day-offset around Mangzhong."""

    def __init__(self) -> None:
        self._offset_mean: dict[int, float] = {}
        self._global_mean: float | None = None

    def fit(self, x: FeatureMatrix, y: FeatureMatrix) -> None:
        if len(x) != len(y):
            raise ValueError("x and y must have the same number of samples")
        if not x:
            raise ValueError("x and y must contain at least one sample")

        buckets: dict[int, list[float]] = defaultdict(list)
        all_targets: list[float] = []

        for x_row, y_row in zip(x, y):
            if not x_row or not y_row:
                raise ValueError("x and y rows must be non-empty")
            offset = int(round(float(x_row[0])))
            target = float(y_row[0])
            buckets[offset].append(target)
            all_targets.append(target)

        self._offset_mean = {
            offset: (sum(values) / len(values)) for offset, values in buckets.items()
        }
        self._global_mean = sum(all_targets) / len(all_targets)

    def predict(self, x: FeatureMatrix) -> FeatureMatrix:
        if self._global_mean is None:
            raise RuntimeError("model must be fitted before predict")

        predictions: FeatureMatrix = []
        for x_row in x:
            if not x_row:
                raise ValueError("x rows must be non-empty")
            offset = int(round(float(x_row[0])))
            y_hat = self._offset_mean.get(offset, self._global_mean)
            predictions.append([float(y_hat)])
        return predictions

    def coefficients(self) -> list[list[float]]:
        if self._global_mean is None:
            raise RuntimeError("model must be fitted before coefficients")
        # Runner expects [slope, intercept]-like shape for logging.
        return [[0.0], [float(self._global_mean)]]
