from typing import Protocol

from forgeflow.interfaces.types import FeatureMatrix


class RegressionModel(Protocol):
    def fit(self, x: FeatureMatrix, y: FeatureMatrix) -> None:
        ...

    def predict(self, x: FeatureMatrix) -> FeatureMatrix:
        ...

    def coefficients(self) -> list[list[float]]:
        ...
