from typing import Protocol

from forgeflow.interfaces.types import FeatureMatrix, State


class RegressionModel(Protocol):
    def fit(self, x: FeatureMatrix, y: FeatureMatrix) -> None:
        ...

    def predict(self, x: FeatureMatrix) -> FeatureMatrix:
        ...

    def coefficients(self) -> list[list[float]]:
        ...


class SimulatorModel(Protocol):
    def simulate(self, initial_state: State, simulation: dict[str, object]) -> list[State]:
        ...
