from typing import Iterable, Protocol

from forgeflow.interfaces.types import (
    AdapterStats,
    FeatureMatrix,
    FeatureStats,
    FeatureVector,
    Record,
    State,
)


class SupervisedAdapter(Protocol):
    name: str
    feature_names: tuple[str, ...]
    target_names: tuple[str, ...]

    def to_state(self, record: Record) -> State:
        ...

    def adapt_records(self, records: Iterable[Record]) -> tuple[list[State], AdapterStats]:
        ...

    def to_feature_vector(self, state: State) -> FeatureVector:
        ...

    def build_feature_matrix(self, states: Iterable[State]) -> tuple[FeatureMatrix, FeatureStats]:
        ...

    def to_target_vector(self, state: State) -> FeatureVector:
        ...

    def build_target_matrix(self, states: Iterable[State]) -> tuple[FeatureMatrix, FeatureStats]:
        ...

    def to_infer_feature_vector(self, record: Record) -> FeatureVector:
        ...

    def build_infer_feature_matrix(self, records: Iterable[Record]) -> tuple[FeatureMatrix, FeatureStats]:
        ...


def ensure_required_fields(record: Record, required_fields: tuple[str, ...]) -> None:
    missing_fields = [field for field in required_fields if field not in record]
    if missing_fields:
        missing = ", ".join(missing_fields)
        raise KeyError(f"missing required fields: {missing}")
