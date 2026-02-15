from typing import Iterable

from forgeflow.interfaces import (
    AdapterStats,
    FeatureMatrix,
    FeatureStats,
    FeatureVector,
    Record,
    State,
    ensure_required_fields,
)


class Poly4CubicAdapter:
    """Adapter for raw 4-variable polynomial samples."""

    name = "poly4_cubic_v1"
    required_fields = ("x1", "x2", "x3", "x4", "y")
    infer_required_fields = ("x1", "x2", "x3", "x4")
    feature_names = ("x1", "x2", "x3", "x4")
    target_names = ("y",)

    def to_state(self, record: Record) -> State:
        ensure_required_fields(record, self.required_fields)
        return {
            "x1": float(record["x1"]),
            "x2": float(record["x2"]),
            "x3": float(record["x3"]),
            "x4": float(record["x4"]),
            "y": float(record["y"]),
        }

    def adapt_records(self, records: Iterable[Record]) -> tuple[list[State], AdapterStats]:
        states: list[State] = []
        stats: AdapterStats = {
            "total_records": 0,
            "valid_states": 0,
            "skipped_state_rows": 0,
        }

        for record in records:
            stats["total_records"] += 1
            try:
                states.append(self.to_state(record))
                stats["valid_states"] += 1
            except (KeyError, TypeError, ValueError):
                stats["skipped_state_rows"] += 1

        return states, stats

    def to_feature_vector(self, state: State) -> FeatureVector:
        ensure_required_fields(state, self.feature_names)
        return [float(state[name]) for name in self.feature_names]

    def build_feature_matrix(self, states: Iterable[State]) -> tuple[FeatureMatrix, FeatureStats]:
        matrix: FeatureMatrix = []
        stats: FeatureStats = {
            "total_states": 0,
            "valid_vectors": 0,
            "skipped_feature_rows": 0,
            "n_features": len(self.feature_names),
        }

        for state in states:
            stats["total_states"] += 1
            try:
                matrix.append(self.to_feature_vector(state))
                stats["valid_vectors"] += 1
            except (KeyError, TypeError, ValueError):
                stats["skipped_feature_rows"] += 1

        return matrix, stats

    def to_target_vector(self, state: State) -> FeatureVector:
        ensure_required_fields(state, self.target_names)
        return [float(state["y"])]

    def build_target_matrix(self, states: Iterable[State]) -> tuple[FeatureMatrix, FeatureStats]:
        matrix: FeatureMatrix = []
        stats: FeatureStats = {
            "total_states": 0,
            "valid_vectors": 0,
            "skipped_feature_rows": 0,
            "n_features": len(self.target_names),
        }

        for state in states:
            stats["total_states"] += 1
            try:
                matrix.append(self.to_target_vector(state))
                stats["valid_vectors"] += 1
            except (KeyError, TypeError, ValueError):
                stats["skipped_feature_rows"] += 1

        return matrix, stats

    def to_infer_feature_vector(self, record: Record) -> FeatureVector:
        ensure_required_fields(record, self.infer_required_fields)
        return [float(record[name]) for name in self.feature_names]

    def build_infer_feature_matrix(self, records: Iterable[Record]) -> tuple[FeatureMatrix, FeatureStats]:
        matrix: FeatureMatrix = []
        stats: FeatureStats = {
            "total_states": 0,
            "valid_vectors": 0,
            "skipped_feature_rows": 0,
            "n_features": len(self.feature_names),
        }

        for record in records:
            stats["total_states"] += 1
            try:
                matrix.append(self.to_infer_feature_vector(record))
                stats["valid_vectors"] += 1
            except (KeyError, TypeError, ValueError):
                stats["skipped_feature_rows"] += 1

        return matrix, stats
