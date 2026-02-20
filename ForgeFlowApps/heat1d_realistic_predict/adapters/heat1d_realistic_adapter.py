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


class Heat1DRealisticPredictAdapter:
    name = "heat1d_realistic_predict"
    required_fields = (
        "obs_center",
        "obs_left",
        "obs_right",
        "miss_center",
        "miss_left",
        "miss_right",
        "x_norm",
        "t_norm",
        "y",
    )
    feature_names = (
        "obs_center",
        "obs_left",
        "obs_right",
        "miss_center",
        "miss_left",
        "miss_right",
        "x_norm",
        "t_norm",
    )
    target_names = ("y",)
    infer_required_fields = feature_names

    def to_state(self, record: Record) -> State:
        ensure_required_fields(record, self.required_fields)
        return {
            "obs_center": float(record["obs_center"]),
            "obs_left": float(record["obs_left"]),
            "obs_right": float(record["obs_right"]),
            "miss_center": float(record["miss_center"]),
            "miss_left": float(record["miss_left"]),
            "miss_right": float(record["miss_right"]),
            "x_norm": float(record["x_norm"]),
            "t_norm": float(record["t_norm"]),
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
        return [float(state[name]) for name in self.target_names]

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

