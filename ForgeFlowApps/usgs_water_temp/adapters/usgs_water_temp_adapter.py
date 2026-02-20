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


class USGSWaterTempAdapter:
    name = "usgs_water_temp"
    required_fields = ("temp_t", "doy_sin", "doy_cos", "dt_days", "y")
    feature_names = ("temp_t", "doy_sin", "doy_cos", "dt_days")
    target_names = ("y",)
    infer_required_fields = feature_names

    def to_state(self, record: Record) -> State:
        ensure_required_fields(record, self.required_fields)
        return {
            "temp_t": float(record["temp_t"]),
            "doy_sin": float(record["doy_sin"]),
            "doy_cos": float(record["doy_cos"]),
            "dt_days": float(record["dt_days"]),
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


class USGSWaterTempTempOnlyAdapter:
    name = "usgs_water_temp_temp_only"
    required_fields = ("temp_t", "y")
    feature_names = ("temp_t",)
    target_names = ("y",)
    infer_required_fields = feature_names

    def to_state(self, record: Record) -> State:
        ensure_required_fields(record, self.required_fields)
        return {
            "temp_t": float(record["temp_t"]),
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
