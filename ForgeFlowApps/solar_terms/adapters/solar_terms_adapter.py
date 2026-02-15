from datetime import date
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


class SolarTermsAdapter:
    """Adapter for Beijing Mangzhong window weather records."""

    name = "solar_terms_v1"
    required_fields = (
        "date",
        "year",
        "day_offset_to_mangzhong",
        "temp_max",
        "temp_min",
        "wind_level",
        "temp_mean",
    )
    infer_required_fields = (
        "date",
        "year",
        "day_offset_to_mangzhong",
        "temp_max",
        "temp_min",
        "wind_level",
    )
    feature_names = (
        "day_offset_to_mangzhong",
        "temp_max",
        "temp_min",
        "wind_level",
        "year",
        "day_of_year",
    )
    target_names = ("temp_mean",)

    def _feature_payload_from_record(self, record: Record) -> dict[str, float]:
        ensure_required_fields(record, self.infer_required_fields)
        dt = date.fromisoformat(str(record["date"]))
        return {
            "day_offset_to_mangzhong": float(record["day_offset_to_mangzhong"]),
            "temp_max": float(record["temp_max"]),
            "temp_min": float(record["temp_min"]),
            "wind_level": float(record["wind_level"]),
            "year": float(record["year"]),
            "day_of_year": float(dt.timetuple().tm_yday),
        }

    def to_state(self, record: Record) -> State:
        ensure_required_fields(record, self.required_fields)
        payload = self._feature_payload_from_record(record)
        payload["temp_mean"] = float(record["temp_mean"])
        return payload

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
        payload = self._feature_payload_from_record(record)
        return [float(payload[name]) for name in self.feature_names]

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
