"""Evaluation helpers with lazy exports to avoid eager optional deps."""

from __future__ import annotations

from typing import Any

__all__ = [
    "ResidualSigmaRule",
    "compute_regression_metrics",
    "get_eval_policy",
    "evaluate_pass_fail",
]


def __getattr__(name: str) -> Any:
    if name == "ResidualSigmaRule":
        from forgeflow.core.evaluation.anomaly import ResidualSigmaRule

        return ResidualSigmaRule
    if name == "compute_regression_metrics":
        from forgeflow.core.evaluation.metrics import compute_regression_metrics

        return compute_regression_metrics
    if name == "get_eval_policy":
        from forgeflow.core.evaluation.policy import get_eval_policy

        return get_eval_policy
    if name == "evaluate_pass_fail":
        from forgeflow.core.evaluation.policy import evaluate_pass_fail

        return evaluate_pass_fail
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
