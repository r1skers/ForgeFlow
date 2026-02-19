__all__ = ["run_verification"]


def __getattr__(name: str):
    if name == "run_verification":
        from forgeflow.core.verification.runner import run_verification

        return run_verification
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
