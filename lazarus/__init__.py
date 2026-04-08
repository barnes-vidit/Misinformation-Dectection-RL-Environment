"""Misinformation Detection Environment — OpenEnv hackathon submission."""

from .models import FactCheckAction, FactCheckObservation, FactCheckState

__all__ = [
    "FactCheckAction",
    "FactCheckObservation",
    "FactCheckState",
    "FactCheckEnvClient",
]


def __getattr__(name: str):
    """Lazy-load FactCheckEnvClient to avoid circular import on startup."""
    if name == "FactCheckEnvClient":
        from .client import FactCheckEnvClient  # noqa: PLC0415
        return FactCheckEnvClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
