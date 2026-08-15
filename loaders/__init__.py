"""Per-modality model loaders.

Submodules are imported lazily by `model_registry.get_loader()` so that the app
starts even when a heavy dependency (diffusers, torch, ...) is not installed.
"""

from loaders.common import (  # noqa: F401
    MissingDependencyError,
    ModelHandle,
    PlaygroundError,
    UnsupportedModelError,
    cache_summary,
    clear_cache,
)

__all__ = [
    "ModelHandle",
    "PlaygroundError",
    "UnsupportedModelError",
    "MissingDependencyError",
    "cache_summary",
    "clear_cache",
]
