"""Shared infrastructure for every loader: model cache, device/dtype detection,
error types and temp-file helpers.

Nothing in this module imports torch/transformers/diffusers at import time — the
Space must start (and `py_compile`) even when the heavy ML stack is missing.
"""

from __future__ import annotations

import gc
import os
import tempfile
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class PlaygroundError(RuntimeError):
    """Base class for errors that are safe to render verbatim in the UI."""


class UnsupportedModelError(PlaygroundError):
    """Raised when a repo_id cannot be mapped onto a supported task/loader."""


class MissingDependencyError(PlaygroundError):
    """Raised when an optional library (diffusers, torch, ...) is not installed."""


def require(module_name: str, extra_hint: str = "") -> Any:
    """Import `module_name` lazily and turn ImportError into a UI-friendly error."""
    try:
        return __import__(module_name)
    except ImportError as exc:  # pragma: no cover - depends on environment
        hint = f" {extra_hint}" if extra_hint else ""
        raise MissingDependencyError(
            f"Module '{module_name}' is not installed in this Space.{hint} "
            f"Add it to requirements.txt and rebuild. ({exc})"
        ) from exc


# --------------------------------------------------------------------------- #
# Device / dtype
# --------------------------------------------------------------------------- #


def get_torch():
    return require("torch", "Pick a GPU hardware tier for this Space.")


def get_device() -> str:
    """Return 'cuda', 'mps' or 'cpu' — never raises when torch is absent."""
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_dtype(preferred: str = "auto"):
    """Map a UI dtype string onto a torch dtype (None => let the library decide)."""
    try:
        import torch
    except ImportError:
        return None

    table = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if preferred in table:
        return table[preferred]

    # "auto": half precision only pays off on CUDA.
    return torch.float16 if get_device() == "cuda" else torch.float32


def dtype_kwarg(transformers_module: Any) -> str:
    """`torch_dtype` was renamed to `dtype` in transformers v5 — pick the right one."""
    version = str(getattr(transformers_module, "__version__", "4"))
    try:
        major = int(version.split(".")[0])
    except ValueError:
        major = 4
    return "dtype" if major >= 5 else "torch_dtype"


def hf_token() -> Optional[str]:
    """Read the Hub token from the usual Space secrets, if present."""
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
        value = os.environ.get(var)
        if value:
            return value
    return None


# --------------------------------------------------------------------------- #
# Model handle + module-level cache
# --------------------------------------------------------------------------- #


@dataclass
class ModelHandle:
    """Everything `run_inference` needs, and nothing else."""

    repo_id: str
    task: str
    modality: str
    pipeline: Any
    meta: Dict[str, Any] = field(default_factory=dict)

    def describe(self) -> str:
        parts = [f"{self.repo_id} :: {self.task} ({self.modality})"]
        for key in ("device", "dtype", "cls"):
            if key in self.meta:
                parts.append(f"{key}={self.meta[key]}")
        return " | ".join(parts)


CacheKey = Tuple[Any, ...]

MAX_CACHED_MODELS = int(os.environ.get("PLAYGROUND_MAX_CACHED_MODELS", "2"))

# Module-level (i.e. process-global) weight cache: repeated calls with the same
# key reuse the already materialised pipeline instead of re-downloading weights.
MODEL_CACHE: "OrderedDict[CacheKey, ModelHandle]" = OrderedDict()
_CACHE_LOCK = threading.RLock()


def cached_load(key: CacheKey, factory: Callable[[], ModelHandle]) -> ModelHandle:
    """Return the cached handle for `key`, otherwise build it via `factory`.

    The factory runs outside the lock's critical section only in the sense that
    two different keys still serialise here; that is deliberate — loading two
    multi-GB pipelines concurrently is the fastest way to OOM a Space.
    """
    with _CACHE_LOCK:
        if key in MODEL_CACHE:
            MODEL_CACHE.move_to_end(key)
            return MODEL_CACHE[key]

        handle = factory()
        MODEL_CACHE[key] = handle
        MODEL_CACHE.move_to_end(key)

        while len(MODEL_CACHE) > max(1, MAX_CACHED_MODELS):
            _, evicted = MODEL_CACHE.popitem(last=False)
            _release(evicted)
        return handle


def _release(handle: ModelHandle) -> None:
    """Drop references to a pipeline and hand the memory back to the allocator."""
    try:
        handle.pipeline = None
    finally:
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def clear_cache() -> str:
    with _CACHE_LOCK:
        count = len(MODEL_CACHE)
        for handle in list(MODEL_CACHE.values()):
            _release(handle)
        MODEL_CACHE.clear()
    return f"Unloaded {count} model(s); freed cache."


def cache_summary() -> str:
    with _CACHE_LOCK:
        if not MODEL_CACHE:
            return "(no models loaded)"
        return "\n".join(f"- {h.describe()}" for h in MODEL_CACHE.values())


# --------------------------------------------------------------------------- #
# Misc helpers
# --------------------------------------------------------------------------- #


def temp_path(suffix: str) -> str:
    """Allocate a path in the Space's tmp dir (Gradio serves files from there)."""
    fd, path = tempfile.mkstemp(suffix=suffix, prefix="playground_")
    os.close(fd)
    return path


def resolve_seed(seed: Optional[int]):
    """Build a torch generator for a seed; -1/None means 'random'."""
    if seed is None or int(seed) < 0:
        return None, None
    try:
        import torch
    except ImportError:
        return None, int(seed)
    device = get_device()
    generator = torch.Generator(device="cpu" if device == "mps" else device)
    generator.manual_seed(int(seed))
    return generator, int(seed)
