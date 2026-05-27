"""Device selection helpers for optional GPU acceleration."""

from __future__ import annotations

import os
from typing import Any


def get_preferred_device() -> str:
    """Return ``cuda`` when available unless AD_DEVICE overrides it.

    Environment variables:
    - ``AD_DEVICE=auto|cuda|cpu|cuda:0`` controls the preferred torch device.
    - ``AD_CUDA_DEVICE=0`` controls the CUDA index used by libraries that accept
      an integer GPU id.
    """

    requested = os.getenv("AD_DEVICE", "auto").strip().lower()
    if requested and requested not in {"auto", ""}:
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            cuda_idx = os.getenv("AD_CUDA_DEVICE")
            return f"cuda:{cuda_idx}" if cuda_idx not in (None, "") else "cuda"
    except ImportError:
        pass
    return "cpu"


def cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def cuda_device_index(default: int = 0) -> int:
    raw = os.getenv("AD_CUDA_DEVICE")
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def maybe_add_supported_kwargs(cls: Any, kwargs: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    """Add constructor kwargs only when the target class advertises support."""

    try:
        import inspect

        params = inspect.signature(cls).parameters
    except (TypeError, ValueError):
        return kwargs
    out = dict(kwargs)
    for key, value in extra.items():
        if key in params and key not in out:
            out[key] = value
    return out
