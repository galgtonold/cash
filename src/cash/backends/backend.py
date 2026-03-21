"""Pluggable cache storage backends -- re-export shim.

All concrete implementations have been split into dedicated modules:

* :mod:`cash.backends._base` -- :class:`CacheBackend` ABC and :class:`CacheMetadata`
* :mod:`cash.backends.memory_backend` -- :class:`InMemoryBackend`
* :mod:`cash.backends.file_backend` -- :class:`FileBackend`
* :mod:`cash.backends.cascading_backend` -- :class:`CascadingBackend`, :class:`AsyncBackendWrapper`

This module re-exports all public names so existing imports from
``cash.backends.backend`` continue to work unchanged.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "Importing from 'cash.backends.backend' is deprecated. "
    "Use 'cash.backends' directly instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export shared types
from ._base import CacheBackend, CacheMetadata
from .cascading_backend import AsyncBackendWrapper, CascadingBackend, _MultiBackendMixin
from .file_backend import FileBackend

# Re-export concrete implementations
from .memory_backend import InMemoryBackend

__all__ = [
    "CacheBackend",
    "CacheMetadata",
    "InMemoryBackend",
    "FileBackend",
    "_MultiBackendMixin",
    "CascadingBackend",
    "AsyncBackendWrapper",
]
