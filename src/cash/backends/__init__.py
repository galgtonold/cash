"""Pluggable cache storage backends."""

from __future__ import annotations

from ._base import CacheBackend, CacheMetadata, MetadataDict
from .cascading_backend import CascadingBackend
from .file_backend import FileBackend
from .memory_backend import InMemoryBackend
from .serialization import ParquetSerializer, PickleSerializer, Serializer, get_serializer
from .sqlite_backend import SQLiteBackend
from .tiered_backend import TieredBackend

# Optional backends are resolved on FIRST ATTRIBUTE ACCESS (PEP 562), not at
# import. Importing them eagerly costs ~1.7s for the redis client alone, on the
# ``import cash`` path that every kernel start and every test subprocess pays,
# and almost nobody configures a remote backend. ``factory.py`` already imports
# both function-locally; these names exist for the public API surface.
#
# The ``None`` contract is preserved: an absent dependency still yields
# ``cash.backends.RedisBackend is None`` rather than raising, so existing
# availability checks keep working.
_OPTIONAL_BACKENDS = {
    "RedisBackend": ".redis_backend",
    "S3Backend": ".s3_backend",
}


def __getattr__(name: str):
    module_name = _OPTIONAL_BACKENDS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        from importlib import import_module

        value = getattr(import_module(module_name, __name__), name)
    except ImportError:
        value = None
    globals()[name] = value  # resolve once; later reads skip this hook
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_OPTIONAL_BACKENDS))

__all__ = [
    'CacheBackend',
    'CacheMetadata',
    'MetadataDict',
    'InMemoryBackend',
    'FileBackend',
    'SQLiteBackend',
    'CascadingBackend',
    'TieredBackend',
    'RedisBackend',
    'S3Backend',
    'Serializer',
    'PickleSerializer',
    'ParquetSerializer',
    'get_serializer',
]
