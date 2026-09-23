"""Pluggable cache storage backends."""

from __future__ import annotations

from importlib import import_module

from ._base import CacheBackend, CacheMetadata, MetadataDict
from .file_backend import FileBackend
from .memory_backend import InMemoryBackend
from .serialization import ParquetSerializer, PickleSerializer, Serializer, get_serializer
from .sqlite_backend import SQLiteBackend
from .tiered_backend import TieredBackend

# The remote backends are resolved on FIRST ATTRIBUTE ACCESS (PEP 562), not at
# import. Importing them eagerly costs ~1.7s for the redis client alone, on the
# ``import cash`` path that every kernel start and every test subprocess pays,
# and almost nobody configures a remote backend. Their modules import without
# the client library installed; constructing one without it raises
# ``DependencyNotFoundError``.
_OPTIONAL_BACKENDS = {
    "RedisBackend": ".redis_backend",
    "S3Backend": ".s3_backend",
}


def __getattr__(name: str):
    module_name = _OPTIONAL_BACKENDS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value  # resolve once; later reads skip this hook
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_OPTIONAL_BACKENDS))


__all__ = [
    "CacheBackend",
    "CacheMetadata",
    "MetadataDict",
    "InMemoryBackend",
    "FileBackend",
    "SQLiteBackend",
    "TieredBackend",
    "RedisBackend",  # noqa: F822 - served by the module __getattr__
    "S3Backend",  # noqa: F822 - served by the module __getattr__
    "Serializer",
    "PickleSerializer",
    "ParquetSerializer",
    "get_serializer",
]
