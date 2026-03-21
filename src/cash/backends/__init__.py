"""Pluggable cache storage backends."""

from __future__ import annotations

from ._base import CacheBackend, CacheMetadata
from .cascading_backend import AsyncBackendWrapper, CascadingBackend
from .file_backend import FileBackend
from .memory_backend import InMemoryBackend
from .serialization import ParquetSerializer, PickleSerializer, Serializer, get_serializer
from .tiered_backend import TieredBackend

try:
    from .redis_backend import RedisBackend
except ImportError:
    RedisBackend = None

try:
    from .s3_backend import S3Backend
except ImportError:
    S3Backend = None

__all__ = [
    'CacheBackend',
    'CacheMetadata',
    'InMemoryBackend',
    'FileBackend',
    'CascadingBackend',
    'TieredBackend',
    'AsyncBackendWrapper',
    'RedisBackend',
    'S3Backend',
    'Serializer',
    'PickleSerializer',
    'ParquetSerializer',
    'get_serializer',
]
