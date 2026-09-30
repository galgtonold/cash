"""Serialization strategies for cache value persistence.

Provides `Serializer` (abstract base) and `PickleSerializer`, which stores
every decorated result, pandas DataFrames included.
"""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from typing import Any

__all__ = ["Serializer", "PickleSerializer"]


class Serializer(ABC):
    """Abstract base class for serializers."""

    @abstractmethod
    def serialize(self, data: Any) -> bytes:
        """Serialize data to bytes."""

    @abstractmethod
    def deserialize(self, data: bytes) -> Any:
        """Deserialize bytes to data."""


class PickleSerializer(Serializer):
    """Serializer using Python's built-in pickle module.

    Protocol 5 writes large buffers (numpy arrays, DataFrame columns) in one
    piece: a 1M-row x 10-column float frame took 47 ms against 466 ms with
    the default protocol 4. It also stores DataFrames: Parquet was 5-40x
    slower to write and 2-20x slower to read, and changed what it could not
    store exactly (cell types, subclasses, ``attrs``). Its only gain was
    smaller files for low-variety columns, which the file backend's
    ``compress`` option recovers for most of them.
    """

    def serialize(self, data: Any) -> bytes:
        return pickle.dumps(data, protocol=5)

    def deserialize(self, data: bytes) -> Any:
        return pickle.loads(data)
