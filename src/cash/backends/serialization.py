"""Serialization strategies for cache value persistence.

Provides `Serializer` (abstract base), `PickleSerializer` and
`ParquetSerializer`, and `get_serializer`, which picks between them.
"""

from __future__ import annotations

import io
import pickle
import sys
from abc import ABC, abstractmethod
from typing import Any

__all__ = ["Serializer", "PickleSerializer", "ParquetSerializer", "get_serializer"]


class Serializer(ABC):
    """Abstract base class for serializers."""

    @abstractmethod
    def serialize(self, data: Any) -> bytes:
        """Serialize data to bytes."""

    @abstractmethod
    def deserialize(self, data: bytes) -> Any:
        """Deserialize bytes to data."""


class PickleSerializer(Serializer):
    """Serializer using Python's built-in pickle module."""

    def serialize(self, data: Any) -> bytes:
        return pickle.dumps(data)

    def deserialize(self, data: bytes) -> Any:
        return pickle.loads(data)


class ParquetSerializer(Serializer):
    """Serializer for pandas DataFrames using Parquet.

    A hit must hand back the frame the call returned, so a frame Parquet
    cannot store as it is -- labels it would turn into strings, an index
    frequency it drops, a column it cannot convert -- is pickled instead.
    The bytes say which: a Parquet file starts with ``PAR1``, a pickle never
    does.
    """

    _MAGIC = b"PAR1"

    def serialize(self, data: Any) -> bytes:
        if not _parquet_keeps(data):
            return pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        buffer = io.BytesIO()
        try:
            # ``index=None``: a RangeIndex is stored as metadata and comes back
            # a RangeIndex; ``index=True`` wrote it out as a plain int64 column.
            data.to_parquet(buffer, index=None)
        except Exception:  # noqa: BLE001 -- any column pyarrow cannot convert
            return pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        return buffer.getvalue()

    def deserialize(self, data: bytes) -> Any:
        if not data.startswith(self._MAGIC):
            return pickle.loads(data)
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("ParquetSerializer requires pandas: pip install pandas") from exc
        buffer = io.BytesIO(data)
        return pd.read_parquet(buffer)


def _parquet_keeps(df: Any) -> bool:
    """Does a Parquet round trip give *df* back unchanged, as far as a cheap
    look can tell? Column labels must be unique strings (Parquet stores names
    as strings), and no index may carry a frequency (read back as None)."""
    try:
        columns = df.columns
        if getattr(columns, "nlevels", 1) != 1 or not columns.is_unique:
            return False
        if len(columns) == 0 or not all(isinstance(c, str) for c in columns):
            return False
        return getattr(df.index, "freq", None) is None
    except Exception:  # noqa: BLE001 -- an exotic frame: pickle it
        return False


def get_serializer(data: Any) -> Serializer:
    """Factory to get the appropriate serializer for the data.

    Asks ``sys.modules`` rather than importing pandas. This runs on EVERY
    store, and importing pandas just to ask "is this a DataFrame?" cost ~800ms
    the first time -- paid by the FIRST cached call in any process, even when
    the result is an int. Measured: a first call whose body was ``return n``
    took 731ms, essentially all of it module loading reached through here.

    Correct because a ``DataFrame`` cannot exist unless pandas is already
    imported: whoever built it imported pandas, and unpickling one imports it
    too. So pandas being absent from ``sys.modules`` answers the question for
    free, and when it IS present the lookup costs nothing.
    """
    pd = sys.modules.get("pandas")
    if pd is not None:
        # ``getattr(..., ())`` guards a partially-initialised pandas: isinstance
        # against an empty tuple is simply False rather than an AttributeError.
        if isinstance(data, getattr(pd, "DataFrame", ())):
            try:
                import pyarrow  # noqa: F401

                return ParquetSerializer()
            except ImportError:
                try:
                    import fastparquet  # noqa: F401

                    return ParquetSerializer()
                except ImportError:
                    pass
            # Fall back to pickle for DataFrames without parquet support
            return PickleSerializer()

    return PickleSerializer()
