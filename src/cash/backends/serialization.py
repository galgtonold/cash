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
    cannot store as it is (see `_parquet_keeps`) is pickled instead.
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


#: Most columns a frame stored as Parquet may have. Parquet pays a fixed cost
#: per column on write and read (about 40 us per column to read, twice that to
#: write) where pickle pays next to nothing: a disk hit on 200 rows x 20,000
#: columns took 2 s against 0.03 s for the pickle of the same frame.
PARQUET_MAX_COLUMNS = 100

#: Nullable pandas dtypes that come back from Parquet as they went in.
_MASKED_DTYPES = frozenset(
    {
        "Int8",
        "Int16",
        "Int32",
        "Int64",
        "UInt8",
        "UInt16",
        "UInt32",
        "UInt64",
        "Float32",
        "Float64",
        "boolean",
    }
)


def _dtype_keeps(dtype: Any) -> bool:
    """Does a column (or index) of *dtype* come back from Parquet as it went in?

    An allowlist, because what Parquet changes is decided by the dtype:
    object columns come back converted (lists and tuples as arrays, dicts
    as padded structs, UUIDs as bytes, ints with a None as floats),
    ``datetime64[s]`` comes back ``[ms]``, python-backed strings come back
    with a different missing value, and a categorical's categories may not
    survive. Numbers, booleans, datetimes finer than a second, and
    pyarrow-backed strings do.
    """
    kind = getattr(dtype, "kind", None)
    name = getattr(dtype, "name", "")
    if (type(dtype).__module__ or "").startswith("numpy"):
        if kind in ("b", "i", "u", "f"):
            return True
        if kind in ("M", "m"):
            return not name.endswith("[s]")
        return False
    type_name = type(dtype).__name__
    if type_name == "DatetimeTZDtype":
        return getattr(dtype, "unit", "s") != "s"
    if type_name == "StringDtype":
        return getattr(dtype, "storage", None) == "pyarrow"
    return name in _MASKED_DTYPES


def _label_keeps(name: Any) -> bool:
    """Parquet stores an axis name as a string: anything else changes type."""
    return name is None or isinstance(name, str)


def _parquet_keeps(df: Any) -> bool:
    """Does a Parquet round trip give *df* back unchanged?

    Decided from the frame's structure, never its size in rows: exactly a
    ``pandas.DataFrame`` (a subclass would come back as the base class,
    losing its type and attributes -- a GeoDataFrame as WKB bytes), at most
    `PARQUET_MAX_COLUMNS` unique string column labels, every column and the
    index of a dtype `_dtype_keeps`, string axis names, no index frequency,
    no ``attrs`` (stored as JSON: tuples come back as lists, int keys as
    strings) and default ``flags``. Anything else is pickled.
    """
    try:
        import pandas as pd

        if type(df) is not pd.DataFrame:
            return False
        columns = df.columns
        if getattr(columns, "nlevels", 1) != 1 or not columns.is_unique:
            return False
        if not 0 < len(columns) <= PARQUET_MAX_COLUMNS or not all(isinstance(c, str) for c in columns):
            return False
        # Labels are read back with the dtype pandas gives a list of strings
        # (object before pandas 3, str from it); an Index of another dtype
        # would come back changed.
        default_labels = pd.Index(["a"]).dtype
        if columns.dtype != default_labels or not _label_keeps(columns.name):
            return False
        index = df.index
        if getattr(index, "nlevels", 1) != 1 or getattr(index, "freq", None) is not None:
            return False
        if not _label_keeps(index.name):
            return False
        if not isinstance(index, pd.RangeIndex) and not _dtype_keeps(index.dtype):
            return False
        if df.attrs or not df.flags.allows_duplicate_labels:
            return False
        return all(_dtype_keeps(dtype) for dtype in df.dtypes)
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
                import pyarrow  # noqa: F401 - an availability probe

                return ParquetSerializer()
            except ImportError:
                try:
                    import fastparquet  # noqa: F401 - an availability probe

                    return ParquetSerializer()
                except ImportError:
                    pass
            # Fall back to pickle for DataFrames without parquet support
            return PickleSerializer()

    return PickleSerializer()
