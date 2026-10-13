"""Serialization strategies for cache value persistence.

Provides `Serializer` (abstract base) and `PickleSerializer`, which stores
every decorated result, pandas DataFrames included.
"""

from __future__ import annotations

import pickle
import sys
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from .. import kept_state

__all__ = ["Serializer", "PickleSerializer", "RESTORE_ERRORS", "rebuild", "restore_value"]

#: Buffers at least this large are handed out of band by `serialize_split`;
#: smaller ones stay in the stream, where a separate part costs more than the
#: copy it saves.
OUT_OF_BAND_MIN_BYTES = 64 * 1024


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

    A polars DataFrame or Series is pickled as its Arrow data
    (`_reduce_polars`), whose buffers protocol 5 hands out of band.
    """

    def serialize(self, data: Any) -> bytes:
        return kept_state.dumps(data, protocol=5, extra=polars_reducers())

    def deserialize(self, data: bytes) -> Any:
        return pickle.loads(data)

    def serialize_split(self, data: Any, *, copy: bool = True) -> tuple[bytes, list[Any]]:
        """The pickle stream and, apart from it, the large buffers it refers to.

        What the file backend writes (``entry_format.MAGIC_SPLIT``): a large
        array is copied once, here, instead of into the stream, then into a
        joined blob, then out of it. Copied because the write runs later, on
        another thread, and the caller may change the value in the meantime.

        ``copy=False`` hands each buffer out as a view of the value's own
        memory, for a value nobody changes until the write is done (the file
        backend's ``private`` values), where the copy would be pure cost.
        """
        buffers: list[Any] = []

        def take(buffer: pickle.PickleBuffer) -> bool:
            raw = buffer.raw()
            if raw.nbytes < OUT_OF_BAND_MIN_BYTES:
                return True  # in band
            buffers.append(raw.tobytes() if copy else raw)
            return False

        stream = kept_state.dumps(
            data, protocol=5, buffer_callback=take, on_retry=buffers.clear, extra=polars_reducers()
        )
        return stream, buffers

    def deserialize_split(self, stream: Any, buffers: list) -> Any:
        """The value `serialize_split` split, using *buffers*' memory as it is."""
        return pickle.loads(stream, buffers=buffers)


#: What turning stored bytes back into an object raises when the entry is
#: damaged (truncated by a killed process or a full disk, overwritten) or
#: names a binding this process lacks (a class renamed, a module gone, a
#: ``__main__`` class from another run). Every backend reads such an entry as
#: a miss: the value is recomputed, never a reason to fail the caller.
RESTORE_ERRORS: tuple[type[BaseException], ...] = (
    pickle.PickleError,
    EOFError,
    ValueError,
    TypeError,
    KeyError,
    IndexError,
    AttributeError,
    ImportError,
    OverflowError,
)


def restore_value(metadata: dict, payload: bytes) -> Any:
    """The value *payload* holds, rebuilt by the serializer *metadata* names.

    Raises one of `RESTORE_ERRORS` when it cannot be rebuilt.
    """
    serializer_cls = metadata.get("serializer_cls", PickleSerializer)
    return rebuild(serializer_cls().deserialize, payload)


def rebuild(load: Callable[..., Any], *args: Any) -> Any:
    """``load(*args)``, raising one of `RESTORE_ERRORS` for anything it raises.

    Unpickling runs the value's own code (``__setstate__``, a ``__reduce__``
    callable), which raises what it likes: torch's ``RuntimeError`` for a
    CUDA tensor read on a machine without a GPU, a library's own error class,
    ``RecursionError``. Such an entry cannot be restored HERE, which is a miss
    like a damaged one, not a crash of every call that reads it.
    """
    try:
        return load(*args)
    except RESTORE_ERRORS:
        raise
    except Exception as exc:
        raise pickle.UnpicklingError(f"rebuilding the stored value raised {type(exc).__name__}: {exc}") from exc


def polars_reducers() -> dict | None:
    """`_reduce_polars` for polars' DataFrame and Series, when polars is
    imported (a value cannot hold a frame before). Exact types: a subclass
    keeps its own reduce."""
    pl = sys.modules.get("polars")
    frame, series = getattr(pl, "DataFrame", None), getattr(pl, "Series", None)
    if not (isinstance(frame, type) and isinstance(series, type)):
        return None  # no polars, or one still importing
    return {frame: _reduce_polars, series: _reduce_polars}


def _reduce_polars(value: Any) -> Any:
    """A polars DataFrame or Series as its Arrow data, rebuilt by
    `_polars_from_arrow`.

    Polars' own pickle writes the frame as one IPC blob inside the stream:
    for 80 MB, 190 ms to write and 90 ms to read. An Arrow table shares the
    frame's buffers, and protocol 5 hands them to the file tier apart from
    the stream, as it does a pandas frame's. Polars' own pickle stays for
    what Arrow cannot take: a dtype this pyarrow lacks, no pyarrow at all.
    An ``Object`` column is refused with `pickle.PicklingError`, so the
    value stays in RAM: polars refuses it too, polars 1 with a panic that
    no ``except Exception`` stops.
    """
    import polars as pl

    flags = value.flags
    if isinstance(value, pl.Series):
        dtype = value.dtype
        dtypes = [dtype]
    else:
        dtype = value.schema
        dtypes = list(dtype.values())
    if any(d == pl.Object for d in dtypes):
        raise pickle.PicklingError("polars does not pickle an Object column")
    try:
        arrow = value.to_arrow(compat_level=pl.CompatLevel.newest())
    except Exception:  # noqa: BLE001 - no pyarrow, a polars without CompatLevel, a dtype Arrow lacks
        return value.__reduce_ex__(5)
    name = value.name if isinstance(value, pl.Series) else None
    return _polars_from_arrow, (arrow, name, dtype, flags)


def _polars_from_arrow(arrow: Any, name: str | None, dtype: Any, flags: dict) -> Any:
    """The polars value `_reduce_polars` stored: from *arrow*, without a
    copy, cast back to its polars *dtype* (a schema for a DataFrame) where
    Arrow has none of its own (an Enum comes back a Categorical), and
    marked sorted where it was (*flags*, ``value.flags``)."""
    import polars as pl

    out = pl.from_arrow(arrow, rechunk=False)
    if name is None:
        if out.schema != dtype:
            out = out.cast(dict(dtype))
        sorted_columns = [_sorted(column, flags[column.name]) for column in out.get_columns()]
        if any(new is not old for new, old in zip(sorted_columns, out.get_columns())):
            out = pl.DataFrame(sorted_columns)
        return out
    if not isinstance(out, pl.Series):
        out = out.to_series()
    out = out.alias(name)
    if out.dtype != dtype:
        out = out.cast(dtype)
    return _sorted(out, flags)


def _sorted(series: Any, flags: dict) -> Any:
    """*series*, marked sorted as *flags* say it was."""
    if flags.get("SORTED_ASC"):
        return series.set_sorted()
    if flags.get("SORTED_DESC"):
        return series.set_sorted(descending=True)
    return series
