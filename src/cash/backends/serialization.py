"""Serialization strategies for cache value persistence.

Provides `Serializer` (abstract base) and `PickleSerializer`, which stores
every decorated result, pandas DataFrames included.
"""

from __future__ import annotations

import pickle
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
    """

    def serialize(self, data: Any) -> bytes:
        return kept_state.dumps(data, protocol=5)

    def deserialize(self, data: bytes) -> Any:
        return pickle.loads(data)

    def serialize_split(self, data: Any) -> tuple[bytes, list[bytes]]:
        """The pickle stream and, apart from it, the large buffers it refers to.

        What the file backend writes (``entry_format.MAGIC_SPLIT``): a large
        array is copied once, here, instead of into the stream, then into a
        joined blob, then out of it. Copied at all because the write runs
        later, on another thread, and the caller may change the value in the
        meantime.
        """
        buffers: list[bytes] = []

        def take(buffer: pickle.PickleBuffer) -> bool:
            raw = buffer.raw()
            if raw.nbytes < OUT_OF_BAND_MIN_BYTES:
                return True  # in band
            buffers.append(raw.tobytes())
            return False

        stream = kept_state.dumps(data, protocol=5, buffer_callback=take)
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
