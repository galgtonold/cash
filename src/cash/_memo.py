"""The one bounded memo, and the size of every memo that uses it.

For caches whose keys keep arriving for as long as a process runs -- one per
code object, per statement, per file -- so that a long session with many
redefined cells does not grow them without limit.

Full, a memo drops the entry used least recently. A memo that stops taking
entries once full becomes a permanent slow path for every key after the cap,
and one that is cleared when full throws the entries in use away with the
rest; dropping the least recently used keeps what the session is working with.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Generic, TypeVar

__all__ = [
    "ARGUMENTS",
    "CODE_OBJECTS",
    "FILE_DIGESTS",
    "FRAMES",
    "FROZEN_RESULTS",
    "MODULE_ANALYSES",
    "MODULE_READ_DIGESTS",
    "NOTEBOOK_FUNCTIONS",
    "PATCH_SITES",
    "READ_PATHS",
    "RECORDS",
    "RESULT_TYPES",
    "SOURCE_FILES",
    "STATE_LEDGERS",
    "LruMemo",
]

K = TypeVar("K")
V = TypeVar("V")


class LruMemo(Generic[K, V]):
    """A mapping of at most *maxsize* entries that drops the least recently
    used one to make room. Reading an entry (``get``, ``in`` does not count)
    marks it used.

    Safe to share between threads the way a dict is: an entry another thread
    drops between two steps of an operation is not an error.
    """

    def __init__(self, maxsize: int) -> None:
        if maxsize < 1:
            raise ValueError("maxsize must be at least 1")
        self.maxsize = maxsize
        self._data: OrderedDict[K, V] = OrderedDict()

    def get(self, key: K, default: V | None = None) -> V | None:
        data = self._data
        try:
            value = data[key]
        except KeyError:
            return default
        try:
            data.move_to_end(key)
        except KeyError:
            pass  # dropped by another thread meanwhile; the value still holds
        return value

    def __setitem__(self, key: K, value: V) -> None:
        data = self._data
        data[key] = value
        try:
            data.move_to_end(key)
        except KeyError:
            pass
        while len(data) > self.maxsize:
            try:
                data.popitem(last=False)
            except KeyError:
                break  # another thread emptied it

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def __len__(self) -> int:
        return len(self._data)

    def keys(self) -> list[K]:
        """The keys, least recently used first (a copy)."""
        return list(self._data)

    def pop(self, key: K, default: V | None = None) -> V | None:
        return self._data.pop(key, default)

    def clear(self) -> None:
        self._data.clear()


# -- Sizes -------------------------------------------------------------------
# Every bounded memo's size, named for what one entry is, with why it is that big.

#: One entry per function, class or code object: a notebook session defines
#: and redefines them freely, and 4096 is far more than one keeps alive.
CODE_OBJECTS = 4096

#: One entry per source file or module name met walking a stack: a kernel
#: with the scientific stack loaded has a few thousand modules.
SOURCE_FILES = 8192

#: One entry per path read outside a cached call: its resolution and stat.
READ_PATHS = 4096

#: One entry per module and pattern, or owner and name, that the file-read
#: patches go on: a fixed list of reader functions.
PATCH_SITES = 1024

#: One analysis per module file whose names are read; each holds the file's
#: top-level bindings.
MODULE_ANALYSES = 1024

#: One digest per module file and set of names read from it.
MODULE_READ_DIGESTS = 4096

#: One content digest per file version: a folder read runs to tens of
#: thousands of files, and an entry is a few hundred bytes.
FILE_DIGESTS = 1 << 17

#: One entry per argument object a cached call was given: a weakref and two
#: digests.
ARGUMENTS = 1024

#: One entry per pandas frame argument, each holding a shallow copy of it:
#: while it is held, pandas copies a block before writing to it in place.
FRAMES = 256

#: One entry per frozen=True result passed as an argument: its use count and
#: audit baseline. One dropped starts its audit schedule over.
FROZEN_RESULTS = 4096

#: One per result type that refuses an attribute: a class made per call
#: would otherwise be held for good.
RESULT_TYPES = 256

#: One stored-key record per cached function, as last read from disk.
RECORDS = 256

#: One key-build ledger per function and state, for `explain()`.
STATE_LEDGERS = 512

#: One source digest per function the notebook's cells reference.
NOTEBOOK_FUNCTIONS = 500
