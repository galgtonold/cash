"""Writing an index's labels in place, on every pandas the tests run on.

Up to pandas 3.0, ``index.array[i] = v`` wrote the label array itself.
pandas 3.1 hands ``Index.array`` (and ``Index.values`` of text labels)
out as a read-only view of it
(GH 38547: "Index.values and Index.array now return read-only arrays"),
and the same line raises ``ValueError: Cannot modify read-only array``.
The memory under the view is as writable as before, and the view made
writable again writes it, as a read-only numpy view made writable does:
the write the tests check cash against is still one code can make.
"""

from __future__ import annotations

from typing import Any


def label_array(index: Any) -> Any:
    """``index.array``, writable where pandas made it a read-only view."""
    return _unlocked(index.array)


def label_values(index: Any) -> Any:
    """``index.values`` where it is an extension array (text labels),
    writable where pandas made it a read-only view."""
    return _unlocked(index.values)


def _unlocked(array: Any) -> Any:
    if getattr(array, "_readonly", False):
        array._readonly = False
    return array
