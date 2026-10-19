"""A stored set iterates, when read back, in the order the stored one did.

A set's iteration order is its hash table's slot order, and a set rebuilt
from its elements -- unpickled from disk, copied by the RAM tier -- lands many
of them in other slots. A notebook's ``print(s)`` showed another order on a hit
than in the run that stored ``s``, and ``{x: i for i, x in enumerate(s)}``
numbered its elements differently, so whatever was built from those numbers
differed too.

The store records where each set's elements sit (`record`, in the entry's
metadata under `FIELD`, never in the pickle or the key), and a hit puts them
back there (`restore`); see ``_c/csets.c``. Only with the C extension, on the
CPythons whose set table it reads (3.12 to 3.14); elsewhere a hit keeps the
order the rebuild gives, as before. A set whose hashes differ in the reading
process (strings under another ``PYTHONHASHSEED``, where plain Python's own
order changes from run to run) keeps that order too.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

from .. import _native

logger = logging.getLogger(__name__)

__all__ = ["FIELD", "record", "restore"]

#: The metadata field the layouts are kept under.
FIELD = "set_layouts"

#: Objects a walk reads at most: a value bigger than that is not laid out.
MAX_NODES = 5_000_000

_KERNELS = (
    _native.kernels
    if _native.kernels is not None
    and hasattr(_native.kernels, "set_layouts")
    and (3, 12) <= sys.version_info[:2] <= (3, 14)
    else None
)


def record(value: Any, metadata: dict) -> None:
    """Note in *metadata* where *value*'s sets keep their elements: empty
    bytes when it holds none to lay out (or is too big to walk), so a later
    tier storing the same value does not walk it again."""
    if _KERNELS is None:
        return
    try:
        layouts = _KERNELS.set_layouts(value, MAX_NODES)
    except Exception:  # an aid: the value is stored either way
        logger.debug("could not record set layouts", exc_info=True)
        return
    metadata[FIELD] = layouts if layouts is not None else b""


def restore(value: Any, metadata: dict | None) -> tuple[int, int, int] | None:
    """Put *value*'s sets' elements where *metadata* says the stored ones had
    them: ``(laid out, in order already, left as the rebuild gave them)``, or
    None when there was nothing to do. Only the paths to the recorded sets
    are followed, not the whole value."""
    layouts = metadata.get(FIELD) if metadata else None
    if _KERNELS is None or not isinstance(layouts, bytes) or not layouts:
        return None
    try:
        return _KERNELS.apply_set_layouts(value, layouts, MAX_NODES)
    except Exception:  # damaged layouts, no memory: the order the rebuild gave stands
        logger.debug("could not lay out restored sets", exc_info=True)
        return None
