"""Storing a computed result: whether to, whole or in chunks, and the
lineage tag it carries."""

from __future__ import annotations

STORE_FAILED_FIX = (
    "read the exception: a full disk, a cache_dir you cannot write to, or a "
    "value that cannot be pickled -- return the data, not the handle that "
    "produced it."
)


#: Result types seen to refuse an attribute (dict, list, ndarray, ...): not
#: tried again (`Cash._attach_lineage`).
UNTAGGABLE_TYPES: set[type] = set()
