"""What restoring a variable from a cache entry records about it.

A value comes back from the cache two ways: the statement restorer hydrates
a statement's outputs on a hit, and the variable restorer brings one
upstream variable back for a cell that needs it. Either way the value is
exactly the entry's value, so :func:`apply_restored_var` records it the same
way: its lineage, what it was built from, the statement that produced it, the
files it depends on, and its session hash.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cash.notebook._protocols import TrackingState
    from cash.notebook.statement._metadata import StatementCacheMetadata

logger = logging.getLogger(__name__)

__all__ = ["apply_restored_var"]

#: Values whose session hash is their entry's lineage rather than a content
#: hash: hashing a large frame or array on every restore costs more than the
#: restore, and the lineage identifies the value just as well.
_LINEAGE_HASHED_TYPES = frozenset({"DataFrame", "Series", "ndarray"})


def apply_restored_var(
    state: TrackingState,
    name: str,
    value: Any,
    metadata: StatementCacheMetadata | None,
    *,
    compute_hash: Callable[[Any], str] | None = None,
) -> None:
    """Record in *state* that *name* now holds *value*, restored from the
    entry *metadata* describes (the caller has already bound it)."""
    lineage = None
    if metadata is not None:
        lineage = (metadata.output_lineages or {}).get(name)
        if lineage is not None:
            state.lineage.record(name, lineage, value=value)
        # What the value was built from. Without it a restored value has no
        # provenance, and the check for "built on an input rebuilt since"
        # compares an empty record and passes.
        if metadata.input_lineages:
            state.executed_input_lineages[name] = dict(metadata.input_lineages)
        if metadata.source_hash:
            state.executed_cell_hashes.setdefault(name, set()).add(metadata.source_hash)
        if metadata.code:
            state.executed_cell_codes[name] = metadata.code
        # Exactly the entry's files, not those of whatever the name held before.
        if metadata.file_dependencies:
            state.executed_file_deps[name] = set(metadata.file_dependencies)
        if metadata.key is not None:
            state.variable_sources[name] = metadata.key
    _record_session_hash(state, name, value, lineage, compute_hash)


def _record_session_hash(
    state: TrackingState,
    name: str,
    value: Any,
    lineage: str | None,
    compute_hash: Callable[[Any], str] | None,
) -> None:
    if type(value).__name__ in _LINEAGE_HASHED_TYPES:
        value_hash = lineage
    elif compute_hash is not None:
        try:
            value_hash = compute_hash(value)
        except (TypeError, ValueError, AttributeError, RecursionError) as e:
            logger.debug("Could not hash restored variable '%s': %s", name, e)
            return
    else:
        return
    if value_hash:
        state.variable_hashes.setdefault(name, set()).add(value_hash)
        state.current_session_hashes[name] = value_hash
