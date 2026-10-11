"""What restoring a variable from a cache entry records about it.

A value comes back from the cache two ways: the statement restorer hydrates
a statement's outputs on a hit, and the upstream check restores an upstream
statement's outputs for a cell that needs them. Either way the value is
exactly the entry's value, so :func:`apply_restored_var` records it the same
way: its lineage, what it was built from, the statement that produced it, the
files it depends on, and its session hash.
"""

from __future__ import annotations

import itertools
import logging
import sys
import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from cash.content_hashers import builtin_hash_family

if TYPE_CHECKING:
    from cash.notebook.statement._metadata import StatementCacheMetadata
    from cash.notebook.tracking_state import TrackingState

logger = logging.getLogger(__name__)

__all__ = ["FORWARD_PROBE_PLACEHOLDER", "apply_held_var", "apply_restored_var", "hashed_by_lineage", "identity_digest"]

#: Stands in the namespace for a variable the upstream check's forward probe
#: found a current-cell cache hit for: a statement whose input is not in the
#: namespace is never looked up (``cacheability_decision._has_missing_lineage``),
#: so without it the hit that restores the variable could not happen. The
#: restore replaces it; it is no object of the notebook's, so nothing holding
#: it stops that hit.
FORWARD_PROBE_PLACEHOLDER = object()

#: Values whose session hash is their entry's lineage rather than a content
#: hash: hashing a large frame or array on every restore costs more than the
#: restore, and the lineage identifies the value just as well.
_LINEAGE_HASHED_TYPES = frozenset({"DataFrame", "Series", "ndarray"})
#: A collection with more items than this is lineage-hashed too.
_LINEAGE_HASHED_ITEMS = 200


def hashed_by_lineage(value: Any) -> bool:
    """Is *value*'s session hash its lineage rather than its content?

    A frame, an array or a table, and a collection of more than
    ``_LINEAGE_HASHED_ITEMS`` items or holding one of those: hashing it in
    full on every output and every restore would cost seconds a hit. No
    check compares such a value's content with its session hash: each one
    that would reads it as changed instead (fail closed), since a lineage
    never equals a content hash.
    """
    t = type(value)
    if t.__name__ in _LINEAGE_HASHED_TYPES or builtin_hash_family(t) is not None or _wraps_a_frame(t):
        return True
    artist = _artist_class()
    if artist is not None and isinstance(value, artist):
        return True
    if t in (list, tuple, dict, set, frozenset):
        if len(value) > _LINEAGE_HASHED_ITEMS:
            return True
        items = value.values() if t is dict else value
        return any(
            type(v).__name__ in _LINEAGE_HASHED_TYPES
            or builtin_hash_family(type(v)) is not None
            or (artist is not None and isinstance(v, artist))
            for v in items
        )
    return False


#: What makes each `identity_digest` one no other binding, in this process
#: or another, gets.
_IDENTITY_PREFIX = f"identity:{uuid.uuid4().hex}:"
_IDENTITY_COUNT = itertools.count()


def _wraps_a_frame(t: type) -> bool:
    """Is *t* one of pandas' own classes, such as a groupby, a rolling window
    or a resampler: an object over a frame, whose content hash pickles the
    frame whole. ``t = df.groupby('user_id')`` took 2.5 s after 0.006 s over
    a million rows."""
    return (getattr(t, "__module__", "") or "").startswith(("pandas.core.", "pandas.api.typing"))


def identity_digest(value: Any) -> str | None:
    """A digest for *value*, a matplotlib figure, axes or artist, that no
    other binding gets, or None for any other value.

    ``for g, ax in zip(groups, axes):`` hashed each axes in full -- a pickle
    of its whole figure -- per iteration, to key statements that draw on it
    and run every time anyway. Each binding is its own instead: a statement
    keyed by one never hits, as when it ran uncached. Not the object's
    ``id``: a later figure can be given the same one.
    """
    artist = _artist_class()
    if artist is None or not isinstance(value, artist):
        return None
    return f"{_IDENTITY_PREFIX}{next(_IDENTITY_COUNT)}"


def _artist_class() -> type | None:
    """``matplotlib.artist.Artist`` once matplotlib is loaded, else None.

    A figure, its axes and what is drawn on them are tied to the figure by
    identity, and a plot statement runs every time: its figure's content
    hash -- a pickle of the whole figure -- decided nothing, and was most
    of cash's own time in a plot cell. Hashed by lineage like a frame:
    a check that would compare its content reads it as changed.
    """
    module = sys.modules.get("matplotlib.artist")
    artist = getattr(module, "Artist", None)
    return artist if isinstance(artist, type) else None


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


def apply_held_var(
    state: TrackingState,
    name: str,
    value: Any,
    lineage: str,
    *,
    compute_hash: Callable[[Any], str] | None = None,
) -> None:
    """Record in *state* that *name*, a variable a statement's entry stores
    with its outputs because it holds one of their objects too
    (``StatementCacheMetadata.holders``), now has *lineage*
    (``lineage_formula.held_lineage``) and holds *value*.

    Its lineage alone moves on: the statement did not produce it, so what it
    was built from, its code and its files stay its own producer's. The same
    whether the statement ran or its entry was restored.
    """
    state.lineage.record(name, lineage, value=value)
    _record_session_hash(state, name, value, lineage, compute_hash)


def _record_session_hash(
    state: TrackingState,
    name: str,
    value: Any,
    lineage: str | None,
    compute_hash: Callable[[Any], str] | None,
) -> None:
    if hashed_by_lineage(value):
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
