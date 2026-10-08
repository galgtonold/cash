"""What the upstream simulation reads from the cache without restoring anything.

:class:`CacheProbe` answers the simulation's questions about the backend: an
entry's metadata, whether the files an entry recorded are still as they were,
and the small records the runtime leaves for an earlier kernel (a bare call's
mutation verdict, a loop's outcome and split point, what a ``from`` import
bound). It never loads a value.
"""

from __future__ import annotations

import ast
import importlib.util
import logging
from collections.abc import Callable
from typing import Any

from ...tracking.file_dep_snapshot import snapshot_is_fresh
from .._protocols import CashInstanceProtocol
from ..cache_key import (
    carrier_advances_key,
    control_outcome_key,
    import_bindings_key,
    module_state_key,
    mutation_verdict_key,
    process_state_key,
)
from ..loop_split import is_split_half, loop_source_hash, store_for_backend
from ..run_memo import file_state_this_run, known_fresh_entry, note_fresh_entry, stats_this_run

__all__ = ["CacheProbe", "ControlOutcome"]

logger = logging.getLogger(__name__)

#: ``(entry lineages, lineages left, files read, their file component)`` of a
#: control structure's recorded run (see ``TrackingState.control_outcomes``).
ControlOutcome = tuple[dict[str, str], dict[str, str], frozenset[str], str]


class CacheProbe:
    """Metadata and file-freshness reads from the cache backend."""

    def __init__(self, cash_instance: CashInstanceProtocol | None) -> None:
        self.cash_instance = cash_instance
        #: Resolved on the first loop-split lookup; ``None`` means "not yet
        #: resolved", not "no splits". See :meth:`loop_split`.
        self._split_store = None
        #: :meth:`import_bindings` answers by statement; cleared by :meth:`reset`.
        self._import_bindings_memo: dict[str, dict[str, dict]] = {}
        #: :meth:`carrier_advances` answers by statement; cleared by :meth:`reset`.
        self._carrier_advances_memo: dict[str, frozenset[str] | None] = {}
        #: :meth:`module_state` answers by statement; cleared by :meth:`reset`.
        self._module_state_memo: dict[str, tuple[frozenset[str], frozenset[str]] | None] = {}
        self._process_state_memo: dict[str, frozenset[str] | None] = {}

    def reset(self) -> None:
        """Forget the memoized import bindings and generator draws."""
        self._import_bindings_memo.clear()
        self._carrier_advances_memo.clear()
        self._module_state_memo.clear()
        self._process_state_memo.clear()

    def backend(self):
        """The cache backend the simulation probes, or None without a Cash."""
        return self.cash_instance.backend if self.cash_instance else None

    def metadata(self, cache_key: str) -> dict | None:
        """The metadata of the entry under *cache_key*, without its value.

        ``backend.get_metadata()`` skips deserializing a large cached object
        (a DataFrame) when only its metadata is needed. Errors propagate.
        """
        backend = self.backend()
        if backend is None:
            return None
        return backend.get_metadata(cache_key)

    def record(self, key: str) -> dict | None:
        """The metadata record under *key*, or ``None`` when there is none or
        it cannot be read."""
        backend = self.backend()
        if backend is None:
            return None
        try:
            return backend.get_metadata(key)
        except (OSError, TypeError, ValueError, AttributeError):
            return None

    @staticmethod
    def files_fresh(hist_files: dict[str, Any], memo_key: str | None = None) -> bool:
        """Return True if all historical file dependencies are still fresh.

        Each entry is ``{path: {'mtime': ..., 'size': ...}}``. When ``size``
        is recorded it is checked too — that catches rewrites within a
        single mtime tick on coarse-resolution filesystems (HFS+/APFS,
        some ext4 configs).

        *memo_key* -- the entry's cache key. A "fresh" verdict holds for the
        rest of the cell run (see :mod:`cash.notebook.run_memo`), because the
        simulation validates the same upstream entry for every statement of
        the cell.
        """
        if known_fresh_entry(memo_key):
            return True
        run = file_state_this_run()
        fresh, stale = snapshot_is_fresh(hist_files, run["memo"] if run is not None else None)
        if not fresh:
            logger.debug("[UPSTREAM] Forward prop failed: stale file dependency (%s)", stale)
            return False
        note_fresh_entry(memo_key)
        return True

    @staticmethod
    def stat_file_deps(hist_files: dict[str, float]) -> dict[str, float]:
        """Stat each path in *hist_files* and return ``{path: mtime}`` for existing files.

        Once per path per cell run (``run_memo.stats_this_run``), and from a directory
        listing where many share a directory."""
        return {p: st.st_mtime for p, (_resolved, st) in stats_this_run(hist_files).items() if st is not None}

    def mutation_verdict(self, source_hash: str) -> set[str] | None:
        """The runtime's verdict on a bare method call, from an earlier kernel.

        See ``mutation_verdict_key``.
        """
        record = self.record(mutation_verdict_key(source_hash))
        if not record or not record.get("mutation_verdict"):
            return None
        return set(record.get("receivers") or ())

    def module_state(self, source_hash: str) -> tuple[frozenset[str], frozenset[str]] | None:
        """``(names, modules)`` of the local modules a statement set state on
        when an earlier kernel ran it, or None when nothing was recorded. See
        ``module_state_key``. Memoized until :meth:`reset`, as
        :meth:`carrier_advances`."""
        memo = self._module_state_memo
        if source_hash in memo:
            return memo[source_hash]
        found = None
        record = self.record(module_state_key(source_hash))
        if record and record.get("module_state"):
            try:
                found = (
                    frozenset(str(name) for name in record.get("names") or ()),
                    frozenset(str(name) for name in record.get("modules") or ()),
                )
            except TypeError:
                found = None
        memo[source_hash] = found
        return found

    def process_state(self, source_hash: str) -> frozenset[str] | None:
        """What of the process a statement changed when an earlier kernel ran
        it, or None when nothing was recorded. See ``process_state_key``.
        Memoized until :meth:`reset`, as :meth:`module_state`."""
        memo = self._process_state_memo
        if source_hash in memo:
            return memo[source_hash]
        found = None
        record = self.record(process_state_key(source_hash))
        if record and record.get("process_state"):
            try:
                found = frozenset(str(kind) for kind in record.get("kinds") or ())
            except TypeError:
                found = None
        memo[source_hash] = found
        return found

    def carrier_advances(self, source_hash: str) -> frozenset[str] | None:
        """The generators a statement drew from when an earlier kernel ran it,
        or None when nothing was recorded. See ``carrier_advances_key``.
        Memoized until :meth:`reset`: every statement the simulation meets
        asks, and most never read a generator."""
        memo = self._carrier_advances_memo
        if source_hash in memo:
            return memo[source_hash]
        found = None
        record = self.record(carrier_advances_key(source_hash))
        if record and record.get("carrier_advances"):
            try:
                found = frozenset(str(name) for name in record.get("names") or ())
            except TypeError:
                found = None
        memo[source_hash] = found
        return found

    def control_outcome(self, stmt_code: str, lineage_now: Callable[[str], str]) -> ControlOutcome | None:
        """A loop's outcome from an earlier kernel, when it may still be trusted.

        See ``control_outcome_key``. Written only for a loop whose outcome was
        all it did; trusted only when every global its callees read has, here
        (*lineage_now*), the lineage it had then -- the entry lineages and file
        state are checked by the caller, exactly as for the session's own
        record. Any doubt returns None, and the loop is replayed.
        """
        record = self.record(control_outcome_key(stmt_code))
        if not record or not record.get("control_outcome") or record.get("code") != stmt_code:
            return None
        try:
            for name, then in (record.get("callees") or {}).items():
                if lineage_now(name) != then:
                    return None
            return (
                dict(record["entry"]),
                dict(record["left"]),
                frozenset(record["files"]),
                str(record["file_component"]),
            )
        except (KeyError, TypeError, ValueError, AttributeError):
            return None

    def import_bindings(self, stmt_code: str) -> dict[str, dict]:
        """What *stmt_code* (a ``from`` import) bound when it last ran -- see
        ``import_bindings_key`` -- or ``{}``. Memoized until :meth:`reset`."""
        memo = self._import_bindings_memo
        if stmt_code in memo:
            return memo[stmt_code]
        found: dict[str, dict] = {}
        record = self.record(import_bindings_key(stmt_code))
        if record and record.get("import_bindings") and record.get("code") == stmt_code:
            found = dict(record.get("bindings") or {})
            found = {k: v for k, v in found.items() if isinstance(v, dict)}
            if record.get("magic") != importlib.util.MAGIC_NUMBER.hex():
                for entry in found.values():
                    entry.pop("code", None)  # another interpreter's bytecode
        memo[stmt_code] = found
        return found

    def loop_split(self, node: ast.AST) -> int | None:
        """Persisted split point for *node*, or ``None`` if it is not split.

        Any failure to resolve the store reads as "not split". A simulator
        that cannot find the store must never guess: a split it invents would
        be one the runtime never recorded.
        """
        if not isinstance(node, ast.For):
            return None
        try:
            if is_split_half(node):
                return None  # never split a half; that recurses
            if self._split_store is None:
                self._split_store = store_for_backend(self.backend())
                if self._split_store is None:
                    return None
            return self._split_store.get(loop_source_hash(node))
        except Exception:  # never let a lookup break simulation
            logger.debug("[UPSTREAM_DEBUG] loop split lookup failed", exc_info=True)
            return None
