"""A costly statement that ran again over a whole collection although most
of it is what it was last time (``CACHE-PER-ITEM-RERUN``).

``clean = F.preprocess(imgs)`` takes 26 s over 3,000 images. Then 400 images
are added, and it takes 29 s over all 3,400 again: ``imgs`` is one value, and
it changed, so nothing of the last run can be reused. Cash cannot split the
work itself, and should not push the user towards catch-up logic of their
own (process only what is new, merge it with what was kept). The fix that
keeps the code simple is a design one: do the work through a cached helper
that takes one item or one chunk (an image, a batch, a season), so the
unchanged ones come back from the cache by themselves.

To say so, this remembers a block fingerprint (:mod:`cash.row_blocks`) of
each input made of rows when a statement that took at least
:data:`MIN_COST_S` is computed, and compares the next computed run's inputs
with it. Only digests are kept, never values, and they are kept with the
statement's other records in the cache, so a run after a restart (or after
the data was regenerated from the shell) compares with the last kernel's.
Hashing is held to :data:`HASH_SHARE` of the time the session's slow
statements took, and to :data:`STEP_SHARE` of any one statement's; past
that the input is not fingerprinted. Once per statement per session. Advice only:
nothing here reaches a key or decides what is served.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from cash._clock import perf_counter as _perf_counter
from cash.diagnostics import log_diagnostic, warn_diagnostic
from cash.exceptions import CashCacheIneffectiveWarning
from cash.notebook.file_round_trips import quote
from cash.row_blocks import HashBudgetSpent, RowBlocks, compare, describe_ranges, fingerprint, row_count

logger = logging.getLogger(__name__)

__all__ = ["CODE", "HASH_SHARE", "MIN_COST_S", "MIN_UNCHANGED", "STEP_SHARE", "PerItemReruns"]

CODE = "CACHE-PER-ITEM-RERUN"

#: What the statement must have taken before its inputs are fingerprinted,
#: and before the hint is worth a line.
MIN_COST_S = 1.0
#: The share of the session's slow-statement time that fingerprints may
#: take altogether.
HASH_SHARE = 0.03
#: The most of one statement's time its fingerprints may take: one big
#: frame on a short step may borrow from what the others left unspent.
STEP_SHARE = 0.10
#: The share of the rows that must be unchanged for the hint: below it, a
#: per-item helper would run again for most items anyway.
MIN_UNCHANGED = 0.5
#: Statements whose fingerprints are kept in memory; past this, the oldest go
#: (a later run reads them back from the cache).
_MAX_SITES = 512
#: Fingerprints remembered by value, for the next statement reading it.
_MAX_TAKEN = 16

#: What a record is: input name -> its fingerprint, and the digest of the
#: lineages of the statement's other inputs (``others``).
Record = dict[str, Any]


#: Plain values an input can be compared by: a recomputed ``best_k = 20``
#: has a new lineage but the same value.
_PLAIN = (int, float, str, bool, type(None))


def _plain(value: Any) -> bool:
    if isinstance(value, _PLAIN):
        return not isinstance(value, str) or len(value) <= 200
    if _numpy_scalar(value):
        return True
    return (
        isinstance(value, tuple) and len(value) <= 16 and all(isinstance(v, _PLAIN) or _numpy_scalar(v) for v in value)
    )


def _numpy_scalar(value: Any) -> bool:
    """A numpy number (``grid_s.index[0]`` unpacks to ``np.int64``)."""
    return type(value).__module__ == "numpy" and getattr(value, "shape", None) == () and value.dtype.kind in "biuf"


def _others_digest(lineages: Mapping[str, str], values: Mapping[str, Any], fingerprinted: Iterable[str]) -> str:
    """A digest of the inputs that are not fingerprinted, by value when it
    is a plain number, string or tuple of them, else by lineage: two runs
    compare only when these are the same."""
    skip = set(fingerprinted)
    parts = []
    for name in sorted(set(lineages) | set(values)):
        if name in skip:
            continue
        value = values.get(name)
        if name in values and _plain(value):
            parts.append(f"{name}:{type(value).__name__}:{value!r}")
        else:
            parts.append(f"{name}={lineages.get(name)}")
    return hashlib.sha256("\0".join(parts).encode()).hexdigest()[:16]


def _seconds(s: float) -> str:
    return f"{s:.0f} s" if s >= 10 else f"{s:.1f} s"


@dataclass
class PerItemReruns:
    """The fingerprints of costly statements' inputs, and the hints given."""

    #: source hash -> the record of its last computed run.
    _records: dict[str, Record] = field(default_factory=dict)
    #: Statements already warned about this session.
    _warned: set[str] = field(default_factory=set)
    #: The last fingerprints taken, by (value id, lineage, grid): no values.
    _taken: dict[tuple, Any] = field(default_factory=dict)
    #: Seconds of hashing the session may still spend (:data:`HASH_SHARE`).
    _budget: list[float] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self._records) + len(self._warned) + len(self._taken) + len(self._budget)

    def clear(self) -> None:
        """Forget the session, as a restarted kernel does."""
        self._records.clear()
        self._warned.clear()
        self._taken.clear()
        self._budget.clear()

    def credit(self, seconds: float) -> float:
        """Add a slow statement's *seconds* to the session's hashing budget,
        also one this hint does not check (a statement of a loop body);
        what the budget holds now."""
        left = self._budget[0] if self._budget else 0.0
        if seconds >= MIN_COST_S:
            left += HASH_SHARE * seconds
            self._budget[:] = [left]
        return left

    def after_run(
        self,
        code: str,
        source_hash: str,
        *,
        values: Mapping[str, Any],
        lineages: Mapping[str, str],
        seconds: float,
        load: Callable[[str], Record | None] = lambda _h: None,
        save: Callable[[str, Record], None] = lambda _h, _r: None,
    ) -> None:
        """Note a statement that was just computed: *code* (as the user wrote
        it; keyed by *source_hash*) took *seconds* reading the inputs
        *values* (name -> value, the ones it did not rebind or change) whose
        lineages as it started were *lineages*. *load* and *save* read and
        write a record that outlives the kernel."""
        if seconds < MIN_COST_S:
            return
        left = self.credit(seconds)
        rowed = {name: value for name, value in values.items() if row_count(value) is not None}
        if not rowed:
            return
        others = _others_digest(lineages, values, rowed)
        before = self._records.get(source_hash)
        if before is None:
            before = load(source_hash)
        started = _perf_counter()
        deadline = started + min(left, STEP_SHARE * seconds)
        try:
            self._fingerprint_and_compare(code, source_hash, rowed, lineages, seconds, before, others, deadline, save)
        finally:
            self._budget[:] = [max(0.0, left - (_perf_counter() - started))]

    def _fingerprint_and_compare(
        self,
        code: str,
        source_hash: str,
        rowed: Mapping[str, Any],
        lineages: Mapping[str, str],
        seconds: float,
        before: Record | None,
        others: str,
        deadline: float,
        save: Callable[[str, Record], None],
    ) -> None:
        same_rest = before is not None and before.get("others") == others
        old_inputs = (before or {}).get("inputs") or {}
        record: Record = {"others": others, "inputs": {}}
        changes = {}
        for name, value in sorted(rowed.items(), key=lambda item: item[0]):
            old = RowBlocks.from_dict(old_inputs[name]) if same_rest and name in old_inputs else None
            # The same value read by the next statement too (`clean` by
            # three feature steps) is hashed once.
            memo = (id(value), lineages.get(name), old.block if old else None, old.rows if old else None)
            taken = self._taken.get(memo) if memo[1] is not None else None
            if taken is None:
                try:
                    taken = fingerprint(value, like=old, deadline=deadline)
                except HashBudgetSpent:
                    logger.debug("%s: fingerprinting %r would cost too much", CODE, name)
                    return  # keep the old record: nothing was compared
                if memo[1] is not None and taken is not None:
                    self._taken[memo] = taken
                    if len(self._taken) > _MAX_TAKEN:
                        del self._taken[next(iter(self._taken))]
            if taken is None:
                continue
            blocks, cut = taken
            record["inputs"][name] = blocks.to_dict()
            if old is not None:
                change = compare(old, blocks, cut)
                if change is not None:
                    changes[name] = change
        if not record["inputs"]:
            return
        self._remember(source_hash, record)
        try:
            save(source_hash, record)
        except Exception:  # a hint never breaks a statement
            logger.debug("%s: could not keep the record", CODE, exc_info=True)
        if source_hash in self._warned or not changes:
            return
        if any(c.changed_rows and c.unchanged_share < MIN_UNCHANGED for c in changes.values()):
            return  # most of an input is new: per-item caching would rerun most of it too
        moved = [(name, c) for name, c in changes.items() if c.changed_rows]
        if not moved:
            return  # the inputs are the same: it ran for another reason
        name, change = max(moved, key=lambda item: item[1].rows)
        self._warned.add(source_hash)
        self._warn(code, name, change, seconds)

    def _remember(self, source_hash: str, record: Record) -> None:
        self._records.pop(source_hash, None)
        self._records[source_hash] = record
        if len(self._records) > _MAX_SITES:
            del self._records[next(iter(self._records))]

    @staticmethod
    def _warn(code: str, name: str, change: Any, seconds: float) -> None:
        changed = change.changed_rows
        what = (
            f"{quote(code)} re-ran over all {change.rows:,} rows of `{name}` ({_seconds(seconds)}) "
            f"but only {changed:,} {'row differs' if changed == 1 else 'rows differ'} from the last run "
            f"({describe_ranges(change.changed)}). Calling a cached helper per item or per chunk "
            f"lets cash reuse the rest."
        )
        fix = (
            "do the work in a @cash.cache function that takes one chunk of rows (a batch, a season, "
            "or one item when each takes a while) and combine the results, e.g. "
            "`np.concatenate([prep(chunk) for chunk in chunks])`. Cut chunks on something that stays put "
            "(a batch column, a fixed row count), so new rows make new chunks."
        )
        log_diagnostic(logger, CODE, what, fix)
        warn_diagnostic(CashCacheIneffectiveWarning, CODE, what, fix)
