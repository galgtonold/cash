"""What a helper's globals fold put into a key, remembered while nothing it
read can have moved.

A cached function that reaches hundreds of helpers (a library's masker
calling its validation, logging and image code) folds the data globals each
of them reads on every key build: ~30 us a helper, tens of milliseconds a
call, for values that did not change. With the C extension, the fast path's
watchers prove that: `hit_memo.watch_helper` watches everything a helper's
code names (its globals by name, the modules and classes they lead to, its
defaults and closure), refusing a helper whose reads hold anything no
watcher sees (a list, an instance, an array). While the world epoch, the
cash epoch and the helper's closure cells stand still, folding it again
would read the same values and give the same parts. A fold that keyed an
environment value (`effects.ENVIRONMENT_READS` moved) is never kept: no
watcher sees the environment.

An entry is the fold's whole effect: the state hash it turned *state_in*
into, and what it left in the key build -- the pairs it marked as folded
(`LoggedSeen`), its capture watch, ledger, reads-folded and folded-code
records -- replayed on reuse. It is looked up by the helper, the cached
function's code and name, the decorator names folded with it, the state
hash it started from and what the build had folded before it, so a reuse
is the same step of the same build. ``CASH_HITPATH_VERIFY=1`` folds again
behind every reuse and raises `HitMemoMismatch` when anything differs.
Without the extension nothing is remembered.
"""

from __future__ import annotations

import contextvars
from collections.abc import Callable, Iterable
from typing import Any

from .._memo import HELPER_FOLDS, LruMemo
from ..code_swaps import FOLDED_CODE, FoldedCode
from ..dependency_state import STATE_LEDGER
from ..effects import ENVIRONMENT_READS
from . import hit_memo
from .call_state import CAPTURE_WATCH

_MASK = (1 << 64) - 1

#: Counters for tests and profiling.
STATS = {"served": 0, "recorded": 0, "refused": 0}


def _mix(item: Any) -> int:
    """A 64-bit mix of *item*'s hash (splitmix64), so a sum of them does not
    cancel out the way a plain sum or xor of small hashes can."""
    z = (hash(item) + 0x9E3779B97F4A7C15) & _MASK
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK
    return z ^ (z >> 31)


class LoggedSeen(set):
    """The *seen* set of a key build's helper folds, which also keeps the
    order things were added in and a fingerprint of all of it.

    The fingerprint (count and a sum of mixed hashes) says what the build
    had folded before a helper, independent of the order; the log is what
    a helper's fold added, replayed when the fold is reused."""

    __slots__ = ("count", "log", "total")

    def __init__(self, items: Iterable = ()) -> None:
        super().__init__()
        self.log: list = []
        self.total = 0
        self.count = 0
        self.update(items)

    def add(self, item: Any) -> None:
        if item not in self:
            super().add(item)
            self.log.append(item)
            self.total = (self.total + _mix(item)) & _MASK
            self.count += 1

    def update(self, *others: Iterable) -> None:
        for other in others:
            for item in other:
                self.add(item)

    def __ior__(self, other: Any) -> LoggedSeen:
        self.update(other)
        return self

    def fingerprint(self) -> tuple[int, int]:
        return self.count, self.total


class _ReadThrough(dict):
    """A key build's reads-folded record as one helper fold sees it: reads
    fall through to the build's, writes stay here, to be kept."""

    def __init__(self, below: dict) -> None:
        super().__init__()
        self.below = below

    def get(self, key: Any, default: Any = None) -> Any:
        if dict.__contains__(self, key):
            return dict.__getitem__(self, key)
        return self.below.get(key, default)

    def __contains__(self, key: object) -> bool:
        return dict.__contains__(self, key) or key in self.below

    def __getitem__(self, key: Any) -> Any:
        if dict.__contains__(self, key):
            return dict.__getitem__(self, key)
        return self.below[key]


class _Entry:
    __slots__ = (
        "added",
        "cells",
        "cash",
        "folded",
        "helper",
        "ledger",
        "owners",
        "reports",
        "owner",
        "state_out",
        "watch",
        "world",
    )


def _effects() -> tuple:
    """The key build's records a helper fold writes to."""
    from .globals_fold import READS_FOLDED

    return CAPTURE_WATCH.get(), STATE_LEDGER.get(), READS_FOLDED.get(), FOLDED_CODE.get()


class HelperFoldMemo:
    """One `GlobalsFold`'s remembered helper folds (see the module doc)."""

    def __init__(self) -> None:
        self._entries: LruMemo[tuple, _Entry] = LruMemo(HELPER_FOLDS)
        #: (id(helper), world epoch) of helpers refused at that epoch.
        self._refused: LruMemo[tuple, Any] = LruMemo(HELPER_FOLDS)

    def fold(
        self,
        helper: Callable,
        key: tuple,
        owner: Any,
        root_module: str | None,
        state_in: str,
        seen: LoggedSeen,
        compute: Callable[[], str],
    ) -> str:
        """``compute()`` -- the helper's globals fold -- or what it gave the
        last time this same step ran while nothing it reads moved."""
        epochs = hit_memo.world_epochs()
        if epochs is None:
            return compute()
        world, cash = epochs
        full_key = (*key, state_in, seen.fingerprint())
        entry = self._entries.get(full_key)
        if (
            entry is not None
            and entry.helper is helper
            and entry.owner is owner
            and entry.world == world
            and entry.cash == cash
            and hit_memo.cells_hold(entry.cells)
        ):
            if hit_memo.VERIFY:
                self._verify(entry, seen, compute)
            else:
                self._replay(entry, seen)
            STATS["served"] += 1
            return entry.state_out
        refused = self._refused.get((id(helper), world))
        if refused is helper:
            return compute()
        try:
            cells = hit_memo.watch_helper(helper, root_module)
        except Exception:  # noqa: BLE001 - Ineligible, or a scan that broke: folded every time
            self._refused[(id(helper), world)] = helper
            STATS["refused"] += 1
            return compute()
        # Watched from here on: what moves during the fold below is seen.
        world_before = hit_memo.world_epochs()
        env_before = ENVIRONMENT_READS[0]
        entry, state_out = self._record(helper, seen, compute)
        if ENVIRONMENT_READS[0] != env_before:
            # It keyed an environment value (os.environ, the cwd), which no
            # watcher sees: folded every time while the world stands still.
            self._refused[(id(helper), world)] = helper
            STATS["refused"] += 1
        elif hit_memo.world_epochs() == world_before and world_before is not None:
            entry.world, entry.cash = world_before
            entry.owner = owner
            entry.cells = cells
            self._entries[full_key] = entry
            STATS["recorded"] += 1
        return state_out

    def _record(self, helper: Callable, seen: LoggedSeen, compute: Callable[[], str]) -> tuple[_Entry, str]:
        """Run the fold with the build's records swapped for empty ones of
        its own, then hand what it wrote on to the build's."""
        from .globals_fold import READS_FOLDED

        watch, ledger, folded, code = _effects()
        start = len(seen.log)
        tokens: list[tuple[contextvars.ContextVar, contextvars.Token]] = []
        own_watch = {} if watch is not None else None
        own_ledger = {} if ledger is not None else None
        own_folded = _ReadThrough(folded) if folded is not None else None
        own_code = FoldedCode() if code is not None else None
        for var, value in (
            (CAPTURE_WATCH, own_watch),
            (STATE_LEDGER, own_ledger),
            (READS_FOLDED, own_folded),
            (FOLDED_CODE, own_code),
        ):
            tokens.append((var, var.set(value)))
        try:
            state_out = compute()
        finally:
            for var, token in reversed(tokens):
                var.reset(token)
        entry = _Entry()
        entry.helper = helper
        entry.state_out = state_out
        entry.added = tuple(seen.log[start:])
        entry.watch = own_watch or {}
        entry.ledger = own_ledger or {}
        entry.folded = dict(own_folded) if own_folded else {}
        entry.owners = dict(own_code.owners) if own_code is not None else {}
        entry.reports = dict(own_code.reports) if own_code is not None else {}
        self._replay(entry, None)
        return entry, state_out

    def _replay(self, entry: _Entry, seen: LoggedSeen | None) -> None:
        """Leave in the key build what the remembered fold left there."""
        watch, ledger, folded, code = _effects()
        if seen is not None:
            seen.update(entry.added)
        if watch is not None and entry.watch:
            watch.update(entry.watch)
        if ledger is not None and entry.ledger:
            ledger.update(entry.ledger)
        if folded is not None and entry.folded:
            folded.update(entry.folded)
        if code is not None:
            code.owners.update(entry.owners)
            code.reports.update(entry.reports)

    def _verify(self, entry: _Entry, seen: LoggedSeen, compute: Callable[[], str]) -> None:
        """Fold again and compare with the entry, as the fast path's verify
        mode compares keys."""
        fresh, state_out = self._record(entry.helper, seen, compute)
        same = (
            state_out == entry.state_out
            and fresh.added == entry.added
            and fresh.watch.keys() == entry.watch.keys()
            and fresh.ledger == entry.ledger
            and fresh.folded.keys() == entry.folded.keys()
            and fresh.owners.keys() == entry.owners.keys()
        )
        if not same:
            raise hit_memo.HitMemoMismatch(
                f"a remembered globals fold of {getattr(entry.helper, '__qualname__', '?')} "
                "differs from folding it again"
            )
