"""What the upstream check learns about files during one cell run, kept for it.

The check before a cell validates every upstream cache entry's file
dependencies, and entries share their files: in one notebook every entry
depended on the same 5,222 documents, in two spellings, and each entry checked
all of them again, twice (freshness, then mtime) -- 7-11 s before every cell of
a notebook that runs in 30 s uncached. So the answers are kept:

* per cache entry, the key of each entry found fresh;
* per file, each freshness answer and each path's resolution and stat.

The rule for both: they last for one cell run (``file_dep_snapshot.HASH_EPOCH``)
and nothing is kept outside a run. Within a run, a statement that writes files
calls :func:`forget_file_state_this_run`, which drops the per-file answers --
a file may have changed -- and keeps the per-entry ones: an entry found fresh
was produced before the write, as it is in a from-the-top run, so checking it
again can only repeat the answer.

The runtime (``statement.processor``) calls the forget; the upstream simulation
reads and fills both. Neither side imports the other for it.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Any

from .._paths import resolve_file_dep_path
from ..tracking import file_dep_snapshot as _fds
from ..tracking.file_dep_snapshot import LISTING_MIN_FILES, FreshnessMemo, stats_from_listings

__all__ = [
    "file_state_this_run",
    "forget_file_state_this_run",
    "known_fresh_entry",
    "note_fresh_entry",
    "stats_this_run",
]

#: This run's cache keys whose file dependencies were found fresh.
_FRESH_ENTRY_VERDICTS: dict = {}

#: This run's per-file answers: the freshness answer for each (path, recorded
#: snapshot) pair, and each path's resolution and stat.
_FILE_STATE_THIS_RUN: dict = {}


def known_fresh_entry(key: str | None) -> bool:
    """True if the entry keyed *key* was found fresh earlier in this run."""
    epoch = _fds.HASH_EPOCH
    if key is None or epoch is None:
        return False
    memo = _FRESH_ENTRY_VERDICTS
    if memo.get("epoch") != epoch:
        memo.clear()
        memo["epoch"] = epoch
        memo["keys"] = set()
    return key in memo["keys"]


def note_fresh_entry(key: str | None) -> None:
    """Remember for the rest of this run that the entry keyed *key* is fresh."""
    epoch = _fds.HASH_EPOCH
    if key is None or epoch is None:
        return
    memo = _FRESH_ENTRY_VERDICTS
    if memo.get("epoch") != epoch:
        memo.clear()
        memo["epoch"] = epoch
        memo["keys"] = set()
    memo["keys"].add(key)


def file_state_this_run() -> dict | None:
    """This cell run's per-file memo, or None outside a run."""
    epoch = _fds.HASH_EPOCH
    if epoch is None:
        return None
    if _FILE_STATE_THIS_RUN.get("epoch") != epoch:
        _FILE_STATE_THIS_RUN.clear()
        _FILE_STATE_THIS_RUN.update(epoch=epoch, memo=FreshnessMemo(), where={})
    return _FILE_STATE_THIS_RUN


def forget_file_state_this_run() -> None:
    """A statement of this cell run wrote files: per-file answers taken before
    it are not answers for entries checked after it."""
    _FILE_STATE_THIS_RUN.clear()


def _locate_files(paths: Iterable[str], run: dict | None) -> dict[str, tuple[str | None, Any]]:
    """``{path: (resolved or None, stat or None)}`` for *paths*, this run's answers first.

    A crowded directory is read with one listing (``stats_from_listings``); a
    listed path is where it was recorded. The rest go through
    ``resolve_file_dep_path``'s relocation fallbacks.
    """
    where = run["where"] if run is not None else {}
    paths = list(paths)
    todo = [p for p in paths if p not in where]
    listed = stats_from_listings(todo) if len(todo) >= LISTING_MIN_FILES else {}
    found: dict[str, tuple[str | None, Any]] = {}
    for p in todo:
        st = listed.get(p)
        found[p] = (p, st) if st is not None else (resolve_file_dep_path(p), None)
    if run is not None and len(where) < 200_000:
        where.update(found)
    return {p: found[p] if p in found else where[p] for p in paths}


def stats_this_run(paths: Iterable[str]) -> dict[str, tuple[str | None, Any]]:
    """``{path: (resolved or None, stat or None)}``, one stat per path per cell run.

    The resolution is ``resolve_file_dep_path``'s (relocation fallbacks
    included); the stat is the listing's where the directory was listed.
    """
    run = file_state_this_run()
    stats = run.setdefault("stat", {}) if run is not None else {}
    paths = list(paths)
    todo = [p for p in paths if p not in stats]
    if todo:
        for p, (resolved, listed) in _locate_files(todo, run).items():
            st = listed
            if st is None and resolved is not None:
                try:
                    st = os.stat(resolved)
                except OSError:
                    st = None
            stats[p] = (resolved, st)
    return {p: stats[p] for p in paths}
