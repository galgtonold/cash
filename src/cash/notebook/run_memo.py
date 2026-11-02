"""What the upstream check learns about files during one cell run, kept for it.

The check before a cell validates every upstream cache entry's file
dependencies, and entries share their files: in one notebook every entry
depended on the same 5,222 documents, in two spellings, and each entry checked
all of them again, twice (freshness, then mtime) -- 7-11 s before every cell of
a notebook that runs in 30 s uncached. So the answers are kept:

* per cache entry, the key of each entry found fresh;
* per file, each freshness answer and each path's resolution and stat;
* per snapshot, each one found fresh whole, and how old its answers are.

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

import operator
import os
import time
from collections.abc import Iterable, Mapping
from typing import Any

from .._paths import is_remote_url, normalize_path, resolve_file_dep_path
from ..tracking import file_dep_snapshot as _fds
from ..tracking.file_dep_snapshot import LISTING_MIN_FILES, FreshnessMemo, stats_from_listings
from ..tracking.read_classification import own_stat

__all__ = [
    "file_component_this_run",
    "fresh_snapshot_this_run",
    "file_state_this_run",
    "forget_file_state_this_run",
    "known_fresh_entry",
    "note_fresh_entry",
    "note_fresh_snapshot",
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
        # The last run's stats, to hand on where a file is as it was then
        # (`stats_this_run`); dropped with the rest after a write.
        last = _FILE_STATE_THIS_RUN.get("stat") or {}
        _FILE_STATE_THIS_RUN.clear()
        _FILE_STATE_THIS_RUN.update(epoch=epoch, memo=FreshnessMemo(), stat={}, since=time.monotonic(), last=last)
    return _FILE_STATE_THIS_RUN


def forget_file_state_this_run() -> None:
    """A statement of this cell run wrote files: per-file answers taken before
    it are not answers for entries checked after it."""
    _FILE_STATE_THIS_RUN.clear()


#: How many snapshots found fresh one run keeps (`note_fresh_snapshot`).
_FRESH_SNAPSHOTS_KEPT = 16


def note_fresh_snapshot(snap: Mapping[str, Any]) -> None:
    """Remember that *snap* (a ``file_dependencies`` snapshot) was found fresh
    whole with this run's per-file answers (``file_state_this_run()["memo"]``).

    Dated by when those answers began to be taken, not by when this check
    ran: an answer it reused is that old. Kept with them, so a statement that
    writes files drops it too.
    """
    run = file_state_this_run()
    if run is None:
        return
    kept = run.setdefault("fresh", [])
    if len(kept) < _FRESH_SNAPSHOTS_KEPT:
        kept.append((snap, run["since"]))


def fresh_snapshot_this_run(snap: Mapping[str, Any], not_before: float) -> bool:
    """Was a snapshot equal to *snap* found fresh whole in this run, from
    answers taken at *not_before* (``time.monotonic()``) or later?

    The upstream check looks up the entry of the cell above, and the statement
    below then checks its own entry, which inherited the same files: a frame
    built from 24,000 documents was checked twice in one cell. The caller
    says how old an answer it takes. Compared as whole dicts, in C.
    """
    epoch = _fds.HASH_EPOCH
    if epoch is None or _FILE_STATE_THIS_RUN.get("epoch") != epoch:
        return False
    return any(
        at >= not_before and len(known) == len(snap) and known == snap
        for known, at in _FILE_STATE_THIS_RUN.get("fresh", ())
    )


#: Every named field of a stat on this platform, beyond the ten its tuple
#: compares (mode, inode, device, links, owner, group, size and the
#: whole-second times): the exact times and what the platform adds (blocks,
#: flags, attributes).
_STAT_FIELDS = operator.attrgetter(
    *sorted(
        name
        for name in dir(os.stat_result)
        if name.startswith("st_")
        and name not in ("st_mode", "st_ino", "st_dev", "st_nlink", "st_uid", "st_gid", "st_size")
    )
)


def same_stat(a: os.stat_result, b: os.stat_result) -> bool:
    """Whether stats *a* and *b* agree in every field."""
    return a == b and _STAT_FIELDS(a) == _STAT_FIELDS(b)


def stats_this_run(paths: Iterable[str]) -> dict[str, tuple[str | None, Any]]:
    """``{path: (resolved or None, stat or None)}``, one stat per path per cell run.

    A path is looked for where it was recorded first; one that is not there
    goes through ``resolve_file_dep_path``'s relocation fallbacks. A crowded
    directory is read with one listing (``stats_from_listings``); a listed
    path is where it was recorded. Stat-ed past cash's own wrapper of
    ``os.stat`` (``own_stat``): the stats are cash's, asked of every file the
    cells above read before each cell.

    One pass over the paths, one tuple each: before every cell below a loop
    over 24,000 documents, the bookkeeping of a stat cost as much as the stat.

    A file whose stat equals, field for field, the one the last cell run took
    keeps that run's record: the same values, and no new objects. The records
    live for the whole run, so new ones reached the collector's oldest
    generation, 200,000 of them before every cell below a loop over 50,000
    documents, and set off a full collection of the heap nearly every cell.
    """
    run = file_state_this_run()
    stats = run.setdefault("stat", {}) if run is not None else {}
    last = run.get("last") if run is not None else None
    if not isinstance(paths, (list, tuple, set, frozenset, dict)):
        paths = list(paths)
    todo = [p for p in paths if p not in stats]
    if todo:
        listed = stats_from_listings(todo) if len(todo) >= LISTING_MIN_FILES else None
        stat = os.stat
        if getattr(stat, "_is_file_tracker_patch", False):
            stat = stat._original_func  # as `own_stat`, once for the lot
        for p in todo:
            st = listed.get(p) if listed else None
            if st is None and not ("://" in p and is_remote_url(p)):
                # Where it was recorded: the stat that finds it is there is
                # the stat it is asked for (`resolve_file_dep_path` asked
                # ``os.path.exists`` first, a stat of its own).
                try:
                    st = stat(p)
                except (OSError, ValueError):
                    st = None
            if st is not None:
                if last:
                    before = last.get(p)
                    # `same_stat`, inline: it is asked once per file read above.
                    if (
                        before is not None
                        and before[0] == p
                        and before[1] == st
                        and _STAT_FIELDS(before[1]) == _STAT_FIELDS(st)
                    ):
                        stats[p] = before
                        continue
                stats[p] = (p, st)
                continue
            resolved = resolve_file_dep_path(p)
            if resolved is not None:
                try:
                    st = own_stat(resolved)
                except OSError:
                    st = None
            stats[p] = (resolved, st)
    return {p: stats[p] for p in paths}


def file_component_this_run(
    paths: set[str], remote: set[str] | None = None, probes: Mapping[str, str] | None = None
) -> str:
    """`compute_file_hash_component` of *paths*, with this run's stats
    (`stats_this_run`) in place of a stat of each file of its own: the same
    fields of the same file in the same run, so the same component; and the
    component itself kept for the rest of the run, as the stats are. A
    directory listing's stat is not handed on (it has no file identity, and
    can lag a file edited through another hard link)."""
    from .statement.file_deps import compute_file_hash_component

    run = file_state_this_run()
    components = run.setdefault("component", {}) if run is not None else None
    key = None
    if components is not None and not remote:  # an object store's token is asked each time
        # One statement's files are asked for again by each lookup of the
        # run: a cell below a read of 5,000 files built the same component
        # three times.
        try:
            key = (frozenset(paths), frozenset(remote or ()), tuple(sorted((probes or {}).items())))
        except TypeError:
            key = None
        if key is not None and key in components:
            return components[key]
    stats = {
        normalize_path(p): st
        for p, (resolved, st) in stats_this_run(paths).items()
        if st is not None and resolved == p and st.st_ino
    }
    component = compute_file_hash_component(paths, remote, probes, stats=stats)
    if key is not None:
        components[key] = component
    return component
