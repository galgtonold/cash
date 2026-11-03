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

import bisect
import itertools
import os
import stat as _stat
import time
from collections.abc import Iterable, Mapping
from typing import Any

from .._paths import is_remote_url, normalize_path, resolve_file_dep_path
from ..tracking import file_dep_snapshot as _fds
from ..tracking.file_dep_snapshot import LISTING_MIN_FILES, FreshnessMemo, stats_from_listings
from ..tracking.read_classification import own_stat, stat_key

__all__ = [
    "file_component_this_run",
    "fresh_snapshot_this_run",
    "file_state_this_run",
    "forget_file_state_this_run",
    "known_fresh_entry",
    "note_fresh_entry",
    "note_fresh_snapshot",
    "stat_keys_this_run",
    "stats_taken_this_run",
    "stat_dates_this_run",
    "stats_this_run",
    "RunStats",
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
        _FILE_STATE_THIS_RUN.update(
            epoch=epoch, memo=FreshnessMemo(taken=stats_taken_this_run), stat={}, since=time.monotonic()
        )
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


class RunStats(Mapping):
    """``{path: (resolved or None, stat or None)}`` over this run's stats,
    each pair made when it is asked for (`stats_this_run`).

    The run keeps a path found where it was recorded as its bare stat, and
    any other as its pair, in a second dict: a ``(path, stat)`` tuple is
    tracked by the garbage collector (a stat result is a collector type), so
    one per file of a 24,000-file check lived through the cell, was promoted
    to the oldest generation, and set off a full collection of the kernel's
    heap every few cells. A bare stat is not tracked.
    """

    __slots__ = ("_at", "_elsewhere", "_paths")

    def __init__(self, paths: dict[str, None], at: dict, elsewhere: dict) -> None:
        self._paths = paths
        self._at = at
        self._elsewhere = elsewhere

    def __getitem__(self, path: str) -> tuple[str | None, Any]:
        if path not in self._paths:
            raise KeyError(path)
        st = self._at.get(path)
        return (path, st) if st is not None else self._elsewhere[path]

    def __contains__(self, path: object) -> bool:
        return path in self._paths

    def __iter__(self):
        return iter(self._paths)

    def __len__(self) -> int:
        return len(self._paths)

    def items(self):  # type: ignore[override]
        # A generator, not a view: one pair alive at a time.
        at = self._at
        elsewhere = self._elsewhere
        for path in self._paths:
            st = at.get(path)
            yield path, ((path, st) if st is not None else elsewhere[path])


def _as_set(paths: dict | set | frozenset):
    """*paths* as a set-like that a dict's keys compare with in C."""
    return paths.keys() if isinstance(paths, dict) else paths


def stats_this_run(paths: Iterable[str]) -> RunStats:
    """``{path: (resolved or None, stat or None)}``, one stat per path per cell run.

    A path is looked for where it was recorded first; one that is not there
    goes through ``resolve_file_dep_path``'s relocation fallbacks. A crowded
    directory is read with one listing (``stats_from_listings``); a listed
    path is where it was recorded. Stat-ed past cash's own wrapper of
    ``os.stat`` (``own_stat``): the stats are cash's, asked of every file the
    cells above read before each cell.

    One pass over the paths, one record each: before every cell below a loop
    over 24,000 documents, the bookkeeping of a stat cost as much as the stat.
    The record is the stat itself for a path found where it was recorded
    (`RunStats`), so the garbage collector has nothing per file to track.

    Dated a chunk at a time (`_STAT_CHUNK` stats): each stat kept is dated
    by when its chunk began (``run["stat_date"]``, one shared float per
    chunk), and ``run["stat_chunks"]`` lists ``(date, first index)`` in the
    order the stats were kept, so `stats_taken_this_run` hands a caller
    that takes only recent stats the recent ones of a long pass, and the
    caller takes the older ones again itself.

    Taken with ``lstat``, and with ``stat`` only for a link or reparse point
    (``run["stat_followed"]`` names those): for anything else the two are
    the same, and the ``lstat`` also tells `file_component_this_run` the
    file is no link, which it took one more ``lstat`` per file to learn.
    """
    run = file_state_this_run()
    if run is not None:
        stats = run.setdefault("stat", {})
        elsewhere = run.setdefault("stat_elsewhere", {})
        dates = run.setdefault("stat_date", {})
        chunks = run.setdefault("stat_chunks", [])
        followed = run.setdefault("stat_followed", set())
        from_listing = run.setdefault("listed", set())
    else:
        stats, elsewhere, dates, chunks, followed, from_listing = {}, {}, {}, [], set(), set()
    if not isinstance(paths, (dict, set, frozenset)):
        paths = dict.fromkeys(paths)  # the paths as asked, in order, once each
    # Asked again by each check of the run: most often every path was
    # stat-ed already, which the run's own dict tells in C.
    todo = [] if stats.keys() >= _as_set(paths) else [p for p in paths if p not in stats and p not in elsewhere]
    if todo:
        at = time.monotonic()
        listed = stats_from_listings(todo) if len(todo) >= LISTING_MIN_FILES else None
        if listed:
            # A listing's stats, dated by when the listing began, kept first:
            # the dates stay in the order the stats are kept.
            chunks.append((at, len(stats)))
            for p in todo:
                st = listed.get(p)
                if st is not None and p not in stats:
                    stats[p] = st
                    dates[p] = at
                    from_listing.add(p)
        stat = os.stat
        if getattr(stat, "_is_file_tracker_patch", False):
            stat = stat._original_func  # as `own_stat`, once for the lot
        lstat = os.lstat
        if getattr(lstat, "_is_file_tracker_patch", False):
            lstat = lstat._original_func
        cwd: list[str] = []  # the working directory, asked once for the lot
        at = time.monotonic()
        chunks.append((at, len(stats)))
        left = _STAT_CHUNK
        for p in todo:
            if p in stats:  # listed, or its other spelling's stat, taken just now
                continue
            left -= 1
            if not left:
                at = time.monotonic()
                chunks.append((at, len(stats)))
                left = _STAT_CHUNK
            st = None
            if not os.path.isabs(p) and not ("://" in p and is_remote_url(p)):
                st = _as_spelled_absolute(p, stats, stat, lstat, cwd, dates, at, followed)
            if st is None and not ("://" in p and is_remote_url(p)):
                # Where it was recorded: the stat that finds it is there is
                # the stat it is asked for (`resolve_file_dep_path` asked
                # ``os.path.exists`` first, a stat of its own).
                try:
                    st = _stat_by_lstat(p, stat, lstat, followed)
                except (OSError, ValueError):
                    st = None
            if st is not None:
                stats[p] = st
                dates[p] = at
                continue
            resolved = resolve_file_dep_path(p)
            if resolved is not None:
                try:
                    st = own_stat(resolved)
                except OSError:
                    st = None
            if resolved == p and st is not None:
                stats[p] = st
                dates[p] = at
                followed.add(p)
            else:
                elsewhere[p] = (resolved, st)
    return RunStats(paths, stats, elsewhere)


#: ``FILE_ATTRIBUTE_REPARSE_POINT``: a symlink or junction on Windows.
_REPARSE_POINT = 0x400


def _stat_by_lstat(path: str, stat: Any, lstat: Any, followed: set[str]) -> os.stat_result:
    """``stat(path)``, taken as ``lstat(path)`` unless *path* is a link or
    reparse point, which is stat-ed as well and noted in *followed*."""
    st = lstat(path)
    if _stat.S_ISLNK(st.st_mode) or getattr(st, "st_file_attributes", 0) & _REPARSE_POINT:
        followed.add(path)
        st = stat(path)
    return st


def _as_spelled_absolute(
    path: str,
    stats: dict[str, Any],
    stat: Any,
    lstat: Any,
    cwd: list[str],
    dates: dict[str, float],
    at: float,
    followed: set[str],
) -> os.stat_result | None:
    """The stat of relative *path* by its absolute spelling in the working
    directory, which this run took or takes now under that spelling too;
    None when there is none, or the spellings may not name the same file.

    A file read by a relative path is recorded under both spellings
    (``FileAccessTracker.track_path``), and every file of a folder of 20,000
    images was stat-ed twice before each cell. Not for a path through
    ``..``: spelled out, it goes up from the folder a link is in, where the
    open goes up from the folder the link points to. Nor for a path naming a
    drive (``C:data.csv``), which is relative to that drive's own folder.
    *cwd* holds the working directory once asked: ``os.path.abspath`` asks
    for it again for each of 20,000 paths.
    """
    if ".." in path.replace("\\", "/").split("/") or os.path.splitdrive(path)[0]:
        return None
    try:
        if not cwd:
            cwd.append(os.getcwd())
        absolute = normalize_path(os.path.normpath(os.path.join(cwd[0], path)))
    except (OSError, ValueError):
        return None
    st = stats.get(absolute)
    if st is None:
        try:
            st = _stat_by_lstat(absolute, stat, lstat, followed)
        except (OSError, ValueError):
            return None
        stats[absolute] = st
        dates[absolute] = at
    return st


#: How many stats `stats_this_run` dates with one time.
_STAT_CHUNK = 256


def stats_taken_this_run(paths: Iterable[str], not_before: float | None = None) -> Mapping[str, os.stat_result]:
    """``{path: stat}`` of this run's stats of *paths* (`stats_this_run`)
    for the paths found where they were recorded, taking the missing ones
    now; outside a run, nothing. May hold more paths than asked.

    With *not_before* (``time.monotonic()``), only the stats taken since,
    by their chunk's date (`stats_this_run`): a caller whose answers last
    no longer than that (``CacheFreshnessChecker``) takes the older ones
    again itself. Dated by the run's oldest stat instead, a lookup after
    a pass of 20,000 stats that took longer than that stat-ed every file
    again.

    What `FreshnessMemo.taken` hands every freshness check of the run: the
    check of an entry's files, the simulation's and the statement lookup's
    each stat-ed the same 20,000 files again before a trivial cell."""
    run = file_state_this_run()
    if run is None:
        return {}
    stats_this_run(paths)
    stats = run["stat"]
    chunks = run.get("stat_chunks")
    if not_before is not None and chunks and chunks[0][0] < not_before:
        first = bisect.bisect_left(chunks, not_before, key=lambda chunk: chunk[0])
        if first == len(chunks):
            stats = {}
        else:
            # The stats are kept in the order they were dated: the recent
            # ones are the last, taken in C.
            stats = dict(itertools.islice(stats.items(), chunks[first][1], None))
    if run["listed"] and isinstance(paths, Mapping):
        # A listed stat stands in only where content decides
        # (`file_dep_is_fresh`): a folder, or a file that could not be read,
        # is decided by its time, which a listing may report late, and each
        # check took a stat of its own of each: the 5,700 folders of a photo
        # set, before every check. The file's own stat instead, taken once
        # for the run's other checks too, when a check first asks for it.
        # Taken now, it is as recent as any caller asks: its date stays the
        # listing's, which only makes it lapse early.
        by_time = {p for p in run["listed"].intersection(paths.keys()) if _decided_by_time(paths[p])}
        if by_time:
            return _Identifying(run, {p: p for p in by_time}, stats)
    return stats


def stat_dates_this_run() -> Mapping[str, float]:
    """``{path: when}`` (``time.monotonic()``) for this run's stats
    (`stats_taken_this_run`): when the chunk each was taken in began."""
    run = file_state_this_run()
    return run.get("stat_date", {}) if run is not None else {}


def _decided_by_time(recorded: Any) -> bool:
    """Is *recorded* a snapshot entry judged by its size and time, with no
    digest to decide by (`file_dep_is_fresh`)?"""
    return isinstance(recorded, Mapping) and "size" in recorded and recorded.get("hash") is None


def _identify(run: dict, path: str) -> os.stat_result | None:
    """Replace the run's listed stat of *path* (`stats_from_listings`) by a
    stat of the file itself, which has its identity and its own times;
    None when there is none to take."""
    try:
        st = own_stat(path)
    except OSError:
        return None  # gone since: the check's own stat says so
    run["stat"][path] = st
    run["listed"].discard(path)
    return st


class _Identifying(Mapping):
    """``{key: stat}`` over the run's stats (``run["stat"]``) in which the
    paths *pending* names (``{key: path}``), listed so far, are each
    stat-ed themselves when first asked for (`_identify`) and the run
    keeps that stat. Asked for, not up front: a check answers most
    relative spellings by their absolute twins and never asks for them."""

    __slots__ = ("_by_key", "_pending", "_run")

    def __init__(self, run: dict, pending: dict[str, str], by_key: Mapping[str, Any] | None = None) -> None:
        self._run = run
        self._pending = pending
        self._by_key = run["stat"] if by_key is None else by_key

    def get(self, key: str, default: Any = None) -> Any:
        path = self._pending.pop(key, None)
        if path is not None and path in self._run["listed"]:
            st = _identify(self._run, path)
            return default if st is None else st
        return self._by_key.get(key, default)

    def __getitem__(self, key: str) -> Any:
        st = self.get(key)
        if st is None:
            raise KeyError(key)
        return st

    def __contains__(self, key: object) -> bool:
        return key in self._pending or key in self._by_key

    def __iter__(self):
        return iter({**self._by_key, **dict.fromkeys(self._pending)})

    def __len__(self) -> int:
        return len(self._by_key.keys() | self._pending.keys())

    def __bool__(self) -> bool:
        # Not by ``__len__``, which builds the union of the keys: asked for
        # each file of a component (``if stats``), that was quadratic in
        # the files, tens of seconds before a cell below 19,000 photos.
        return bool(self._by_key) or bool(self._pending)


def stat_keys_this_run(paths: Iterable[str]) -> Mapping[str, tuple[int, int, int] | None]:
    """``{path: stat_key or None}`` of this run's stats (`stats_this_run`):
    the `stat_key` of the stat that found each path, wherever that was, and
    None for a path found nowhere. Holds every path asked, and may hold more.

    Kept for the rest of the run with the stats: several cached cells above
    a loop over 24,000 documents carry the same files, and each check made
    a key of every stat again before every cell. With the keys kept, a
    check whose files are all as recorded is one comparison of two dicts'
    items, in C."""
    run = file_state_this_run()
    keys = run.setdefault("stat_key", {}) if run is not None else {}
    if not isinstance(paths, (dict, set, frozenset)):
        paths = dict.fromkeys(paths)
    if not keys.keys() >= _as_set(paths):
        for p, (_resolved, st) in stats_this_run([p for p in paths if p not in keys]).items():
            keys[p] = stat_key(st) if st is not None else None
    return keys


def file_component_this_run(
    paths: set[str], remote: set[str] | None = None, probes: Mapping[str, str] | None = None
) -> str:
    """`compute_file_hash_component` of *paths*, with this run's stats
    (`stats_this_run`) in place of a stat of each file of its own: the same
    fields of the same file in the same run, so the same component; and the
    component itself kept for the rest of the run, as the stats are. A
    directory listing's stat is not handed on (it has no file identity, and
    can lag a file edited through another hard link): the file's own stat
    is taken in its place when the component asks for it, once for the run
    (`_Identifying`), where each component built in the run took one of
    its own.

    The run's stat of a file that is no link is an ``lstat``
    (`stats_this_run`), handed on as such too: resolving each file's real
    path took one more ``lstat`` of every file the statement read, 19,000 of
    them before a cell below a read of a photo collection."""
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
    listed = run["listed"] if run is not None else None
    followed = run.get("stat_followed", ()) if run is not None else None
    stats: Mapping[str, os.stat_result] = {}
    lstats = {}
    isabs = os.path.isabs
    for p, (resolved, st) in stats_this_run(paths).items():
        if st is not None and resolved == p and (st.st_ino if listed is None else p not in listed):
            stats[normalize_path(p)] = st
            # By its own spelling only: a relative one was stat-ed from the
            # working directory of the pass.
            if followed is not None and p not in followed and isabs(p):
                lstats[p] = st
    if listed and not listed.isdisjoint(paths):
        stats = _Identifying(run, {normalize_path(p): p for p in listed.intersection(paths)}, stats)
    component = compute_file_hash_component(paths, remote, probes, stats=stats, lstats=lstats)
    if key is not None:
        components[key] = component
    return component
