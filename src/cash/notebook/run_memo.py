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
from collections.abc import Collection, Iterable, Mapping
from typing import Any

from .._paths import is_remote_url, normalize_path, resolve_file_dep_path
from .._user_attrs import attribute_or
from ..tracking import file_dep_snapshot as _fds
from ..tracking import file_watch as _fw
from ..tracking.file_dep_snapshot import LISTING_MIN_FILES, FreshnessMemo, stats_from_listings
from ..tracking.read_classification import own_lstat, own_stat, stat_key

__all__ = [
    "HeldPaths",
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
        # The changes reported since the last run, before any stat is taken
        # as current (`file_watch`).
        watch = _fw.watcher()
        if watch is not None:
            watch.drain(check_dirs=True)
        _FILE_STATE_THIS_RUN.update(
            epoch=epoch,
            memo=FreshnessMemo(taken=stats_taken_this_run, verdicts=RUN_VERDICTS),
            stat={},
            since=time.monotonic(),
            watch=watch,
            # This run's, until a write drops its stats (`_Held.unsure`).
            token=object(),
        )
    return _FILE_STATE_THIS_RUN


def forget_file_state_this_run() -> None:
    """A statement of this cell run wrote files: per-file answers taken before
    it are not answers for entries checked after it."""
    _FILE_STATE_THIS_RUN.clear()


#: How many snapshots found fresh one run keeps (`note_fresh_snapshot`).
_FRESH_SNAPSHOTS_KEPT = 16


#: Snapshots found fresh whole while the stat of each of their files was
#: current under the folder watch (`file_watch.FileWatch.cover`), each with
#: that cover and the working directory: ``(snapshot, cover, cwd)``. They
#: outlive the cell run, and hold while no folder of the cover reports a
#: change.
_COVERED_SNAPSHOTS: list[tuple[Mapping[str, Any], "_Held", str]] = []

#: How many covered snapshots are kept, across runs.
_COVERED_KEPT = 16


def note_fresh_snapshot(snap: Mapping[str, Any], at: float | None = None) -> None:
    """Remember that *snap* (a ``file_dependencies`` snapshot) was found fresh
    whole, from answers resting on stats taken at *at* (``time.monotonic()``)
    or later; by default this run's per-file answers
    (``file_state_this_run()["memo"]``), dated by when those began to be
    taken, not by when this check ran: an answer it reused is that old.
    Kept with them, so a statement that writes files drops it too.

    Kept across runs as well while every file's stat is current under the
    folder watch (`file_watch`): the verdict then holds until a folder of
    its files reports a change, and a cell below 19,000 photos checks them
    as one comparison instead of a stat each.
    """
    run = file_state_this_run()
    if run is None:
        return
    kept = run.setdefault("fresh", [])
    if len(kept) < _FRESH_SNAPSHOTS_KEPT and not any(known is snap for known, _ in kept):
        kept.append((snap, run["since"] if at is None else at))
    watch = run.get("watch")
    if watch is not None:
        _note_covered(watch, snap)


def _note_covered(watch: Any, snap: Mapping[str, Any]) -> None:
    covered = _COVERED_SNAPSHOTS
    # Found fresh again, with no folder of it reporting a change: the cover
    # is the one kept. Its loose files (a second hard link each) were what
    # the check just looked at; covering it again went through every path
    # in Python, twice per statement over a read of 13,000 photos.
    if any(known is snap and held.settled() for known, held, _cwd in covered):
        return
    held = _held_of(watch, snap)
    if held is None:
        return
    try:
        cwd = os.getcwd()
    except OSError:
        return
    # This one's earlier verdict, then the oldest, make room.
    covered[:] = [entry for entry in covered if entry[0] is not snap][-(_COVERED_KEPT - 1) :]
    covered.append((snap, held, cwd))


class _Held:
    """What the folder watch vouches for of a snapshot found fresh
    (`FileWatch.cover_by_folder`): its files by folder, the ones no folder
    vouches for (``loose``), and where the modules lie that files beside
    the code are checked in (`_CodePlaces`)."""

    __slots__ = ("code", "loose", "parts", "seen")

    def __init__(self, parts: list, loose: list[str], code: tuple | None) -> None:
        self.parts = parts
        self.loose = loose
        self.code = code
        #: The run (``file_state_this_run()["token"]``) whose stats found
        #: the loose paths as recorded, after the last `renew`.
        self.seen: object | None = None

    def renew(self, fresh: tuple[list, list[str]]) -> None:
        """The paths `unsure` named were found as recorded just now: the
        watch vouches for them again by *fresh* (`_held_keys_parts` of
        them), in place of the folders that reported a change."""
        parts, loose = fresh
        self.parts = [part for part in self.parts if part[0].alive and part[0].version == part[1]] + parts
        self.loose = loose
        self.seen = None

    def settled(self) -> bool:
        """Does the watch vouch for every folder as it did (`unsure` names
        no path but the loose ones)?"""
        if self.code is not None and _fds.code_places(self.code[0]) != self.code[1]:
            return False
        return not _fw.FileWatch.changed_in(self.parts)

    def unsure(self, run: object | None = None) -> list[str] | None:
        """The snapshot's paths to check again: those in a folder that
        reported a change since, and the loose ones; None for all of them.
        The loose ones only once in the run *run* (`seen`): its stats, which
        found them as recorded, are the ones a check would take again."""
        if self.code is not None and _fds.code_places(self.code[0]) != self.code[1]:
            return None
        changed = _fw.FileWatch.changed_in(self.parts)
        if not changed:
            return [] if run is not None and self.seen is run else list(self.loose)
        return list(dict.fromkeys(changed + self.loose)) if self.loose else list(dict.fromkeys(changed))


def _held_of(watch: Any, snap: Mapping[str, Any]) -> _Held | None:
    """`_Held` for *snap*, a relative path by its absolute spelling; None
    when a file is checked elsewhere than where it was recorded (beside the
    code: in this process's copy of that code), or most files are loose."""
    by_key: dict[str, str] = {}
    loose: list[str] = []
    isabs = os.path.isabs
    here = _fds.dep_path_for_this_process
    modules: set[str] = set()
    for p, recorded in snap.items():
        if not isinstance(recorded, Mapping):
            return None
        if "code_rel" in recorded:
            if here(p, recorded) != p:
                return None
            module = recorded.get("code_mod")
            if isinstance(module, str):
                modules.add(module)
        if isabs(p):
            by_key[p] = p
        else:
            twin = _fds.spelled_absolute(p)
            if twin is None:
                loose.append(p)
            else:
                by_key[p] = twin
    parts, unvouched = watch.cover_by_folder(by_key)
    loose += unvouched
    if 2 * len(loose) > len(snap):
        return None
    return _Held(parts, loose, (frozenset(modules), _fds.code_places(modules)) if modules else None)


def _cover_of(watch: Any, paths: Iterable[str], snap: Mapping[str, Any] | None = None) -> tuple | None:
    """The watch's cover (`FileWatch.cover`) of *paths*, a relative one by
    its absolute spelling; None when one has no current stat, or is checked
    elsewhere than where it was recorded (beside the code: in this
    process's copy of that code, `file_dep_snapshot.dep_path_for_this_process`)."""
    absolute = []
    isabs = os.path.isabs
    here = _fds.dep_path_for_this_process
    modules: set[str] = set()
    for p in paths:
        if snap is not None:
            recorded = snap[p]
            if not isinstance(recorded, Mapping):
                return None
            if "code_rel" in recorded:
                if here(p, recorded) != p:
                    return None
                module = recorded.get("code_mod")
                if isinstance(module, str):
                    modules.add(module)
        if not isabs(p):
            twin = _fds.spelled_absolute(p)
            if twin is None:
                return None
            p = twin
        absolute.append(p)
    cover = watch.cover(absolute)
    if cover is None or not modules:
        return cover
    return (*cover, (_CodePlaces(modules), _fds.code_places(modules)))


class _CodePlaces:
    """A part of a cover (`FileWatch.still`) that holds while the modules a
    snapshot's files lie beside are still loaded from where they were: a
    file beside the code is checked in this process's copy of it."""

    __slots__ = ("modules",)

    alive = True

    def __init__(self, modules: set[str]) -> None:
        self.modules = frozenset(modules)

    @property
    def version(self) -> tuple:
        return _fds.code_places(self.modules)


def fresh_snapshot_this_run(snap: Mapping[str, Any], not_before: float | None) -> bool:
    """Was a snapshot equal to *snap* found fresh whole in this run, from
    answers taken at *not_before* (``time.monotonic()``) or later, or since
    under a folder watch that has reported no change of its files?

    The upstream check looks up the entry of the cell above, and the statement
    below then checks its own entry, which inherited the same files: a frame
    built from 24,000 documents was checked twice in one cell. The caller
    says how old an answer it takes. Compared as whole dicts, in C.
    """
    return RUN_VERDICTS.known(snap, not_before) is not None


class _RunVerdicts:
    """The verdicts of whole snapshots shared by every check of a cell run
    (`FreshnessMemo.verdicts`): one check of a snapshot of 19,000 files per
    cell, where thirteen checks each went through every file."""

    __slots__ = ()

    @staticmethod
    def known(snap: Mapping[str, Any], not_before: float | None) -> float | None:
        """When the stats under a fresh verdict on *snap* were taken, if one
        was given from stats no older than *not_before* (any, for None), or
        is held by the watch; None otherwise."""
        run = file_state_this_run()
        if run is None:
            return None
        for known, at in run.get("fresh", ()):
            if (not_before is None or at >= not_before) and len(known) == len(snap) and known == snap:
                return at
        held = _RunVerdicts.partly(snap, not_before)
        return held[0] if held is not None and not held[1] else None

    @staticmethod
    def partly(snap: Mapping[str, Any], not_before: float | None) -> tuple[float, list[str]] | None:
        """``(when, unsure)`` for *snap* found fresh in an earlier check,
        held by the folder watch but for the paths *unsure* names, which the
        caller checks itself; None when no such verdict is kept."""
        run = file_state_this_run()
        if run is None:
            return None
        watch = run.get("watch")
        if watch is None or not _COVERED_SNAPSHOTS:
            return None
        cwd = None
        drained = False
        for known, held, at_cwd in _COVERED_SNAPSHOTS:
            if len(known) != len(snap):
                continue
            if cwd is None:
                try:
                    cwd = os.getcwd()
                except OSError:
                    return None
            if at_cwd != cwd or not (known is snap or known == snap):
                continue
            if not drained:
                watch.drain()
                drained = True
            unsure = held.unsure()
            if unsure is None:
                return None
            return watch.confirmed_at, unsure
        return None

    @staticmethod
    def note(snap: Mapping[str, Any], at: float | None) -> None:
        note_fresh_snapshot(snap, at)


RUN_VERDICTS = _RunVerdicts()


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
        missing = run.setdefault("missing", set())
    else:
        stats, elsewhere, dates, chunks, followed, from_listing, missing = {}, {}, {}, [], set(), set(), set()
    watch = run.get("watch") if run is not None else None
    if not isinstance(paths, (dict, set, frozenset)):
        paths = dict.fromkeys(paths)  # the paths as asked, in order, once each
    # The snapshot's entries, when asked with them: one recorded as absent is
    # looked for where it was recorded only.
    recorded_of = paths if isinstance(paths, Mapping) and not isinstance(paths, RunStats) else None
    # Asked again by each check of the run: most often every path was
    # stat-ed already, which the run's own dict tells in C.
    if stats.keys() >= _as_set(paths):
        todo = []
    else:
        # Set differences, in C: a snapshot holding paths not there (kept in
        # ``elsewhere``) went through this per path on every check.
        left = _as_set(paths) - stats.keys()
        if elsewhere:
            left -= elsewhere.keys()
        todo = [p for p in paths if p in left] if left else []
    if todo and watch is not None:
        # A stat kept while its folder was watched, with no change reported
        # since, is the file's stat now (`file_watch`): no stat is taken.
        watch.drain()
        at = time.monotonic()
        chunks.append((at, len(stats)))
        current = watch.current
        absent = watch.absent
        rest = []
        for p in todo:
            st = current(p)
            if st is None:
                if recorded_of is not None and absent(p) and _fds.is_absent_entry(recorded_of.get(p)):
                    # Vouched not there: an entry recorded as absent asks
                    # nothing more (`_looked_for_where_recorded`).
                    elsewhere[p] = (None, None)
                    missing.add(p)
                    continue
                rest.append(p)
            elif p not in stats:
                stats[p] = st
                dates[p] = at
                if watch.is_listed(p):
                    from_listing.add(p)
        todo = rest
    if todo:
        at = time.monotonic()
        # The watch keeps a listed stat of a relative path by its absolute
        # spelling, as `_as_spelled_absolute` keeps a stat of one: kept by
        # the relative spelling, which names no folder, it was never kept,
        # and every check before a cell listed those files again.
        watched_as = (
            {p: _watched_spelling(p) for p in todo} if watch is not None and len(todo) >= LISTING_MIN_FILES else None
        )
        before = (
            {p: watch.before_stat(twin) for p, twin in watched_as.items() if twin is not None}
            if watched_as is not None
            else None
        )
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
                    if before is not None and p in before:
                        watch.record(before[p], watched_as[p], st, listed=True)
        stat = os.stat
        if attribute_or(stat, "_is_file_tracker_patch", False):
            stat = stat._original_func  # as `own_stat`, once for the lot
        lstat = os.lstat
        if attribute_or(lstat, "_is_file_tracker_patch", False):
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
            if recorded_of is not None and _fds.is_absent_entry(recorded_of.get(p)):
                found = _looked_for_where_recorded(p, stat, lstat, followed, watch, missing)
                if found is None:
                    elsewhere[p] = (None, None)
                else:
                    stats[p] = found
                    dates[p] = at
                continue
            if not os.path.isabs(p) and not ("://" in p and is_remote_url(p)):
                st = _as_spelled_absolute(p, stats, stat, lstat, cwd, dates, at, followed, watch)
            if st is None and not ("://" in p and is_remote_url(p)):
                # Where it was recorded: the stat that finds it is there is
                # the stat it is asked for (`resolve_file_dep_path` asked
                # ``os.path.exists`` first, a stat of its own).
                try:
                    st = _stat_by_lstat(p, stat, lstat, followed, watch)
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


def _watched_spelling(path: str) -> str | None:
    """The spelling the folder watch keeps *path*'s stat by: itself when
    absolute, a relative one by its absolute spelling (`_as_spelled_absolute`'s
    rules: not through ``..``, not naming a drive), and None for a remote URL
    or a path with no such spelling."""
    if os.path.isabs(path):
        return path
    if "://" in path and is_remote_url(path):
        return None
    if os.path.splitdrive(path)[0]:
        return None
    return _fds.spelled_absolute(path)


#: ``FILE_ATTRIBUTE_REPARSE_POINT``: a symlink or junction on Windows.
_REPARSE_POINT = 0x400


def _looked_for_where_recorded(
    path: str, stat: Any, lstat: Any, followed: set[str], watch: Any, missing: set[str]
) -> os.stat_result | None:
    """The stat of *path*, an entry recorded as absent, or None when it is
    still not there. Looked for only where it was recorded, as its check
    looks (`file_dep_is_fresh`): the moved-project fallbacks of
    ``resolve_file_dep_path`` asked ``os.path.exists`` of several more
    paths for each of them, and a notebook over installed packages records
    thousands (the import system's probes). With the folder *watch*, a path
    not there is kept as such (`FileWatch.record_absent`) and handed on as
    `file_dep_snapshot.MISSING` (`stats_taken_this_run`)."""
    token = watch.before_stat(path) if watch is not None else None
    try:
        st = lstat(path)
    except FileNotFoundError:
        if token is not None:
            watch.record_absent(token, path)
            missing.add(path)
        return None
    except (OSError, ValueError):
        return None
    if _stat.S_ISLNK(st.st_mode) or attribute_or(st, "st_file_attributes", 0) & _REPARSE_POINT:
        followed.add(path)
        try:
            return stat(path)
        except (OSError, ValueError):
            return None
    if token is not None:
        watch.record(token, path, st)
    return st


def _stat_by_lstat(path: str, stat: Any, lstat: Any, followed: set[str], watch: Any = None) -> os.stat_result:
    """``stat(path)``, taken as ``lstat(path)`` unless *path* is a link or
    reparse point, which is stat-ed as well and noted in *followed*.

    With the folder *watch*, its current stat of *path* when it has one,
    and otherwise the stat taken is kept by it (`FileWatch.record`), the
    folder's watch armed first. Not a link's: its target may lie in a
    folder nobody watches."""
    token = None
    if watch is not None:
        st = watch.current(path)
        if st is not None:
            return st
        token = watch.before_stat(path)
    try:
        st = lstat(path)
    except FileNotFoundError:
        if token is not None:
            # Not there, which a component built over it rests on too
            # (`FileWatch.cover`); the caller still looks elsewhere.
            watch.record_absent(token, path)
        raise
    if _stat.S_ISLNK(st.st_mode) or attribute_or(st, "st_file_attributes", 0) & _REPARSE_POINT:
        followed.add(path)
        return stat(path)
    if token is not None:
        watch.record(token, path, st)
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
    watch: Any = None,
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
            st = _stat_by_lstat(absolute, stat, lstat, followed, watch)
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
    missing = run.get("missing")
    chunks = run.get("stat_chunks")
    if not_before is not None and chunks and chunks[0][0] < not_before:
        first = bisect.bisect_left(chunks, not_before, key=lambda chunk: chunk[0])
        older = len(stats) if first == len(chunks) else chunks[first][1]
        # The stats are kept in the order they were dated: the recent
        # ones are the last, taken in C.
        recent = dict(itertools.islice(stats.items(), older, None)) if older < len(stats) else {}
        watch = run.get("watch")
        if watch is not None and older:
            # An older stat the folder watch still holds is as recent as its
            # last drain (`stat_dates_this_run`): a check of 19,000 files
            # that took longer than the answers last stat-ed each again.
            watch.drain()
            current = watch.current
            held = {p: st for p, st in itertools.islice(stats.items(), older) if current(p) is st}
            if held:
                held.update(recent)
                recent = held
        stats = recent
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
            stats = _Identifying(run, {p: p for p in by_time}, stats)
    if missing:
        watch = run.get("watch")
        if watch is None:
            return stats
        if not_before is not None:
            # As recent as the caller asks: still vouched for after a drain.
            watch.drain()
            missing = {p for p in missing if watch.absent(p)}
        return _WithMissing(stats, missing)
    return stats


class _WithMissing(Mapping):
    """The run's stats, and `file_dep_snapshot.MISSING` for each path the
    folder watch vouches is not there (`_looked_for_where_recorded`)."""

    __slots__ = ("_missing", "_stats")

    def __init__(self, stats: Mapping[str, Any], missing: set[str]) -> None:
        self._stats = stats
        self._missing = missing

    def get(self, key: str, default: Any = None) -> Any:
        st = self._stats.get(key)
        if st is not None:
            return st
        return _fds.MISSING if key in self._missing else default

    def __getitem__(self, key: str) -> Any:
        st = self.get(key)
        if st is None:
            raise KeyError(key)
        return st

    def __contains__(self, key: object) -> bool:
        return key in self._stats or key in self._missing

    def __iter__(self):
        yield from self._stats
        yield from (p for p in self._missing if p not in self._stats)

    def __len__(self) -> int:
        return len(self._stats) + sum(1 for p in self._missing if p not in self._stats)

    def __bool__(self) -> bool:
        return bool(self._stats) or bool(self._missing)


def stat_dates_this_run() -> Mapping[str, float]:
    """``{path: when}`` (``time.monotonic()``) for this run's stats
    (`stats_taken_this_run`): when the chunk each was taken in began."""
    run = file_state_this_run()
    if run is None:
        return {}
    dates = run.get("stat_date", {})
    watch = run.get("watch")
    return dates if watch is None else _WatchedDates(dates, run["stat"], watch)


class _WatchedDates(Mapping):
    """``{path: when}`` over the run's stat dates, in which a stat the
    folder watch still holds (`FileWatch.current`) is dated by the watch's
    last drain: no change was reported to its folder up to then."""

    __slots__ = ("_dates", "_stats", "_watch")

    def __init__(self, dates: dict[str, float], stats: dict[str, Any], watch: Any) -> None:
        self._dates = dates
        self._stats = stats
        self._watch = watch

    def get(self, path: str, default: Any = None) -> Any:
        at = self._dates.get(path)
        if at is None:
            return default
        watch = self._watch
        if at < watch.confirmed_at:
            st = self._stats.get(path)
            if st is not None and watch.current(path) is st:
                return watch.confirmed_at
        return at

    def __getitem__(self, path: str) -> float:
        at = self.get(path)
        if at is None:
            raise KeyError(path)
        return at

    def __contains__(self, path: object) -> bool:
        return path in self._dates

    def __iter__(self):
        return iter(self._dates)

    def __len__(self) -> int:
        return len(self._dates)


def _decided_by_time(recorded: Any) -> bool:
    """Is *recorded* a snapshot entry judged by its size and time, with no
    digest to decide by (`file_dep_is_fresh`)?"""
    return isinstance(recorded, Mapping) and "size" in recorded and recorded.get("hash") is None


def _identify(run: dict, path: str) -> os.stat_result | None:
    """Replace the run's listed stat of *path* (`stats_from_listings`) by a
    stat of the file itself, which has its identity and its own times;
    None when there is none to take."""
    watch = run.get("watch")
    try:
        if watch is not None:
            # Kept by the folder watch in place of the listing's stat.
            token = watch.before_stat(path)
            st = own_lstat(path)
            if _stat.S_ISLNK(st.st_mode) or attribute_or(st, "st_file_attributes", 0) & _REPARSE_POINT:
                st = own_stat(path)
            else:
                watch.record(token, path, st)
        else:
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
        # Asked once per file by a truth test (`stats.get(p) if stats`): the
        # union of 38,000 keys each time cost 26 s a cell on a photo set.
        by_key = self._by_key
        return len(by_key) + sum(1 for key in self._pending if key not in by_key)

    def __bool__(self) -> bool:
        return bool(self._pending) or bool(self._by_key)


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
        asked = (
            {p: entry for p, entry in paths.items() if p not in keys}
            if isinstance(paths, dict)
            else [p for p in paths if p not in keys]
        )
        for p, (_resolved, st) in stats_this_run(asked).items():
            keys[p] = stat_key(st) if st is not None else None
    return keys


def stat_keys_as_recorded(recorded: Mapping[str, tuple[int, int, int] | None]) -> bool:
    """Is every path of *recorded* (``{path: stat_key}``) found now with the
    `stat_key` recorded for it (`stat_keys_this_run`)? False when one is
    not, or was recorded as found nowhere (None).

    Asked of each cached cell's files before every cell: below a cell that
    read 25,000 files, every trivial cell went through each of them in
    Python, a stat or a look-up in the folder watch each. A verdict found
    while the watch held a stat of every file (`file_watch`) is kept
    across runs, with the folders vouching for them (`_Held`): while none
    of them reports a change it holds, and a later ask checks only the files
    of a folder that did, and those no folder vouches for (a link, a file
    with a second hard link, a folder that cannot be watched). The cost of
    the check is then that of what changed, not of what was read.

    The very mapping asked again is found by its identity, before any of
    its paths is looked at: below a loop over 50,000 review files each
    trivial cell compared each cached cell's 100,000 recorded keys with the
    held copy, and looked for a path recorded as found nowhere among them,
    twice a cell. So a mapping asked again must not have been changed in
    place since (the simulation's snapshots never are); a changed record is
    another mapping, compared whole."""
    run = file_state_this_run()
    watch = run.get("watch") if run is not None else None
    # Held only once found with every key as recorded, none of them None.
    held = _held_keys_of(watch, recorded) if watch is not None else None
    if held is None and None in recorded.values():
        return False
    if held is not None:
        # Each cached cell below a read carries an equal dict, held once:
        # its loose files (a second hard link each) were looked over, and
        # the watch asked of each, once per cached cell before every cell.
        token = run.get("token") if run is not None else None
        unsure = held.unsure(token)
        if unsure is not None:
            if not unsure:
                return True
            part = {p: recorded[p] for p in unsure}
            if part.items() <= stat_keys_this_run(part).items():
                held.renew(_held_keys_parts(watch, part))
                held.seen = token
                return True
            return False
    if not recorded.items() <= stat_keys_this_run(recorded).items():
        return False
    if watch is not None:
        _note_held_keys(watch, recorded, run.get("token") if run is not None else None)
    return True


#: Stat-key dicts found as recorded while the folder watch held a stat of
#: their files (`stat_keys_as_recorded`), across runs: each as a copy, with
#: its `_Held`, the working directory, the watch and the mapping it was
#: found for, newest last. Found by that mapping's identity, else by
#: comparing the copy whole, in C: a dict built anew with the same keys is
#: answered too, and another with other files, or other keys, never is.
_HELD_KEYS: list[tuple[dict, "_Held", str, Any, Mapping[str, Any]]] = []

#: How many paths the copies in `_HELD_KEYS` hold at most, all told.
_HELD_KEYS_PATHS = 500_000


def _held_keys_of(watch: Any, recorded: Mapping[str, Any]) -> _Held | None:
    n = len(recorded)
    found = None
    for entry in reversed(_HELD_KEYS):
        copy, _held, _cwd, by, source = entry
        if source is recorded and by is watch and len(copy) == n:
            found = entry
            break
    if found is None:
        for entry in reversed(_HELD_KEYS):
            copy, _held, _cwd, by, _source = entry
            if len(copy) == n and by is watch and copy == recorded:
                # Asked again, the equal mapping is found by its identity.
                found = (copy, _held, _cwd, by, recorded)
                _HELD_KEYS[_HELD_KEYS.index(entry)] = found
                break
    if found is None:
        return None
    try:
        here = os.getcwd()
    except OSError:
        return None
    if found[2] != here:
        return None
    # The run's first drain is the one before any stat (`file_state_this_run`);
    # this one takes what a statement of the run wrote since.
    watch.drain()
    return found[1]


def _held_keys_parts(watch: Any, recorded: Mapping[str, Any]) -> tuple[list, list[str]]:
    """`FileWatch.cover_by_folder` of the paths of *recorded* whose stat
    the watch holds now with the recorded `stat_key` (a relative one by its
    absolute spelling); the rest are loose.

    Asked only once every path of *recorded* was found with its recorded key
    this run (`stat_keys_this_run`): a stat the watch holds that is the very
    stat the run keyed is as recorded without a key made of it again."""
    by_key: dict[str, str] = {}
    loose: list[str] = []
    current = watch.current
    isabs = os.path.isabs
    spelled = _fds.spelled_absolute
    run_stats = stats_this_run(recorded)
    for p, key in recorded.items():
        twin = p if isabs(p) else spelled(p)
        st = current(twin) if twin is not None else None
        if st is None:
            loose.append(p)
            continue
        found = run_stats.get(p)
        if found is not None and found[1] is st:
            by_key[p] = twin
        elif stat_key(st) != key:
            loose.append(p)
        else:
            by_key[p] = twin
    parts, unvouched = watch.cover_by_folder(by_key)
    return parts, loose + unvouched


#: Stat-key dicts `_note_held_keys` found the watch could not vouch for,
#: across runs: ``[copy, working directory, watch, runs to pass, runs
#: passed after the last refusal, run last asked in]``, newest last.
_REFUSED_KEYS: list[list[Any]] = []

#: How many refused dicts `_REFUSED_KEYS` keeps.
_REFUSED_KEYS_KEPT = 8


def _refused_keys_entry(watch: Any, recorded: Mapping[str, Any], cwd: str) -> list[Any] | None:
    n = len(recorded)
    for entry in _REFUSED_KEYS:
        copy = entry[0]
        if entry[2] is watch and entry[1] == cwd and len(copy) == n and copy == recorded:
            return entry
    return None


def _note_held_keys(watch: Any, recorded: Mapping[str, Any], token: object | None = None) -> None:
    """Hold the verdict that *recorded* is as recorded across runs
    (`_HELD_KEYS`), when the watch vouches for most of its files.

    A dict the watch does not vouch for (files with a second hard link, as
    a dataset cache leaves them) is not looked over again in the runs that
    follow its refusal: one more after the first, then two, four, up to
    `_REFUSED_OFFERS_PASSED`; nor again in the run that refused it, when
    the snapshot of each cell below the read carries an equal dict.
    Looking at every path only to refuse it cost a pass in Python over
    each file, per cached cell, before every cell (1.2 s of a cell below
    13,000 photos on Windows). Passing over it changes no answer: the
    check that is not held is made in full each time."""
    try:
        cwd = os.getcwd()
    except OSError:
        return
    epoch = _fds.HASH_EPOCH
    refused = _refused_keys_entry(watch, recorded, cwd) if _REFUSED_KEYS else None
    if refused is not None:
        if refused[5] == epoch:
            return
        refused[5] = epoch
        if refused[3] > 0:
            refused[3] -= 1
            return
    parts, loose = _held_keys_parts(watch, recorded)
    if 2 * len(loose) > len(recorded):
        if refused is None:
            _REFUSED_KEYS[:] = _REFUSED_KEYS[-(_REFUSED_KEYS_KEPT - 1) :] + [[dict(recorded), cwd, watch, 1, 1, epoch]]
        else:
            refused[4] = min(2 * refused[4], _REFUSED_OFFERS_PASSED)
            refused[3] = refused[4]
        return
    if refused is not None:
        _REFUSED_KEYS.remove(refused)
    copy = dict(recorded)
    kept = [entry for entry in _HELD_KEYS if not (len(entry[0]) == len(copy) and entry[0] == copy)]
    held = _Held(parts, loose, None)
    held.seen = token  # every file was found as recorded with this run's stats
    kept.append((copy, held, cwd, watch, recorded))
    total = 0
    for at in range(len(kept) - 1, -1, -1):
        total += len(kept[at][0])
        if total > _HELD_KEYS_PATHS and at < len(kept) - 1:
            kept = kept[at + 1 :]
            break
    _HELD_KEYS[:] = kept


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
    from .statement import file_deps
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
    watch = run.get("watch") if run is not None and key is not None and not probes else None
    place = None
    if watch is not None:
        # Built in an earlier run from stats the folder watch still holds:
        # the same fields of the same files, so the same component. Asked
        # before any stat of them: a stat of each was the cost it saves.
        try:
            place = (os.getcwd(), file_deps.get_notebook_path())
        except (OSError, ValueError):
            watch = None
        held = _COVERED_COMPONENTS.get(key) if watch is not None else None
        if held is not None and held[2] == place:
            watch.drain()
            if held[1].holds(watch, run):
                components[key] = held[0]
                return held[0]
    taken = stats_this_run(paths)  # it starts the run's records read below
    # What the watch vouches for now, before the component is built from
    # the stats it holds.
    parts = _component_parts(watch, paths) if watch is not None else None
    listed = run["listed"] if run is not None else None
    followed = run["stat_followed"] if run is not None else None
    seen = None
    if key is not None and not probes and len(paths) >= _SEEN_MIN_FILES:
        # Built in an earlier run from stats the run found again unchanged:
        # the same fields of the same files, so the same component.
        seen = _seen_as(taken, listed, followed)
        held_seen = _SEEN_COMPONENTS.get(key) if seen is not None else None
        if held_seen is not None and held_seen[1] == seen:
            _SEEN_COMPONENTS[key] = _SEEN_COMPONENTS.pop(key)  # the newest
            components[key] = held_seen[0]
            return held_seen[0]
    stats: Mapping[str, os.stat_result] = {}
    lstats = {}
    isabs = os.path.isabs
    for p, (resolved, st) in taken.items():
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
    if seen is not None:
        # Under the versions known now: building it may have linked some
        # (a file written in this process is versioned by its bytes), and
        # the next build takes those.
        from ..tracking.file_versions import versions_seen

        _keep_seen(key, component, (*seen[:-1], versions_seen()))
    if watch is not None and parts is not None:
        hold = _ComponentHold.of(watch, parts, stats_this_run(paths), run)
        if hold is not None:
            if len(_COVERED_COMPONENTS) >= _COVERED_KEPT and key not in _COVERED_COMPONENTS:
                _COVERED_COMPONENTS.pop(next(iter(_COVERED_COMPONENTS)))
            _COVERED_COMPONENTS[key] = (component, hold, place)
    return component


#: Components built in an earlier run, across runs, each with what it was
#: built from (`_seen_as`): ``{key: (component, seen)}``, newest last.
_SEEN_COMPONENTS: dict[tuple, tuple[str, tuple]] = {}

#: How many components `_SEEN_COMPONENTS` keeps, and how many paths all told.
_SEEN_KEPT = 16
_SEEN_PATHS = 500_000

#: Fewer files than this are built afresh: comparing costs about as much.
_SEEN_MIN_FILES = 64


def _seen_as(taken: RunStats, listed: Collection[str] | None, followed: Collection[str] | None) -> tuple | None:
    """What `compute_file_hash_component` builds the component of *taken*'s
    paths from, besides the paths: the run's stat of each (its kind,
    identity, size and times; None when not there), the real path of each
    folder, the working directory, the notebook's real path, and the file
    versions known (`file_versions.versions_seen`). None when a path is
    answered otherwise than by its own stat where it was recorded: found
    elsewhere, through a link, by a listing, or spelled with a short name.

    The files that the folder watch cannot vouch for (a second hard link
    each, as a dataset cache leaves them) are stat-ed again each run, and
    the component of the 13,000 photos below a read was then built again
    from those stats in each check that asked for it: a real path, a
    version and a line of text a file, 1 s a cell on Windows. Found
    unchanged, it is the same component."""
    from ..tracking.file_versions import versions_seen
    from .statement.file_deps import get_notebook_path

    by_path: dict[str, tuple | None] = {}
    folders: set[str] = set()
    dirname = os.path.dirname
    for p, (resolved, st) in taken.items():
        if st is None:
            if resolved is not None:
                return None
            by_path[p] = None
            continue
        if resolved != p or "~" in p or (followed and p in followed) or (listed and p in listed):
            return None
        by_path[p] = (st.st_mode, st.st_ino, st.st_dev, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
        folders.add(dirname(p))
    try:
        cwd = os.getcwd()
        notebook = get_notebook_path()
        real = {folder: _fds.realpath_this_run(folder or ".") for folder in folders}
        if notebook:
            notebook = _fds.realpath_this_run(notebook)
    except (OSError, ValueError):
        return None
    return by_path, real, cwd, notebook, versions_seen()


def _keep_seen(key: tuple, component: str, seen: tuple) -> None:
    """Keep *component*, built from *seen* (`_seen_as`), for later runs."""
    _SEEN_COMPONENTS.pop(key, None)
    _SEEN_COMPONENTS[key] = (component, seen)
    total = sum(len(kept[1][0]) for kept in _SEEN_COMPONENTS.values())
    while len(_SEEN_COMPONENTS) > 1 and (len(_SEEN_COMPONENTS) > _SEEN_KEPT or total > _SEEN_PATHS):
        total -= len(_SEEN_COMPONENTS.pop(next(iter(_SEEN_COMPONENTS)))[1][0])


#: Components built from stats the folder watch held, across runs:
#: ``{key: (component, hold, (cwd, notebook path))}`` (`file_component_this_run`).
_COVERED_COMPONENTS: dict[tuple, tuple[str, "_ComponentHold", tuple]] = {}


def _component_parts(watch: Any, paths: Iterable[str]) -> list | None:
    """`FileWatch.cover_by_folder` of *paths*, a relative one by its
    absolute spelling; None when the watch does not vouch for each of them
    (a link, a path through ``..``, a folder that cannot be watched)."""
    by_key: dict[str, str] = {}
    isabs = os.path.isabs
    spelled = _fds.spelled_absolute
    for p in paths:
        twin = p if isabs(p) else spelled(p)
        if twin is None:
            return None
        by_key[p] = twin
    parts, loose = watch.cover_by_folder(by_key)
    return None if loose else parts


class _ComponentHold:
    """What a file component (`file_component_this_run`) built while the
    folder watch vouched for each of its files rests on: the watched
    folders, each with its version and its paths (`FileWatch.cover_by_folder`),
    and the stat each path was found with (None: not there).

    While no folder reports a change the component holds as it is. A folder
    that reports one has only its own paths stat-ed again: each found where
    it was, no link, of the same kind and with the same `stat_key` as before
    gives the same component, and the folder is held again from now. Kept
    whole, the component of a frame read from 50,000 files was built again
    (a real path and a stat of each file, opening each on Windows) whenever
    anything was written beside one of them, or the notebook was saved in
    the folder of the one CSV it was round-tripped through."""

    __slots__ = ("parts", "rows")

    def __init__(self, parts: list, rows: list[list[Any]]) -> None:
        self.parts = parts
        self.rows = rows

    @classmethod
    def of(cls, watch: Any, parts: list, found: RunStats, run: dict | None) -> "_ComponentHold | None":
        """The hold of *parts*, whose paths the run found as *found*
        (`stats_this_run`); None when a folder reported a change since
        *parts* were taken, or a path was found elsewhere, through a link,
        or with another stat than the watch holds."""
        if run is None or _fw.FileWatch.changed_in(parts):
            return None
        rows = cls._rows(watch, parts, found, run)
        return None if rows is None else cls(parts, rows)

    @staticmethod
    def _rows(watch: Any, parts: list, found: RunStats, run: dict) -> list[list[Any]] | None:
        """The stat the watch holds of each path of *parts* (None: held as
        not there), each the stat the run found it with too."""
        followed = run.get("stat_followed", ())
        current = watch.current
        isabs = os.path.isabs
        spelled = _fds.spelled_absolute
        rows = []
        for _d, _version, keys in parts:
            row = []
            for p in keys:
                resolved, st = found[p]
                held = current(p if isabs(p) else spelled(p))
                if held is None:
                    if st is not None:
                        return None
                elif st is None or resolved != p or p in followed or (st is not held and not _same_file(st, held)):
                    return None
                row.append(held)
            rows.append(row)
        return rows

    def holds(self, watch: Any, run: dict | None) -> bool:
        """Is the component still the one these files give? Looks only at
        the paths of a folder that reported a change since."""
        changed = {at for at, (d, version, _keys) in enumerate(self.parts) if not (d.alive and d.version == version)}
        if not changed:
            return True
        if run is None or any(not self.parts[at][0].alive for at in changed):
            return False  # moved, or no longer watched: its real paths may be others
        asked = [p for at in sorted(changed) for p in self.parts[at][2]]
        found = stats_this_run(asked)  # takes again what the watch no longer holds
        fresh = _component_parts(watch, asked)
        if fresh is None:
            return False
        rows = self._rows(watch, fresh, found, run)
        if rows is None or _fw.FileWatch.changed_in(fresh):
            return False
        before = {p: st for at in changed for p, st in zip(self.parts[at][2], self.rows[at])}
        for (_d, _version, keys), row in zip(fresh, rows):
            for p, st in zip(keys, row):
                was = before[p]
                if (st is None) != (was is None) or (st is not was and st is not None and not _same_file(st, was)):
                    return False
        # A folder's own part (a listed folder is vouched for by its own
        # watch and its holder's) asked again through the other is replaced.
        renewed = {id(d) for d, _version, _keys in fresh}
        asked_once = set(asked)
        kept = [
            at
            for at, (d, _version, keys) in enumerate(self.parts)
            if at not in changed and not (id(d) in renewed and asked_once.issuperset(keys))
        ]
        self.parts = [self.parts[at] for at in kept] + fresh
        self.rows = [self.rows[at] for at in kept] + rows
        return True


def _same_file(a: os.stat_result, b: os.stat_result) -> bool:
    """The same kind of entry with the same `stat_key`: what a file
    component is built from (`compute_file_hash_component`)."""
    return _stat.S_IFMT(a.st_mode) == _stat.S_IFMT(b.st_mode) and stat_key(a) == stat_key(b)


#: The most offers of a refused set `HeldPaths.worth_keeping` passes over
#: before `HeldPaths.keep` looks at it again.
_REFUSED_OFFERS_PASSED = 64


class HeldPaths:
    """What a caller worked out from this run's stats of a set of paths
    (`stats_this_run`), kept across cell runs while the folder watch vouches
    for each of them: their folders with their versions (`_Held`).

    Below a loop over 50,000 review files every trivial cell worked the
    same things out again from all of them -- which of them a writer's file
    is (`file_writers`), each path's comparable spellings, the file
    component of the statements over the frame -- a stat or a watch look-up
    of each, and their spellings, in Python: 2-3.6 s a cell on Windows,
    where plain Python takes 0.05 s. Kept, a later run asks only of the
    paths of a folder that reported a change since, and of those no folder
    vouches for (`find`).

    Found by an equal set of paths, compared whole in C, in the same working
    directory, under the same watch.

    A set the watch does not vouch for most of is not kept, and is not
    offered again each run (`worth_keeping`): the look at every path that
    found it so (`_vouched`) cost as much as the answer it would have held,
    each cell, for nothing -- 0.3 s of a trivial cell below 13,000 photos
    on Linux, more on Windows."""

    __slots__ = ("_entries", "_kept", "_refused")

    def __init__(self, kept: int = 4) -> None:
        #: ``[paths, working directory, watch, _Held, value]``, newest last.
        self._entries: list[list[Any]] = []
        self._kept = kept
        #: ``[paths, working directory, watch, offers to pass, offers passed
        #: after the last refusal]`` of each set `keep` could not keep,
        #: newest last (`worth_keeping`).
        self._refused: list[list[Any]] = []

    def find(self, paths: Collection[str]) -> tuple[list[Any], Any, list[str]] | None:
        """``(entry, value, unsure)`` kept for a set equal to *paths*:
        *unsure* the paths to look at again (an empty list when the value
        holds as it is), handed back to `renew` with *entry* once the value
        is brought up to date. None when nothing is kept, or the watch no
        longer vouches for it (a module the files lie beside moved)."""
        run = file_state_this_run()
        watch = run.get("watch") if run is not None else None
        if watch is None or not self._entries:
            return None
        try:
            here = os.getcwd()
        except OSError:
            return None
        n = len(paths)
        for entry in reversed(self._entries):
            copy, cwd, by, held, value = entry
            if by is watch and cwd == here and len(copy) == n and (copy is paths or copy == paths):
                # The run's first drain is the one before any stat; this one
                # takes what a statement of the run wrote since.
                watch.drain()
                unsure = held.unsure()
                if unsure is None:
                    self._entries.remove(entry)
                    return None
                return entry, value, unsure
        return None

    def keep(self, paths: Collection[str], value: Any, found: Mapping[str, Any], loose: Iterable[str] = ()) -> None:
        """Keep *value*, worked out from *found* (``{path: stat or None}``,
        this run's stat of each of *paths*), while the watch holds that very
        stat of each (or holds it as not there); *loose* paths are asked
        again by each `find` whatever the watch says. Nothing is kept when
        most paths are loose."""
        run = file_state_this_run()
        watch = run.get("watch") if run is not None else None
        if watch is None:
            return
        try:
            here = os.getcwd()
        except OSError:
            return
        parts, unvouched = _vouched(watch, paths, found, set(loose))
        copy = paths if isinstance(paths, frozenset) else frozenset(paths)
        n = len(copy)
        refused = self._refused_entry(copy, here, watch)
        if 2 * len(unvouched) > len(paths):
            # Passed over by the next offers, twice as many after each
            # refusal: a set the watch comes to vouch for is still kept.
            if refused is None:
                self._refused = self._refused[-(self._kept - 1) :] + [[copy, here, watch, 1, 1]]
            else:
                refused[4] = min(2 * refused[4], _REFUSED_OFFERS_PASSED)
                refused[3] = refused[4]
            return
        if refused is not None:
            self._refused.remove(refused)
        self._entries = [e for e in self._entries if not (len(e[0]) == n and e[0] == copy)][-(self._kept - 1) :]
        self._entries.append([copy, here, watch, _Held(parts, unvouched, None), value])

    def worth_keeping(self, paths: Collection[str]) -> bool:
        """Whether to offer *paths* to `keep` this time, and so to work out
        what it takes. No while the watch is off, and for the next offers
        of a set equal to one `keep` refused (the watch vouched for less
        than half of it): one more after each refusal, then two, four, up
        to `_REFUSED_OFFERS_PASSED`. A set the caller does not keep is
        worked out afresh each time, so passing over one changes no
        answer."""
        run = file_state_this_run()
        watch = run.get("watch") if run is not None else None
        if watch is None:
            return False
        if not self._refused:
            return True
        try:
            here = os.getcwd()
        except OSError:
            return False
        refused = self._refused_entry(paths, here, watch)
        if refused is None or refused[3] <= 0:
            return True
        refused[3] -= 1
        return False

    def _refused_entry(self, paths: Collection[str], here: str, watch: Any) -> list[Any] | None:
        n = len(paths)
        for entry in self._refused:
            copy, cwd, by = entry[0], entry[1], entry[2]
            if by is watch and cwd == here and len(copy) == n and (copy is paths or copy == paths):
                return entry
        return None

    @staticmethod
    def renew(
        entry: list[Any], value: Any, found: Mapping[str, Any], unsure: list[str], loose: Iterable[str] = ()
    ) -> None:
        """*entry* (`find`) holds again with *value*: its *unsure* paths were
        looked at again this run and found as *found*; *loose* still are."""
        _copy, _cwd, watch, held, _value = entry
        held.renew(_vouched(watch, unsure, found, set(loose)))
        entry[4] = value


def _vouched(watch: Any, paths: Iterable[str], found: Mapping[str, Any], loose: set[str]) -> tuple[list, list[str]]:
    """`FileWatch.cover_as_found` of each of *paths* with its stat in
    *found* (None: not there), by the spelling the watch keeps it under
    (`_watched_spelling`); those with no such spelling, and *loose*, are
    loose."""
    out: list[str] = []
    asked: list[tuple[str, str, Any]] = []
    isabs = os.path.isabs
    get = found.get
    for p in paths:
        twin = p if isabs(p) else _watched_spelling(p)
        if twin is None or p in loose:
            out.append(p)
        else:
            asked.append((p, twin, get(p)))
    parts, unvouched = watch.cover_as_found(asked, _same_file)
    return parts, out + unvouched
