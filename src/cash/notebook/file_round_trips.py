"""A notebook that hands a result from one statement to the next through a
file, written in this session and read back (``NOTEBOOK-FILE-ROUNDTRIP``).

One cell writes ``report.xlsx``; the next opens it again, formats it, saves
it and reads it back to check it. Every write gives the file new bytes (an
Excel file's zip timestamps, a CSV's float formatting being enough), so the
statement that reads it runs again every time, and so does everything built
from what it read: nothing below the read can be reused. Cash cannot fix
that; the user can, by keeping the data in a variable and writing the file
once, at the end.

Built from what the statement processor records anyway: the files a
statement was seen writing (``write_observer``), the files it read
(``FileAccessTracker``), and the file the freshness check found changed when
it threw the statement's entry away. Nothing is stat'ed or resolved unless a
statement wrote a file, or read one this session wrote.

Only a file a statement of this session wrote counts: a checkpoint an earlier
kernel wrote and this one reads is what files are for. Only when it costs: the
reading statement and the statements that ran on what it read took at least
:data:`MIN_COST_S` together. Once per file per session.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable, Set
from dataclasses import dataclass, field

from cash._paths import normalize_path
from cash.diagnostics import log_diagnostic, warn_diagnostic
from cash.exceptions import CashCacheIneffectiveWarning
from cash.tracking.file_dep_snapshot import realpath_this_run
from cash.tracking.file_versions import file_version

logger = logging.getLogger(__name__)

__all__ = ["CODE", "MIN_COST_S", "FileRoundTrips"]

CODE = "NOTEBOOK-FILE-ROUNDTRIP"

#: What the reread and the statements run on it must cost before the hint is
#: worth a line: below this, rereading the file is cheaper than reading it.
MIN_COST_S = 1.0

#: Files written this session that are remembered; past this, the oldest go
#: (and the versions statements read of them start afresh). A loop writing
#: thousands of files is not a round trip anyone reads back.
_MAX_WRITTEN = 2048

#: How much of a statement the message quotes.
_QUOTE_CHARS = 80


def quote(code: str) -> str:
    """The first line of *code*, cut to :data:`_QUOTE_CHARS`, in backticks."""
    lines = [line.strip() for line in code.strip().splitlines() if line.strip()]
    first = lines[0] if lines else ""
    if len(first) > _QUOTE_CHARS or len(lines) > 1:
        first = first[: _QUOTE_CHARS - 3].rstrip() + "..."
    return f"`{first}`"


def _shown(path: str) -> str:
    """*path* relative to the working directory when it is below it."""
    try:
        rel = os.path.relpath(path)
    except ValueError:  # another drive
        return path
    return path if rel.startswith("..") else normalize_path(rel)


def _canonical(path: str) -> str:
    """The spelling the file tracker records a read of *path* under: its
    directory resolved (once per run), normalised."""
    parent, name = os.path.split(path)
    return normalize_path(os.path.join(realpath_this_run(parent), name) if parent else realpath_this_run(path))


@dataclass
class _Chain:
    """A detected round trip of one file, not costly enough yet: the names
    the reread bound, and what the statements run on them cost so far."""

    writer: str
    reader: str
    names: set[str]
    cost: float


@dataclass
class FileRoundTrips:
    """What this session wrote, and the rereads of it that cost a rerun."""

    #: canonical path -> (source hash, text) of the statement that last wrote it.
    _writers: dict[str, tuple[str, str]] = field(default_factory=dict)
    #: (reader's source hash, canonical path) -> the file's version it read.
    _read_versions: dict[tuple[str, str], tuple] = field(default_factory=dict)
    #: canonical path -> a round trip found, while its cost adds up.
    _chains: dict[str, _Chain] = field(default_factory=dict)
    #: Files already warned about.
    _warned: set[str] = field(default_factory=set)

    def __len__(self) -> int:
        """How many things it remembers: none once :meth:`clear` ran."""
        return len(self._writers) + len(self._read_versions) + len(self._chains) + len(self._warned)

    def clear(self) -> None:
        """Forget the session, as a restarted kernel does."""
        self._writers.clear()
        self._read_versions.clear()
        self._chains.clear()
        self._warned.clear()

    def after_run(
        self,
        code: str,
        source_hash: str,
        *,
        read: Set[str],
        written: Iterable[str],
        stale_file: str | None,
        inputs: Iterable[str],
        outputs: Iterable[str],
        cost: float,
    ) -> None:
        """Note a statement that just ran: *code* (its text as the user wrote
        it; keyed by *source_hash*) read *read* and wrote *written*; its cache entry was thrown away
        because *stale_file* changed (None when it was not); it took *cost*
        seconds, reading *inputs* and binding *outputs*."""
        if self._chains:
            self._grow_chains(inputs, outputs, cost)
        if self._writers and read:
            self._find_round_trips(code, source_hash, read, stale_file, outputs, cost)
        for path in written:
            self._note_written(_canonical(path), source_hash, code)

    def _note_written(self, path: str, source_hash: str, code: str) -> None:
        self._writers.pop(path, None)  # the newest last, for the cap below
        self._writers[path] = (source_hash, code)
        # A new write starts a new round: what the last reread cost is spent.
        self._chains.pop(path, None)
        if len(self._writers) > _MAX_WRITTEN:
            del self._writers[next(iter(self._writers))]

    def _find_round_trips(
        self,
        code: str,
        source_hash: str,
        read: Set[str],
        stale_file: str | None,
        outputs: Iterable[str],
        cost: float,
    ) -> None:
        """For each file of *read* this session wrote: was it rewritten since
        this statement last used it, which is why the statement ran?"""
        writers = self._writers
        if len(read) < len(writers):
            hits = [p for p in read if p in writers]
        else:
            hits = [p for p in writers if p in read]
        stale = normalize_path(stale_file) if stale_file else None
        for path in hits:
            writer_hash, writer = writers[path]
            if writer_hash == source_hash:
                continue  # its own file from its last run: a rewrite, not a hand-over
            try:
                version = file_version(path, os.stat(path))
            except (OSError, ValueError):
                continue
            key = (source_hash, path)
            before = self._read_versions.get(key)
            if len(self._read_versions) >= _MAX_WRITTEN and key not in self._read_versions:
                self._read_versions.clear()
            self._read_versions[key] = version
            # Its entry was dropped because the file changed, or this session
            # ran it before on another version of the file.
            if path != stale and (before is None or before == version):
                continue
            if path in self._warned:
                continue
            chain = _Chain(writer, code, set(outputs), cost)
            if chain.cost >= MIN_COST_S:
                self._warn(path, chain)
            else:
                self._chains[path] = chain

    def _grow_chains(self, inputs: Iterable[str], outputs: Iterable[str], cost: float) -> None:
        """Add a statement that ran on what a reread bound to its chain."""
        inputs = set(inputs)
        for path, chain in list(self._chains.items()):
            if not inputs & chain.names:
                continue
            chain.cost += cost
            chain.names.update(outputs)
            if chain.cost >= MIN_COST_S:
                del self._chains[path]
                self._warn(path, chain)

    def _warn(self, path: str, chain: _Chain) -> None:
        self._warned.add(path)
        what = (
            f"{_shown(path)} is written by {quote(chain.writer)} and read back by {quote(chain.reader)} "
            f"in this session: each write gives the file new bytes, so the read and what is computed "
            f"from it run again ({chain.cost:.1f}s this time) and cannot be reused."
        )
        fix = (
            "keep the data in a variable (a DataFrame) and pass that on to the next step; "
            "write the file once, as the last step."
        )
        log_diagnostic(logger, CODE, what, fix)
        warn_diagnostic(CashCacheIneffectiveWarning, CODE, what, fix)
