"""Superseded versions of one statement, and how many of them are worth keeping.

A statement re-run on changed inputs stores a new entry under a new key, and
the old one stays: nothing but the byte cap would remove it, and the cap is a
quarter of the free disk. A cleaning chain of a few ~500 MB frames, re-run a
handful of times, grows to 10 GB for a 200 MB input folder.

So a statement's versions are pruned as they are written, before the cache is
full, by what they are worth. The newest superseded version always stays: an
undo is the common way back, and the eviction simulation lost 9.3% of the
achievable savings without it against 1.6% with it
(``benchmarks/eviction_sim``). Older ones stay, newest first, while all the
superseded versions kept fit a byte budget of ``BYTES_PER_COMPUTE_SECOND``
times what that version took to compute. Minutes of work condensed into a few
bytes keep up to ``MAX_SUPERSEDED`` versions; a big frame that is cheap to
rebuild keeps one. A version this process has read is never pruned: it is in
use, whatever its slot says.

A *slot* is what makes two entries versions of each other. The statement
layer names it (``version_slot`` in the metadata); an entry without one --
decorator calls, call units, raw backend use -- is never pruned here, only by
the cap.

Like the rank index, the record is ADVISORY: a line per write, access and
removal, appended by any process and read once per process. Lost or stale, it
costs pruning precision, never a value.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "Version",
    "VersionIndex",
    "superseded_to_drop",
    "INDEX_FILENAME",
    "BYTES_PER_COMPUTE_SECOND",
    "MAX_SUPERSEDED",
]

#: No entry suffix, so every entry glob passes it by.
INDEX_FILENAME = "_versions.log"

#: Superseded bytes one second of compute pays for. A 1.4 s build of a 700 MB
#: frame affords 90 MB -- the newest superseded version, which always stays,
#: and nothing more. A one-minute fit affords 3.8 GB.
#:
#: Defined in `value_policy`, which applies the same rate to every entry as it
#: is written. One number: a value the writer would refuse is not one this
#: would then keep a spare copy of.
from .value_policy import BYTES_PER_COMPUTE_SECOND  # noqa: E402

#: However cheap they are to hold, no more superseded versions than this.
MAX_SUPERSEDED = 16

_REMOVED = "-"


@dataclass
class Version:
    key: str
    size: int
    cost: float
    used: float


def superseded_to_drop(
    versions: dict[str, Version], current: str, in_use: set[str] | frozenset[str] = frozenset()
) -> list[str]:
    """Keys of *versions* other than *current* that are not worth keeping."""
    older = sorted((v for k, v in versions.items() if k != current), key=lambda v: v.used, reverse=True)
    drop: list[str] = []
    kept = kept_bytes = 0
    for i, v in enumerate(older):
        affordable = kept < MAX_SUPERSEDED and kept_bytes + v.size <= BYTES_PER_COMPUTE_SECOND * max(v.cost, 0.0)
        if i == 0 or v.key in in_use or affordable:
            kept += 1
            kept_bytes += v.size
        else:
            drop.append(v.key)
    return drop


class VersionIndex:
    """Append-only record of each slot's versions for one cache directory.

    Format, one record per line::

        <slot> <key> <size> <cost> <used>
        - <key>
    """

    def __init__(self, cache_dir: str, untracked: Any) -> None:
        self.path = os.path.join(cache_dir, INDEX_FILENAME)
        self._untracked = untracked
        self._lock = threading.Lock()
        self._slots: dict[str, dict[str, Version]] | None = None
        self._slot_of: dict[str, str] = {}
        self._lines = 0

    def _load(self) -> None:
        slots: dict[str, dict[str, Version]] = {}
        slot_of: dict[str, str] = {}
        lines = 0
        try:
            with self._untracked(), open(self.path, encoding="ascii", errors="replace") as fh:
                for line in fh:
                    lines += 1
                    parts = line.split()
                    if len(parts) == 2 and parts[0] == _REMOVED:
                        slot = slot_of.pop(parts[1], None)
                        if slot is not None:
                            slots.get(slot, {}).pop(parts[1], None)
                        continue
                    if len(parts) != 5:
                        continue
                    try:
                        v = Version(parts[1], int(parts[2]), float(parts[3]), float(parts[4]))
                    except ValueError:
                        continue
                    old = slot_of.get(v.key)
                    if old is not None and old != parts[0]:
                        slots.get(old, {}).pop(v.key, None)
                    slots.setdefault(parts[0], {})[v.key] = v
                    slot_of[v.key] = parts[0]
        except FileNotFoundError:
            pass
        except OSError:
            logger.debug("Could not read version index %s", self.path, exc_info=True)
        self._slots, self._slot_of, self._lines = slots, slot_of, lines

    def _append(self, lines: list[str]) -> None:
        if not lines:
            return
        try:
            with self._untracked(), open(self.path, "a", encoding="ascii") as fh:
                fh.write("".join(lines))
            self._lines += len(lines)
        except OSError:
            logger.debug("Could not append to version index %s", self.path, exc_info=True)

    @staticmethod
    def _valid(*tokens: str) -> bool:
        return all(t and t.isascii() and not any(c.isspace() for c in t) for t in tokens)

    def record(self, slot: str, key: str, size: int, cost: float, used: float) -> dict[str, Version]:
        """Note a write (or a use) of *key* in *slot*; returns the slot's versions."""
        if not self._valid(slot, key):
            return {}
        with self._lock:
            if self._slots is None:
                self._load()
            old = self._slot_of.get(key)
            if old is not None and old != slot:
                self._slots.get(old, {}).pop(key, None)
            versions = self._slots.setdefault(slot, {})
            versions[key] = Version(key, int(size), float(cost), float(used))
            self._slot_of[key] = slot
            self._append([f"{slot} {key} {int(size)} {float(cost)!r} {float(used)!r}\n"])
            return dict(versions)

    def touch(self, key: str, used: float) -> None:
        """A read of *key*: it becomes its slot's most recently used version."""
        with self._lock:
            if self._slots is None:
                self._load()
            slot = self._slot_of.get(key)
            v = self._slots.get(slot, {}).get(key) if slot is not None else None
            if v is None:
                return
            v.used = max(v.used, used)
            self._append([f"{slot} {key} {v.size} {v.cost!r} {v.used!r}\n"])

    def forget(self, keys: list[str]) -> None:
        with self._lock:
            if self._slots is None:
                self._load()
            lines = []
            for key in keys:
                slot = self._slot_of.pop(key, None)
                if slot is not None:
                    self._slots.get(slot, {}).pop(key, None)
                    if self._valid(key):
                        lines.append(f"{_REMOVED} {key}\n")
            self._append(lines)
            live = len(self._slot_of)
            if self._lines > 2 * live + 1000:
                self._compact()

    def _compact(self) -> None:
        """Rewrite as exactly what is known, through a temp file and a rename;
        a failed rename leaves the longer, still-correct file."""
        tmp = f"{self.path}.{os.getpid()}.tmp"
        body = "".join(
            f"{slot} {v.key} {v.size} {v.cost!r} {v.used!r}\n"
            for slot, versions in self._slots.items()
            for v in versions.values()
        )
        try:
            with self._untracked():
                with open(tmp, "w", encoding="ascii") as fh:
                    fh.write(body)
                os.replace(tmp, self.path)
            self._lines = body.count("\n")
        except OSError:
            logger.debug("Could not compact version index %s", self.path, exc_info=True)
            try:
                os.remove(tmp)
            except OSError:
                pass

    def remove(self) -> None:
        with self._lock:
            self._slots, self._slot_of, self._lines = None, {}, 0
            try:
                with self._untracked():
                    os.remove(self.path)
            except FileNotFoundError:
                pass
            except OSError:
                logger.debug("Could not remove version index %s", self.path, exc_info=True)
