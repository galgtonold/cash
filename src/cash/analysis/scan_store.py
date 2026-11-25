"""Module scans kept on disk, so a fresh process does not parse every module
its cached functions reach again.

The walk asks, for the module of every helper it audits, which of its
globals some function in it reassigns or mutates
(`mutable_globals.modified_globals_in_source`). That needs the whole module
parsed and walked, which for a large library is most of what a fresh
process's first calls cost on top of the user's own work. The answer is a
function of three things only: the module's source TEXT, the Python that
parses it, and cash's own code that scans it. So it is stored under a digest
of all three, and a stored answer is used only when all three match exactly:

* the text is digested in full on every lookup (never a file stat), so an
  edit, a revert, a moved or copied file all compare by what is in them;
* the Python is ``sys.version`` and ``sys.implementation.cache_tag``;
* cash's code is its version and the size and mtime of every ``.py`` file
  in the package (`_cash_fingerprint`), so any edit to cash reads as a
  different analyser.

One file per module text digest would grow with every edit, so the store is
one file per (fingerprint, file path) holding the digest it answers for;
an edit overwrites it.

It lives in the active cache dir, beside the results (``<cache
dir>/analysis``), and only once that dir exists: what a process works out
before then (its first call, a miss, runs before the result that makes the
dir is stored) is held (`_PENDING`) and written once the dir is there, at
the latest when the process exits, so the second process does not pay for
it again. Each record held is written into the folder of the call that
worked it out (`target_dir`), not the one the settings name when it is
written: at exit no call runs (`written_to`). ``CASH_ANALYSIS_STORE=0``
turns it off, ``CASH_ANALYSIS_STORE=<dir>`` moves it. With
``CASH_ANALYSIS_MEMO_VERIFY=1`` every stored answer used is worked out again
and `StoredScanMismatch` raised when it differs.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

from .. import _verify
from .._active import active_config
from ..backends.cache_dir import ANALYSIS_DIRNAME
from ..tracking import io_watch
from ..tracking.tracker_context import untracked
from . import tree_memo

__all__ = [
    "StoredScanMismatch",
    "enabled",
    "load_record",
    "read_json",
    "save_record",
    "scan",
    "ready",
    "status",
    "store_dir",
    "target_dir",
    "write_json",
    "written_to",
]

#: Layout of a stored file; a change makes every older one a miss.
_FORMAT = 1

STATS = {"loaded": 0, "stored": 0, "missed": 0, "verified": 0}


class StoredScanMismatch(_verify.Mismatch):
    """``CASH_ANALYSIS_MEMO_VERIFY=1``: a stored module scan is not what
    scanning the module again gives."""


def status() -> dict:
    return {"dir": str(_dir() or ""), **STATS}


def _setting() -> str:
    return os.environ.get("CASH_ANALYSIS_STORE", "1").strip()


_FINGERPRINT: list[str] = []


def _cash_fingerprint() -> str:
    """cash's version and the size and mtime of each of its ``.py`` files,
    with the Python that runs it: what a stored scan was made by."""
    if _FINGERPRINT:
        return _FINGERPRINT[0]
    import cash

    root = Path(cash.__file__).parent
    h = hashlib.blake2b(digest_size=16)
    h.update(f"{_FORMAT}\0{getattr(cash, '__version__', '')}\0{sys.version}\0{sys.implementation.cache_tag}".encode())
    entries: list[str] = []
    _py_file_prints(str(root), "", entries)
    for entry in sorted(entries):
        h.update(entry.encode())
        h.update(b"\n")
    _FINGERPRINT.append(h.hexdigest())
    return _FINGERPRINT[0]


def _py_file_prints(folder: str, prefix: str, into: list[str]) -> None:
    """``relative path, size, mtime`` of each ``.py`` file under *folder*,
    ``__pycache__`` left out, as ``os.walk`` and an ``os.stat`` of each found
    them: one listing a folder, each file's stat taken from it
    (``DirEntry.stat``, which Windows answers from the listing: 418 files
    stat'd one by one were 17 ms of a process's first cached call there). A
    linked folder is not entered, a folder that cannot be listed is passed
    over, and a ``.py`` that cannot be stat'd raises, as before."""
    try:
        listing = os.scandir(folder)
    except OSError:
        return
    subfolders = []
    with listing:
        for entry in listing:
            try:
                is_dir = entry.is_dir()
            except OSError:
                is_dir = False
            rel = entry.name if not prefix else os.path.join(prefix, entry.name)
            if is_dir:
                if entry.name != "__pycache__" and not entry.is_symlink():
                    subfolders.append((entry.path, rel))
            elif entry.name.endswith(".py"):
                st = entry.stat()
                into.append(f"{rel}\0{st.st_size}\0{st.st_mtime_ns}")
    for path, rel in subfolders:
        _py_file_prints(path, rel, into)


def _dir(*, made: bool = True) -> Path | None:
    """``<cache dir>/analysis/<fingerprint>``: beside the results, wherever
    the settings of the code running now put them (a pool worker's are its
    parent's). None when that cache dir does not exist yet (*made*): the
    store never makes one of its own."""
    setting = _setting()
    if setting.lower() in ("0", "false", "no", "off", ""):
        return None
    if setting.lower() in ("1", "true", "yes", "on"):
        cache_dir = active_config().cache_dir
        if not cache_dir or (made and not os.path.isdir(cache_dir)):
            return None
        base = Path(os.path.abspath(cache_dir)) / ANALYSIS_DIRNAME
    else:
        base = Path(setting)
    return base / _cash_fingerprint()


def scan(kind: str, path: str, source: str, compute: Callable[[str], frozenset[str] | None]) -> frozenset[str]:
    """``compute(source)``, from disk when an earlier process stored it for
    this exact *source* text, under this Python and this cash. *kind* names
    the scan, *path* the file the text was read from (it only names the
    stored file; the text decides). *compute* returns None when it gave up
    for a reason outside the text (a recursion limit): the answer is then
    empty and not stored."""
    directory = _store_dir()
    digest = hashlib.blake2b(source.encode("utf-8", "surrogatepass"), digest_size=20).hexdigest()
    if directory is None:
        answer = compute(source)
        if answer is None:
            return frozenset()
        _save(kind, path, {"kind": kind, "digest": digest, "names": sorted(answer)})
        return answer
    file = _slot_file(directory, kind, path)
    stored = _read(file, kind, digest)
    if stored is not None:
        STATS["loaded"] += 1
        if tree_memo.VERIFY:
            STATS["verified"] += 1
            fresh = compute(source)
            if fresh is not None and fresh != stored:
                raise StoredScanMismatch(f"stored {kind} scan of {path} differs from scanning it again")
        return stored
    STATS["missed"] += 1
    answer = compute(source)
    if answer is None:
        return frozenset()
    _write(directory, file, {"kind": kind, "digest": digest, "names": sorted(answer)})
    return answer


def _slot_file(directory: Path, kind: str, path: str) -> Path:
    slot = hashlib.blake2b(f"{kind}\0{path}".encode("utf-8", "surrogatepass"), digest_size=12).hexdigest()
    return directory / f"{slot}.json"


def _store_dir() -> Path | None:
    try:
        with untracked():
            return _dir()
    except OSError:
        return None


def target_dir() -> Path | None:
    """Where what the code running now works out goes: `store_dir`, also
    while its cache dir does not exist yet (`ready`); None when the store is
    off.

    A record held to be written later (in a batch, or at exit) keeps this,
    taken while it is worked out: later, and at exit above all, no call is
    running, so the settings are the default instance's. Looked up then, the
    records the calls of a ``Cash(cache_dir=...)`` of the program's own
    worked out went into the default cache dir whenever that existed (a
    ``.cash`` in the project, or above it), every later process looked for
    them in its own folder, and walked the same helpers again."""
    try:
        with untracked():
            return _dir(made=False)
    except OSError:
        return None


def ready(directory: Path) -> bool:
    """May *directory*, a `target_dir`, be written now: is the store on, and
    the cache dir it sits in there? The store never makes one of its own."""
    setting = _setting().lower()
    if setting in ("0", "false", "no", "off", ""):
        return False
    if setting in ("1", "true", "yes", "on"):
        try:
            with untracked():
                return os.path.isdir(directory.parent.parent)
        except OSError:
            return False
    return True


def written_to(directory: Path, *, at_exit: bool = False) -> Path | None:
    """The folder a record held for *directory* (a `target_dir`) is written
    into now: *directory* once it is `ready`; None while it is not, and the
    record stays held.

    At exit (*at_exit*), the last chance, a record whose cache dir never
    came to be or was removed goes into the store folder there is then
    (`store_dir`) instead of nowhere: what it holds is a function of source texts, this
    Python and this cash (the folder's name), never of the cache dir it
    sits in."""
    if ready(directory):
        return directory
    return _store_dir() if at_exit else None


def enabled() -> bool:
    """Is the store on (``CASH_ANALYSIS_STORE`` not ``0``)? It may still
    have no folder yet (`store_dir`)."""
    return _setting().lower() not in ("0", "false", "no", "off", "")


def store_dir() -> Path | None:
    """Where stored analysis goes for this Python and this cash; None when
    the store is off or the cache dir does not exist yet."""
    return _store_dir()


#: A stored file that cannot be read back: gone, unreadable, not JSON, or
#: nested deeper than the parser recurses (or too big to hold). Each is a
#: miss, which the next store overwrites.
_UNREADABLE = (OSError, ValueError, RecursionError, MemoryError)


def read_json(file: Path) -> object:
    """The JSON in *file*, or None when it cannot be read or parsed."""
    try:
        with untracked():
            return json.loads(file.read_text(encoding="utf-8"))
    except _UNREADABLE:
        return None


def write_json(directory: Path, file: Path, data: dict) -> bool:
    """Write *data* to *file* in *directory*: best effort and atomic. Did it?"""
    return _write(directory, file, data, counted=False)


def load_record(kind: str, path: str, digest: str) -> dict | None:
    """The record an earlier process stored (`save_record`) for the file at
    *path* when its text digested to *digest*, under this Python and this
    cash; None when there is none, or the store is off."""
    directory = _store_dir()
    if directory is None:
        return None
    return _load(_slot_file(directory, kind, path), kind, digest)


def save_record(kind: str, path: str, digest: str, data: dict) -> None:
    """Store *data* (JSON) as what the text of *path* digested to *digest*
    gives; it replaces the record of any other text of that file. Best effort."""
    _save(kind, path, {**data, "kind": kind, "digest": digest})


#: ``(store folder, kind, path)`` -> the record worked out while that
#: folder's cache dir did not exist yet, written by `flush`.
_PENDING: dict[tuple[Path, str, str], dict] = {}
#: More records than any one process should hold for a cache dir to come.
_MAX_PENDING = 4096


def _save(kind: str, path: str, record: dict) -> None:
    directory = target_dir()
    if directory is None:
        return
    if not ready(directory):
        slot = (directory, kind, path)
        if len(_PENDING) < _MAX_PENDING or slot in _PENDING:
            _PENDING[slot] = record
        return
    if _PENDING:
        flush()
    _write(directory, _slot_file(directory, kind, path), record)


def flush(*, at_exit: bool = False) -> None:
    """Write each record held into the folder it was worked out for, once
    that folder's cache dir exists; the others stay held (`written_to`)."""
    for slot in list(_PENDING):
        directory, kind, path = slot
        try:
            where = written_to(directory, at_exit=at_exit)
        except Exception:  # noqa: BLE001 - at exit: a store that cannot be found keeps nothing
            return
        if where is None:
            continue
        record = _PENDING.pop(slot, None)
        if record is not None:
            _write(where, _slot_file(where, kind, path), record)


atexit.register(flush, at_exit=True)


def _load(file: Path, kind: str, digest: str) -> dict | None:
    try:
        with untracked():
            data = json.loads(file.read_text(encoding="utf-8"))
    except _UNREADABLE:
        return None
    if not isinstance(data, dict) or data.get("kind") != kind or data.get("digest") != digest:
        return None
    return data


def _read(file: Path, kind: str, digest: str) -> frozenset[str] | None:
    data = _load(file, kind, digest)
    if data is None:
        return None
    names = data.get("names")
    if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
        return None
    return frozenset(names)


def _write(directory: Path, file: Path, data: dict, counted: bool = True) -> bool:
    """Best effort, atomic: a reader sees the old file or the new one.

    Cash's own file, as the library map's (`library_versions`): no write of
    the code running. Stored while a cached call inside another runs, it was
    the outer call's write (`write_observer`), which kept it from the cache."""
    try:
        with untracked(), io_watch.own_work():
            directory.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp-", suffix=".json")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    # One string, one write: ``json.dump`` writes a chunk
                    # per token, several times slower for a large record.
                    f.write(json.dumps(data))
                os.replace(tmp, file)
            except BaseException:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
    except OSError:
        return False
    if counted:
        STATS["stored"] += 1
    return True
