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
dir>/analysis``), and only once that dir exists. ``CASH_ANALYSIS_STORE=0``
turns it off, ``CASH_ANALYSIS_STORE=<dir>`` moves it. With
``CASH_ANALYSIS_MEMO_VERIFY=1`` every stored answer used is worked out again
and `StoredScanMismatch` raised when it differs.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

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
    "status",
    "store_dir",
    "write_json",
]

#: Layout of a stored file; a change makes every older one a miss.
_FORMAT = 1

STATS = {"loaded": 0, "stored": 0, "missed": 0, "verified": 0}


class StoredScanMismatch(AssertionError):
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
    entries = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for name in filenames:
            if name.endswith(".py"):
                path = os.path.join(dirpath, name)
                st = os.stat(path)
                entries.append(f"{os.path.relpath(path, root)}\0{st.st_size}\0{st.st_mtime_ns}")
    for entry in sorted(entries):
        h.update(entry.encode())
        h.update(b"\n")
    _FINGERPRINT.append(h.hexdigest())
    return _FINGERPRINT[0]


def _dir() -> Path | None:
    """``<cache dir>/analysis/<fingerprint>``: beside the results, wherever
    the settings put them (a pool worker's are its parent's). None when that
    cache dir does not exist yet: the store never makes one of its own."""
    setting = _setting()
    if setting.lower() in ("0", "false", "no", "off", ""):
        return None
    if setting.lower() in ("1", "true", "yes", "on"):
        cache_dir = active_config().cache_dir
        if not cache_dir or not os.path.isdir(cache_dir):
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
    if directory is None:
        return compute(source) or frozenset()
    digest = hashlib.blake2b(source.encode("utf-8", "surrogatepass"), digest_size=20).hexdigest()
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


def enabled() -> bool:
    """Is the store on (``CASH_ANALYSIS_STORE`` not ``0``)? It may still
    have no folder yet (`store_dir`)."""
    return _setting().lower() not in ("0", "false", "no", "off", "")


def store_dir() -> Path | None:
    """Where stored analysis goes for this Python and this cash; None when
    the store is off or the cache dir does not exist yet."""
    return _store_dir()


def read_json(file: Path) -> object:
    """The JSON in *file*, or None when it cannot be read or parsed."""
    try:
        with untracked():
            return json.loads(file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
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
    directory = _store_dir()
    if directory is not None:
        _write(directory, _slot_file(directory, kind, path), {**data, "kind": kind, "digest": digest})


def _load(file: Path, kind: str, digest: str) -> dict | None:
    try:
        with untracked():
            data = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
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
                    json.dump(data, f)
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
