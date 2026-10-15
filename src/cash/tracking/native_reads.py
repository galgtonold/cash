"""Files that C libraries open themselves: the native file hook.

Python's audit hook reports every open made through Python. A library that
opens its file in C -- libarrow, libsqlite3, a Rust crate, C++'s
``std::ifstream`` -- raises no event, which is why `reader_patches` wraps
those libraries' entry points one by one. This hook sees such opens
generically: the C extension (``cfilehook.c``) rebinds the open family
(``open``, ``openat``, ``fopen``, ``opendir`` and their variants on Linux;
``CreateFileW``/``A``/``2`` and the CRT's ``_open``, ``_wsopen_s``, ``fopen``,
``_wfopen`` and their variants on Windows) in the symbol tables (GOT) or
import address tables of the shared libraries already loaded, without
LD_PRELOAD, DLL injection or ptrace, and records the paths opened while a file
tracker is running. Each
tracker gets them when it closes, through the methods the audit events use:
a read is a dependency, a missing file an absent one, a file the library
created or truncated is the block's own output.

**On by default** where it is supported: Linux with glibc on x86-64 and
aarch64, and Windows on x64; elsewhere `status_line` says "off: not supported
here". ``CASH_NATIVE_FILE_HOOK=0`` keeps it off at import; `enable` /
`disable` switch it at run time (`disable` puts every rebound slot back).

What it does not see (``cfilehook.c`` has the details): direct system calls
(io_uring, Go runtimes, ``NtCreateFile``), libraries linked statically against
the C library (``/MT`` on Windows), opens inside the C library itself,
``stat``/``access`` probes, directory listings on Windows, calls through a
function pointer the library looked up itself (``dlsym``, ``GetProcAddress``,
ctypes), and a library loaded and used inside a single tracked block (it is
hooked from the next block on). A
native open made by ANY thread while a tracker is open is credited to every
tracker open at the time: a library's I/O threads carry no Python context,
and an extra dependency can only cost a recompute, never a stale result.
The file is hashed when the tracker closes, not at the moment it was opened.
"""

from __future__ import annotations

import os
import threading
from typing import Any

from cash import _native

__all__ = ["ENV", "active", "close_window", "disable", "enable", "open_window", "stats", "status_line"]

#: The environment switch read at import.
ENV = "CASH_NATIVE_FILE_HOOK"

_cw = _native.cwatch
_fh = _cw if _cw is not None and hasattr(_cw, "filehook_supported") else None

#: Stands in, as a dependency that is never fresh, for the reads a block lost
#: when the hook's buffer was full.
LOST_READS = "<native file reads lost: buffer full>"

_lock = threading.Lock()
_on = False


def _switched_off(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("0", "false", "no", "off")


def supported() -> tuple[bool, str]:
    """``(True, how)`` where the hook works here, else ``(False, why not)``."""
    if _fh is None:
        return False, f"needs the C extension ({_native.REASON or 'too old a build'})"
    ok, why = _fh.filehook_supported()
    return bool(ok), str(why)


def active() -> bool:
    """Is the hook on?"""
    return _on


def enable() -> bool:
    """Turn the hook on; False (and nothing changes) where it is not supported."""
    global _on
    ok, _ = supported()
    if not ok:
        return False
    with _lock:
        if not _on:
            _fh.filehook_enable()
            _on = True
    return True


def disable() -> None:
    """Turn the hook off: every rebound slot gets its original back."""
    global _on
    with _lock:
        if _on:
            _on = False
            _fh.filehook_disable()


def open_window() -> int | None:
    """Start recording for a tracker; the window to close, or None when off.

    Also hooks the libraries imported since the last window.
    """
    if not _on:
        return None
    return _fh.filehook_window_open()


def close_window(window: int | None, tracker: Any) -> None:
    """Stop the recording *window* and give *tracker* what was opened in it."""
    if window is None:
        return
    try:
        events, lost = _fh.filehook_window_close(window)
    except Exception:  # noqa: BLE001 - never fail the user's call over this
        tracker.add_tracked_unresolved(LOST_READS)
        return
    for kind, raw in events:
        path = os.fsdecode(raw)
        try:
            if kind == _READ:
                tracker.track_path(path)
            elif kind == _ABSENT:
                tracker.track_absent(path)
            elif kind == _CREATED:
                tracker.note_entry_changed(path)
                tracker.note_created(path)
            elif kind == _WRITE:
                tracker.note_entry_changed(path)
            elif kind == _UNRESOLVED:
                tracker.add_tracked_unresolved(path)
        except Exception:  # noqa: BLE001 - a read that cannot be recorded cannot be vouched for
            tracker.add_tracked_unresolved(path)
    if lost:
        tracker.add_tracked_unresolved(LOST_READS)


def stats() -> dict[str, Any]:
    """The hook's counters, and the libraries it rebound slots in."""
    if _fh is None or not hasattr(_fh, "filehook_stats"):
        return {"enabled": False}
    return dict(_fh.filehook_stats())


def status_line() -> str:
    """One line for ``%cash_status``: on, off, or off and why."""
    ok, why = supported()
    if not ok:
        return f"off: {why}"
    if not _on:
        return f"off ({ENV}=0 or disable(); unset it to turn the hook on)"
    s = stats()
    libs = len(s.get("hooked_objects", ()))
    return f"on ({s.get('slots', 0)} slots in {libs} librar{'y' if libs == 1 else 'ies'})"


if _fh is not None and hasattr(_fh, "FH_READ"):
    _READ, _ABSENT, _CREATED, _WRITE, _UNRESOLVED = (
        _fh.FH_READ,
        _fh.FH_ABSENT,
        _fh.FH_CREATED,
        _fh.FH_WRITE,
        _fh.FH_UNRESOLVED,
    )
else:
    _READ, _ABSENT, _CREATED, _WRITE, _UNRESOLVED = 1, 2, 3, 4, 5

if not _switched_off(ENV):
    enable()
