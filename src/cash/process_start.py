"""When this process started, by the wall clock: what tells a source file
edited after the import from one edited before it (`cash.loaded_code`).

Read from ``/proc`` on Linux and from ``GetProcessTimes`` on Windows, the
readings psutil makes, so a script does not import psutil for them; psutil
answers elsewhere, and cash's own import time when nothing does. Read once
per process.
"""

from __future__ import annotations

import os
import sys
import time

from ._lazy_module import LazyModule
from .source_reading import read_code_file

psutil = LazyModule("psutil")  # imported on first use: ~11 ms off `import cash`

_IMPORT_TIME = time.time()
_PROCESS_START: float | None = None


def _proc_start_time() -> float | None:
    """This process's start time from ``/proc`` (Linux), or None elsewhere.

    psutil's own formula -- boot time plus the start tick over the clock rate
    -- read directly, so a script does not import psutil (~11 ms) for it.
    """
    if not sys.platform.startswith("linux"):
        return None
    try:
        # Read untracked (`read_code_file`): cash's own read, not the user's.
        stat = read_code_file("/proc/self/stat")
        # Field 22, counted after the ")" that ends the command name (which
        # may itself hold spaces or parentheses).
        start_ticks = int(stat[stat.rindex(b")") + 2 :].split()[19])
        boot = next(
            int(line.split()[1]) for line in read_code_file("/proc/stat").splitlines() if line.startswith(b"btime ")
        )
        return boot + start_ticks / os.sysconf("SC_CLK_TCK")
    except Exception:  # noqa: BLE001 - not Linux, or no /proc: psutil answers
        return None


def _windows_start_time() -> float | None:
    """This process's start time from ``GetProcessTimes`` (Windows), the call
    psutil makes, so a script does not import psutil for it. None elsewhere."""
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        created, exited, kernel, user = (wintypes.FILETIME() for _ in range(4))
        # A private handle on kernel32, so the signatures set here leave the
        # shared ``ctypes.windll.kernel32`` as other code expects it. Without
        # argtypes the pseudo-handle (-1 as a HANDLE) overflowed a C int.
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
        kernel32.GetProcessTimes.restype = wintypes.BOOL
        if not kernel32.GetProcessTimes(
            kernel32.GetCurrentProcess(),
            ctypes.byref(created),
            ctypes.byref(exited),
            ctypes.byref(kernel),
            ctypes.byref(user),
        ):
            return None
        # 100 ns ticks since 1601-01-01, the FILETIME epoch.
        ticks = (created.dwHighDateTime << 32) | created.dwLowDateTime
        return ticks / 1e7 - 11644473600.0
    except Exception:  # noqa: BLE001 - no kernel32: psutil answers
        return None


def process_start_time() -> float:
    """Wall-clock time this process started, best effort, cached."""
    global _PROCESS_START
    if _PROCESS_START is not None:
        return _PROCESS_START

    started = _proc_start_time()
    if started is None:
        started = _windows_start_time()
    if started is None:
        try:
            started = float(psutil.Process().create_time())
        except Exception:  # noqa: BLE001 - no start time just falls back to cash's import time
            started = None
    if started is None:
        # cash's own import time. Misses only a file edited in the gap between
        # this process starting and cash being imported -- normally the first
        # lines of the program.
        started = _IMPORT_TIME
    _PROCESS_START = started
    return started
