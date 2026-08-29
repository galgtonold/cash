"""``os.replace`` is atomic everywhere, but on Windows it is not always allowed.

If any handle currently has the destination open, Windows fails the replace
with ``ERROR_ACCESS_DENIED`` instead of waiting -- where POSIX simply swaps the
directory entry and lets the reader keep the old inode.

This backend expects concurrent readers by design (``_atomic_write``'s own
docstring says so) and performs writes on a background thread, so writer and
reader collide routinely rather than exceptionally. Measured before the fix: a
WinError 5 on ~every Windows CI job, and on 10 of 12 consecutive local runs of
``tests/test_core/test_file_dep_propagation.py``. Each one silently discarded
the entry and forced a recompute -- a cache that quietly stops caching, on the
platform where the failure is invisible because it degrades gracefully.

Instrumenting the failure showed the blocking handle is released almost at
once: the destination became writable on the very first 10 ms retry in every
observed case. So a short bounded retry is the whole fix.
"""
from __future__ import annotations

import os
import sys
import threading
import time

import pytest

from cash.backends.file_backend import FileBackend
from cash.backends.entry_format import ENTRY_SUFFIX, pack_entry, read_entry


def _backend(tmp_path):
    return FileBackend(cache_dir=str(tmp_path / "cache"))


def test_a_transient_permission_error_does_not_lose_the_write(tmp_path, monkeypatch):
    """The write must survive a destination that is briefly locked.

    Platform-agnostic: the real collision only happens on Windows, so the
    denial is injected rather than provoked, and the assertion is that the
    payload lands anyway.
    """
    backend = _backend(tmp_path)
    target = str(tmp_path / "cache" / f"entry{ENTRY_SUFFIX}")
    os.makedirs(os.path.dirname(target), exist_ok=True)

    real_replace = os.replace
    calls = {"n": 0}

    def flaky_replace(src, dst, *a, **kw):
        calls["n"] += 1
        if calls["n"] <= 3:                      # denied three times, then clear
            raise PermissionError(5, "Access is denied")
        return real_replace(src, dst, *a, **kw)

    monkeypatch.setattr(os, "replace", flaky_replace)
    backend._atomic_write(target, b"payload-that-must-survive")

    assert calls["n"] == 4, "expected three denials then a success"
    with open(target, "rb") as f:
        assert f.read() == b"payload-that-must-survive"


def test_a_permanent_permission_error_still_raises(tmp_path, monkeypatch):
    """Retrying must not turn a genuine, persistent denial into silence."""
    backend = _backend(tmp_path)
    target = str(tmp_path / "cache" / f"entry{ENTRY_SUFFIX}")
    os.makedirs(os.path.dirname(target), exist_ok=True)

    def always_denied(src, dst, *a, **kw):
        raise PermissionError(5, "Access is denied")

    monkeypatch.setattr(os, "replace", always_denied)
    with pytest.raises(PermissionError):
        backend._atomic_write(target, b"nope")

    leftovers = [p for p in os.listdir(os.path.dirname(target)) if ".part" in p]
    assert not leftovers, f"partial files left behind: {leftovers}"


@pytest.mark.skipif(sys.platform != "win32", reason="only Windows denies the replace")
def test_a_real_windows_reader_holding_the_destination_does_not_lose_the_write(tmp_path):
    """The actual production collision, provoked rather than injected."""
    backend = _backend(tmp_path)
    target = str(tmp_path / "cache" / f"entry{ENTRY_SUFFIX}")
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, "wb") as f:
        f.write(b"old")

    holder_open = threading.Event()

    def hold():
        with open(target, "rb"):
            holder_open.set()
            time.sleep(0.15)          # release well inside the retry budget

    t = threading.Thread(target=hold)
    t.start()
    holder_open.wait(timeout=5)

    backend._atomic_write(target, b"new")
    t.join()

    with open(target, "rb") as f:
        assert f.read() == b"new"
