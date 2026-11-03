"""Removing a folder's watch on Windows never waits for good.

On the PC a kernel watching a folder froze for good once the folder was
deleted and made again: the folder's read had ended and was not issued
again, and removing the watch then waited, without end, for a cancelled
read that was never pending. These tests stand in for kernel32 so they run
on every OS.
"""

from __future__ import annotations

import ctypes
import types

from cash.tracking import file_watch

_INCOMPLETE = file_watch._ERROR_IO_INCOMPLETE


class _Kernel:
    """kernel32 as the watch calls it: `result` answers each GetOverlappedResult
    (True, or the last error), and a wait never ends, like the real one did."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.error = 0
        self.closed = []
        self.cancelled = []
        self.waited = False

    def result(self, handle, overlapped, moved, wait):
        if wait:
            self.waited = True
            raise AssertionError("waited on a read without end")
        answer = self.answers.pop(0) if self.answers else _INCOMPLETE
        if answer is True:
            return True
        self.error = answer
        return False

    def issue(self, handle, overlapped, buffer):
        return False  # the folder is gone: the read is not issued again


def _watch(kernel):
    w = object.__new__(file_watch._DirectoryChanges)
    w._ctypes = types.SimpleNamespace(
        byref=lambda x: x,
        get_last_error=lambda: kernel.error,
        memset=ctypes.memset,
        sizeof=ctypes.sizeof,
    )
    w._wintypes = types.SimpleNamespace(DWORD=ctypes.c_uint32)
    w._result = kernel.result
    w._cancel = lambda handle, overlapped: kernel.cancelled.append(handle) or False
    w._close = kernel.closed.append
    w._issue = kernel.issue
    w._open = {1: ("h1", object(), object())}
    w._idle = set()
    w._kept = []
    w._next = 1
    return w


def test_a_folder_whose_read_ended_is_removed_without_waiting():
    kernel = _Kernel([True])  # the read ended (the folder was deleted)
    w = _watch(kernel)
    assert w.poll() == [(1, True)]
    w.remove(1)
    assert kernel.closed == ["h1"]
    assert kernel.cancelled == []
    assert not kernel.waited


def test_a_cancelled_read_that_never_ends_is_given_up_on_and_kept(monkeypatch):
    monkeypatch.setattr(file_watch, "_CANCEL_WAIT_SECONDS", 0.05)
    kernel = _Kernel([])  # always still incomplete
    w = _watch(kernel)
    w.remove(1)
    assert kernel.closed == ["h1"]
    assert not kernel.waited
    assert len(w._kept) == 1  # its buffer is never freed under the system


def test_a_cancelled_read_that_ends_frees_its_buffer():
    kernel = _Kernel([_INCOMPLETE, 995])  # then ERROR_OPERATION_ABORTED
    w = _watch(kernel)
    w.remove(1)
    assert kernel.cancelled == ["h1"]
    assert kernel.closed == ["h1"]
    assert w._kept == []
