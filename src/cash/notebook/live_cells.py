"""Receive live cell sources pushed by cash's JupyterLab extension.

cash reads the cells it did not execute from the saved `.ipynb`, so an unsaved
edit is invisible to it. Colab reads live cells natively and VS Code's hot-exit
backup supplies them there; on JupyterLab the only route is a frontend
component, because the kernel has no other copy of the document.

It PUSHES rather than answering a request: a comm sent while a cell is executing
is queued until that cell ends (pinned by
tests/test_notebook_integration/test_comm_reply_during_execution.py), so a
request/response design could never serve a read at the moment cash needs one.
The extension pushes on change and flushes before execution; shell messages are
FIFO, so a push sent before an execute_request is processed first.

Everything here tolerates nonsense: the payload crosses a process boundary from
JavaScript, and a malformed one must cost the fallback, never an exception.
"""
from __future__ import annotations

TARGET = "cash_live_cells"

_store: dict = {"seq": -1, "cells": None}


def handle_message(data) -> None:
    """Accept one pushed snapshot. Ignores anything that is not one."""
    if not isinstance(data, dict):
        return
    seq = data.get("seq")
    cells = data.get("cells")
    # bool is an int in Python, and a bool sequence number is nonsense that
    # would compare as 0/1 and wedge the store.
    if not isinstance(seq, int) or isinstance(seq, bool):
        return
    if not isinstance(cells, list) or not all(isinstance(c, dict) for c in cells):
        return
    # Never move backwards: a retry or duplicate must not replace a newer
    # snapshot with an older one.
    if seq <= _store["seq"]:
        return
    _store["seq"] = seq
    _store["cells"] = cells


def latest_cells() -> list[dict] | None:
    """The most recent pushed snapshot, or None if the extension never spoke."""
    return _store["cells"]


def reset() -> None:
    _store["seq"] = -1
    _store["cells"] = None


# Thread-safety note. The extension forces its comm onto the MAIN shell
# (`commsOverSubshells = 'disabled'`), so `handle_message` runs on the kernel's
# own thread and the plain dict above needs no lock. If that is ever relaxed --
# JupyterLab 4.6 defaults to delivering comms on a subshell thread -- this store
# becomes cross-thread mutable state and the two-field update below stops being
# atomic. Do not remove that line in the extension without revisiting this.


def register_target(shell) -> bool:
    """Register the comm target. Returns False if that is not possible.

    False is the ordinary case for anyone without the extension, so it is a
    return value rather than an exception.
    """
    try:
        def _on_open(comm, msg):
            # A comm_open is a NEW frontend connection superseding any previous
            # one, so the high-water mark from the old one must not outlive it.
            #
            # The extension's `seq` is a closure variable in the plugin
            # activation, so it restarts at 0 on every browser page load, while
            # this store lives as long as the KERNEL. Without this reset, every
            # push after an F5 is dropped as "older" and `latest_cells()` keeps
            # serving the PRE-RELOAD snapshot -- source text that, since a
            # reload discards unsaved edits, no longer exists anywhere. That is
            # strictly worse than the saved-file fallback it displaces, which is
            # the one outcome this feature must never produce.
            #
            # Clearing is the safe direction: `latest_cells()` returning None
            # makes `_try_extension_cells` return None, and cash falls through
            # to the saved .ipynb rather than believing the notebook is empty.
            # Pinned by test_live_cells.py.
            reset()
            handle_message((msg or {}).get("content", {}).get("data"))

            @comm.on_msg
            def _on_msg(m):
                handle_message((m or {}).get("content", {}).get("data"))

        shell.kernel.comm_manager.register_target(TARGET, _on_open)
        return True
    except Exception:  # noqa: BLE001 - no kernel, old ipykernel, anything
        return False
