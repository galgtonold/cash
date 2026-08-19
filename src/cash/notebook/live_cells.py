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

A snapshot is valid for exactly one NON-SILENT execution -- the one it
preceded. See ``expire``, which is the whole reason this module is more than
a variable, including the ``silent=True`` gap in that guarantee.

Everything here tolerates nonsense: the payload crosses a process boundary from
JavaScript, and a malformed one must cost the fallback, never an exception.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

TARGET = "cash_live_cells"

#: ``cells`` is the servable snapshot and the ONLY gate on serving one: it is
#: ``None`` whenever no snapshot may be served, so ``latest_cells`` cannot
#: disagree with some other field. ``seq`` is a high-water mark that OUTLIVES
#: the snapshot -- see ``expire`` for why it must.
_store: dict = {"seq": -1, "cells": None}


def _cell_shape_ok(cell: dict) -> bool:
    """A cell dict is safe to store only if its ``source``, when present, is a
    plain string.

    ``_extract_cell_entry`` (``server_discovery.py``) only special-cases a
    ``list`` source (joining it); anything else -- an int, ``None``, a dict, a
    bool -- passes through UNCHANGED as the "cell source". That value then
    reaches ``ast.parse``, which raises ``TypeError`` on a non-str argument --
    not the ``(SyntaxError, ValueError)`` the upstream checker catches, so it
    would escape all the way out to the user's own cell execution.
    """
    source = cell.get("source", "")
    return isinstance(source, str)


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
    if not isinstance(cells, list) or not all(
        isinstance(c, dict) and _cell_shape_ok(c) for c in cells
    ):
        return
    # Never move backwards: a retry or duplicate must not replace a newer
    # snapshot with an older one.
    if seq <= _store["seq"]:
        return
    _store["seq"] = seq
    _store["cells"] = cells


def latest_cells() -> list[dict] | None:
    """The snapshot pushed for the CURRENT execution, or None.

    None means "no live source", and the caller falls back to the saved
    ``.ipynb`` -- both when the extension never spoke at all, and when it spoke
    for an earlier execution but not for this one.
    """
    return _store["cells"]


def expire() -> None:
    """Retire the snapshot at the end of the execution it arrived for.

    THE safety property of this feature. This store lives as long as the
    KERNEL, while the frontend that fills it does not: a page reload onto the
    same kernel, ``jupyter labextension disable`` plus a reload, a second client
    attaching without the extension, or the extension simply erroring after
    having worked once, all stop the pushes without closing anything. Without
    expiry the last snapshot is then served for the rest of the kernel's life --
    cash checks the user's edited upstream cell against FROZEN text and finds no
    change, and because ``"extension"`` counts as a live source in
    ``staleness._LIVE_SOURCES`` the once-per-session "cash cannot see unsaved
    edits here" notice is suppressed as well. Confidently wrong, and silent
    about it: strictly worse than the staleness this feature exists to fix.

    So a snapshot is valid for exactly the ONE non-silent execution it
    preceded (see the ``silent=True`` gap below). The extension flushes before
    every ``execute_request`` (``labextension/src/index.ts``), so requiring a
    fresh push per execution costs a working extension nothing, and the
    moment one stops pushing cash falls back to the saved file AND the
    reactive notice fires -- both correct.

    Called from IPython's ``post_run_cell``, the last event of a non-silent
    execution. Measured on a real kernel (ipykernel 7.2, IPython 9.13): a push
    sent immediately before an ``execute_request`` is processed BEFORE that
    execution begins, and every one of cash's cell reads happens before
    ``post_run_cell`` fires -- so one snapshot survives all of them, however
    many times cash reads within a single cell. ``pre_run_cell`` would be the
    wrong hook: the push has already landed by the time it fires, so it would
    throw away the snapshot that just arrived for the execution now starting.

    The ``silent=True`` gap: both places that fire ``post_run_cell`` --
    ``IPython.core.interactiveshell.InteractiveShell.run_cell`` and
    ``ipykernel.ipkernel.IPythonKernel.do_execute`` -- guard the trigger behind
    ``if not silent``, so a silent execution never calls this function and any
    snapshot pushed for it survives into whatever runs next. That is a real
    hole in "exactly one execution," not a hedge -- but it costs nothing in
    practice: JupyterLab never sends a silent ``execute_request`` for a user's
    own cell, so nothing reachable through the UI this feature targets ever
    exercises it.

    ``seq`` deliberately survives. It is a duplicate-suppression high-water
    mark, not part of the snapshot: dropping it here would let a retransmitted
    copy of the snapshot just retired re-arm itself for the NEXT execution,
    which is the exact staleness being closed. Only ``reset`` -- a new frontend
    connection, or a new notebook -- may move it backwards.
    """
    _store["cells"] = None


def reset() -> None:
    _store["seq"] = -1
    _store["cells"] = None


# Thread-safety note. The extension forces its comm onto the MAIN shell
# (`commsOverSubshells = 'disabled'`), so `handle_message` runs on the kernel's
# own thread and the plain dict above needs no lock. If that is ever relaxed --
# JupyterLab 4.6 defaults to delivering comms on a subshell thread -- this store
# becomes cross-thread mutable state and the two-field update below stops being
# atomic. Do not remove that line in the extension without revisiting this.


def _on_post_run_cell(result=None) -> None:
    """IPython event adapter for ``expire``.

    A module-level function rather than a closure or a bound method, so its
    identity is stable across ``CashMagics`` re-instantiations
    (``cash.reset_session()``, a second ``Cash()``, ``%load_ext`` after a reset)
    and ``install_expiry_hook`` can unregister the previous registration by
    passing this same object.
    """
    expire()


def install_expiry_hook(shell) -> bool:
    """Register ``expire`` on ``post_run_cell``. False if that is not possible.

    False is the ordinary case outside a real IPython shell (MockShell, an
    embedded interpreter), so it is a return value rather than an exception --
    and it is harmless there, because nothing pushes a snapshot in the first
    place.
    """
    try:
        shell.events.unregister("post_run_cell", _on_post_run_cell)
    except (ValueError, KeyError, AttributeError, TypeError):
        pass  # not registered yet, which is the common case
    try:
        shell.events.register("post_run_cell", _on_post_run_cell)
        return True
    except (AttributeError, KeyError, TypeError) as e:
        logger.warning(
            "Could not register the live-cell expiry hook on post_run_cell: "
            "%s. A snapshot pushed by cash's JupyterLab extension would then "
            "be served for the rest of the kernel's life instead of expiring "
            "with the execution it arrived for, and cash's own \"cannot see "
            "unsaved edits\" notice would stay suppressed along with it.", e,
        )
        return False


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
