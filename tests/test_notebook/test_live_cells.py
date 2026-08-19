"""The kernel side of the JupyterLab live-cell push.

A frontend extension pushes the notebook's current cell sources over a comm
whenever they change, and flushes before a cell executes. cash reads the last
push instead of the saved .ipynb, so an unsaved edit is visible to the upstream
check.

This half needs no frontend: the comm handler is ordinary Python, and a test can
deliver exactly the payload the extension would.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

from cash.notebook import server_discovery as sd
from cash.notebook.live_cells import TARGET, handle_message, latest_cells, register_target, reset

CELLS = [
    {"cell_type": "code", "id": "a", "source": "THRESHOLD = 0.9"},
    {"cell_type": "code", "id": "b", "source": "y = THRESHOLD * 2"},
]


def setup_function():
    reset()


def test_nothing_pushed_means_none():
    """The control: with no extension present cash must fall through to the
    saved file, not serve an empty notebook."""
    assert latest_cells() is None


def test_a_push_becomes_readable():
    handle_message({"seq": 1, "cells": CELLS})
    got = latest_cells()
    assert got is not None
    assert got[0]["source"] == "THRESHOLD = 0.9"


def test_a_newer_push_replaces_an_older_one():
    handle_message({"seq": 1, "cells": CELLS})
    handle_message({"seq": 2, "cells": [{"cell_type": "code", "id": "a", "source": "THRESHOLD = 1.5"}]})
    assert latest_cells()[0]["source"] == "THRESHOLD = 1.5"


def test_an_out_of_order_push_is_ignored():
    """Comms are FIFO per channel, but a retry or duplicated send must never move
    the store backwards -- serving an older snapshot than one already seen would
    be worse than serving the file."""
    handle_message({"seq": 5, "cells": [{"cell_type": "code", "id": "a", "source": "NEW"}]})
    handle_message({"seq": 2, "cells": [{"cell_type": "code", "id": "a", "source": "OLD"}]})
    assert latest_cells()[0]["source"] == "NEW"


def test_a_malformed_payload_is_ignored_not_raised():
    """The payload crosses a process boundary from JavaScript. Every shape is
    optional; none may break a user's cell execution."""
    for bad in (None, 42, "cells", {}, {"seq": "x", "cells": []},
                {"seq": 1, "cells": "nope"}, {"seq": 1, "cells": [1, 2, 3]},
                {"seq": True, "cells": []}):
        handle_message(bad)
    assert latest_cells() is None


def test_reset_clears_the_store():
    """Switching notebooks or restarting must not serve the previous one's cells."""
    handle_message({"seq": 1, "cells": CELLS})
    reset()
    assert latest_cells() is None


# --- Wiring: through server_discovery.get_notebook_cells(), not just the store
# directly. Three features in this project have shipped with correct logic and
# missing wiring, each passing every unit test above while doing nothing in
# production -- so this is the test that actually proves the reader is reached.


def _write_notebook(path, source: str) -> None:
    path.write_text(
        json.dumps({
            "cells": [{"cell_type": "code", "source": source}],
            "metadata": {}, "nbformat": 4, "nbformat_minor": 5,
        }),
        encoding="utf-8",
    )


def test_wiring_a_pushed_snapshot_is_served_instead_of_the_file(tmp_path):
    nb_path = tmp_path / "notebook.ipynb"
    _write_notebook(nb_path, "from_file = True")
    sd.invalidate_notebook_cells_cache()

    handle_message({"seq": 1, "cells": CELLS})

    assert sd.get_notebook_cells(str(nb_path)) == [
        "THRESHOLD = 0.9", "y = THRESHOLD * 2",
    ]


def test_wiring_control_nothing_pushed_reads_the_file(tmp_path):
    """The control that makes the test above mean something: with nothing
    pushed, the same entry point must fall through to the file's cells, not
    serve an empty notebook."""
    nb_path = tmp_path / "notebook.ipynb"
    _write_notebook(nb_path, "from_file = True")
    sd.invalidate_notebook_cells_cache()

    assert sd.get_notebook_cells(str(nb_path)) == ["from_file = True"]


# --- A comm_open supersedes the previous connection (CAS-274 review, C2) ------
#
# These drive the REAL closures `register_target` installs, rather than calling
# `handle_message` directly, because the behaviour under test lives in _on_open
# -- and a test that reaches only the store cannot see it.


class _FakeComm:
    """Enough of ipykernel's Comm for the closures in `register_target`."""

    def __init__(self):
        self._handler = None

    def on_msg(self, fn):  # used as a decorator inside _on_open
        self._handler = fn
        return fn

    def deliver(self, data):
        """What the frontend's `comm.send(...)` becomes on this side."""
        self._handler({"content": {"data": data}})


class _FakeCommManager:
    def __init__(self):
        self.targets = {}

    def register_target(self, name, fn):
        self.targets[name] = fn


class _FakeShell:
    def __init__(self):
        self.kernel = SimpleNamespace(comm_manager=_FakeCommManager())


def _open_comm(shell, data=None):
    """Drive one comm_open through the registered target, as ipykernel would."""
    comm = _FakeComm()
    shell.kernel.comm_manager.targets[TARGET](
        comm, {"content": {"data": {} if data is None else data}}
    )
    return comm


def test_a_new_comm_open_resets_the_high_water_mark():
    """A browser reload must not leave cash serving pre-reload source.

    The extension's `seq` is a closure variable in the plugin activation, so it
    restarts at 0 on every page load, while this store lives as long as the
    KERNEL. Without a reset on open, every post-reload push is dropped as
    "older" and cash keeps serving a snapshot of text that -- since a reload
    discards unsaved edits -- no longer exists anywhere.
    """
    shell = _FakeShell()
    assert register_target(shell)

    first = _open_comm(shell)
    for n in (1, 2, 3, 4, 5):
        first.deliver({"seq": n, "cells": [{"cell_type": "code", "id": "a", "source": f"v = {n}"}]})
    assert latest_cells()[0]["source"] == "v = 5"

    # F5. A new comm opens and the frontend's counter starts over.
    second = _open_comm(shell)
    second.deliver({"seq": 1, "cells": [{"cell_type": "code", "id": "a", "source": "AFTER RELOAD"}]})

    assert latest_cells()[0]["source"] == "AFTER RELOAD"


def test_the_open_payload_itself_is_not_treated_as_a_snapshot():
    """`comm.open({})` carries no cells. Resetting on open must not then let the
    empty open payload masquerade as a snapshot of an empty notebook."""
    shell = _FakeShell()
    assert register_target(shell)
    _open_comm(shell, data={})
    assert latest_cells() is None


def test_a_comm_reopened_after_a_close_still_delivers():
    """The kernel-side precondition for the frontend's re-open recovery.

    A fresh kernel refuses the first comm_open outright -- the
    flush-before-execute ordering guarantees it arrives before the cell that
    runs `import cash` registers the target -- so the extension has to open a
    NEW comm on the next flush. That only helps if the target accepts repeat
    opens and wires each comm its own on_msg. A control, not a regression test:
    the fix for that defect is in the frontend, and is pinned by
    test_labextension_packaging.py.
    """
    shell = _FakeShell()
    assert register_target(shell)

    first = _open_comm(shell)
    first.deliver({"seq": 1, "cells": CELLS})
    assert latest_cells()[0]["source"] == "THRESHOLD = 0.9"

    second = _open_comm(shell)
    second.deliver({"seq": 2, "cells": [{"cell_type": "code", "id": "a", "source": "REOPENED"}]})
    assert latest_cells()[0]["source"] == "REOPENED"


def test_wiring_a_reopen_falls_back_to_the_file_until_the_new_snapshot_lands(tmp_path):
    """The direction the reset must fail in.

    Re-opening clears the store, and there is a window -- one message wide --
    before the replacement snapshot arrives. What cash does in that window has
    to be "read the saved .ipynb", never "believe the notebook is empty": an
    empty upstream list would disable the upstream check entirely, which is the
    failure that made an earlier tier strictly worse than not having the feature.
    """
    nb_path = tmp_path / "notebook.ipynb"
    _write_notebook(nb_path, "from_file = True")

    shell = _FakeShell()
    assert register_target(shell)
    first = _open_comm(shell)
    first.deliver({"seq": 1, "cells": CELLS})
    sd.invalidate_notebook_cells_cache()
    assert sd.get_notebook_cells(str(nb_path)) == ["THRESHOLD = 0.9", "y = THRESHOLD * 2"]

    _open_comm(shell)  # reload: store cleared, replacement not yet sent
    sd.invalidate_notebook_cells_cache()
    assert sd.get_notebook_cells(str(nb_path)) == ["from_file = True"]
