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

from cash.notebook import server_discovery as sd
from cash.notebook.live_cells import handle_message, latest_cells, reset

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
