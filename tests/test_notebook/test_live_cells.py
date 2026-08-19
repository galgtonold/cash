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

from IPython.core.events import available_events

from cash.notebook import server_discovery as sd
from cash.notebook.live_cells import (
    TARGET,
    expire,
    handle_message,
    install_expiry_hook,
    latest_cells,
    register_target,
    reset,
)

CELLS = [
    {"cell_type": "code", "id": "a", "source": "THRESHOLD = 0.9"},
    {"cell_type": "code", "id": "b", "source": "y = THRESHOLD * 2"},
]


def setup_function():
    reset()


def teardown_function():
    # Belt-and-suspenders: a test that armed the store (or the file-signature
    # caches _try_extension_cells sits in front of) must not leak it into
    # whatever test xdist schedules next on this worker. `_try_extension_cells`
    # is the FIRST branch of `_read_notebook_code_cells` and ignores
    # `notebook_path` entirely, so a leaked snapshot here silently overrides
    # every other reader -- observed taking down test_colab_cell_source.py and
    # test_vscode_cell_source.py under `--dist worksteal`, which (unlike a
    # sequential run) can schedule this file's tests interleaved with theirs.
    reset()
    sd.invalidate_notebook_cells_cache()


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
    optional; none may break a user's cell execution.

    This covers both the CONTAINER shapes (top-level, seq, cells) and the
    FIELD shapes inside an otherwise well-formed cell dict: a non-str
    `source` (int, None, list, dict, bool) passes `_extract_cell_entry`
    (server_discovery.py) UNCHANGED -- it only special-cases `list` for the
    join, so anything else threads through as a "cell source" all the way to
    `ast.parse`, which raises `TypeError`. That is not a `(SyntaxError,
    ValueError)` the upstream checker catches, so it would reach the user out
    of a cell they never touched.
    """
    for bad in (None, 42, "cells", {}, {"seq": "x", "cells": []},
                {"seq": 1, "cells": "nope"}, {"seq": 1, "cells": [1, 2, 3]},
                {"seq": True, "cells": []},
                {"seq": 1, "cells": [{"cell_type": "code", "source": 123}]},
                {"seq": 1, "cells": [{"cell_type": "code", "source": None}]},
                {"seq": 1, "cells": [{"cell_type": "code", "source": True}]},
                {"seq": 1, "cells": [{"cell_type": "code", "source": {}}]}):
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


def test_wiring_a_non_string_source_falls_back_to_the_file_not_ast_parse(tmp_path):
    """CAS-274 review, item 2. A cell whose `source` is not a string must not
    become a `[123]`-shaped "notebook cell" that reaches `ast.parse` --
    `_extract_cell_entry` uses a non-list `source` UNCHANGED, and `ast.parse`
    raises `TypeError` on it, which the upstream checker's `except
    (SyntaxError, ValueError)` does not catch. Drives the real reader chain,
    not just `handle_message`+`latest_cells()`, because that is where the
    bad value used to surface."""
    nb_path = tmp_path / "notebook.ipynb"
    _write_notebook(nb_path, "from_file = True")
    sd.invalidate_notebook_cells_cache()

    handle_message({"seq": 1, "cells": [{"cell_type": "code", "source": 123}]})

    assert latest_cells() is None
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


# --- A snapshot is valid for ONE execution (CAS-274 review, Critical) ---------
#
# `_store` lives as long as the KERNEL; the frontend that fills it does not. A
# page reload with the extension disabled, a second client attached without it,
# or the extension erroring after having worked once all stop the pushes WITHOUT
# closing the comm, so `reset()`-on-open never runs. Before `expire`, the last
# snapshot was then served for the rest of the kernel's life: cash compared the
# user's edited upstream cell against frozen text, found no change -- and,
# because "extension" is in `staleness._LIVE_SOURCES`, suppressed the notice
# that would have said cash could not see unsaved edits. Confidently wrong AND
# silent, which is worse than the staleness the feature exists to fix.


def test_a_snapshot_does_not_outlive_the_execution_it_arrived_for():
    handle_message({"seq": 1, "cells": CELLS})
    assert latest_cells() is not None
    expire()
    assert latest_cells() is None


def test_every_read_within_one_execution_still_sees_the_snapshot():
    """cash reads the cells several times per cell -- measured at 7 on a real
    kernel. Expiry is per EXECUTION, not per read, so all of them must succeed
    off one push; a consume-on-read design would serve the first and starve the
    rest."""
    handle_message({"seq": 1, "cells": CELLS})
    for _ in range(7):
        assert latest_cells()[0]["source"] == "THRESHOLD = 0.9"
    expire()
    assert latest_cells() is None


def test_a_fresh_push_re_arms_the_store_after_an_expiry():
    """The control: expiry must not be a one-way door, or the extension would
    work for exactly one cell and then look permanently absent."""
    handle_message({"seq": 1, "cells": CELLS})
    expire()
    handle_message({"seq": 2, "cells": [{"cell_type": "code", "id": "a", "source": "NEXT"}]})
    assert latest_cells()[0]["source"] == "NEXT"


def test_a_retransmitted_snapshot_cannot_re_arm_itself_after_expiry():
    """`expire` retires the cells but NOT the sequence high-water mark.

    Dropping `seq` too would let a duplicate or retried send of the snapshot
    just retired re-arm itself for the next execution -- reintroducing exactly
    the staleness being closed. Only `reset` (a new frontend connection, a new
    notebook) may move the mark backwards.
    """
    handle_message({"seq": 4, "cells": CELLS})
    expire()
    handle_message({"seq": 4, "cells": CELLS})       # duplicate
    handle_message({"seq": 3, "cells": CELLS})       # late retry of an older one
    assert latest_cells() is None


def test_wiring_a_frontend_that_stops_pushing_falls_back_to_the_file(tmp_path):
    """THE regression, through the production entry point.

    One execution's worth of pushes, then silence -- the state a reload with the
    extension disabled leaves behind, with the comm never re-opened. cash must
    read the saved .ipynb from the next execution onward, and must SAY it did.
    """
    nb_path = tmp_path / "notebook.ipynb"
    _write_notebook(nb_path, "from_file = True")

    handle_message({"seq": 1, "cells": CELLS})
    sd.invalidate_notebook_cells_cache()
    assert sd.get_notebook_cells(str(nb_path)) == ["THRESHOLD = 0.9", "y = THRESHOLD * 2"]
    assert sd.last_cell_source() == "extension"

    expire()  # end of that execution; the frontend pushes nothing more

    sd.invalidate_notebook_cells_cache()
    assert sd.get_notebook_cells(str(nb_path)) == ["from_file = True"]
    assert sd.last_cell_source() == "file"


def test_wiring_the_unsaved_edits_notice_fires_again_once_a_snapshot_expires():
    """The other half of the Critical, and the half a cells-only test misses.

    `last_cell_source()` feeds `StalenessTracker`, and "extension" counts as a
    live source -- so a store that kept serving a dead frontend's snapshot also
    switched OFF the once-per-session "cash cannot see unsaved edits here"
    notice. Being wrong is bad; being wrong with the warning suppressed is the
    failure this feature must never produce.
    """
    from cash.notebook.staleness import StalenessTracker

    handle_message({"seq": 1, "cells": CELLS})
    sd.invalidate_notebook_cells_cache()
    sd.get_notebook_cells(str(_MISSING_NOTEBOOK))
    live = StalenessTracker()
    live.note_source(sd.last_cell_source())
    assert live.can_verify()
    assert not live.take_unverifiable_announcement()

    expire()

    sd.invalidate_notebook_cells_cache()
    sd.get_notebook_cells(str(_MISSING_NOTEBOOK))
    dead = StalenessTracker()
    dead.note_source(sd.last_cell_source())
    assert not dead.can_verify()
    assert dead.take_unverifiable_announcement()


#: A path that cannot exist, so the file reader contributes nothing and the only
#: variable in the test above is which reader answered.
_MISSING_NOTEBOOK = "no-such-notebook-cas274.ipynb"


# --- The hook that calls `expire` -- registration, not just the function -----


class _FakeEvents:
    """Mimics the part of IPython's ``EventManager`` this file depends on --
    including its closed set of event names. Real ``InteractiveShell.__init__``
    builds ``self.events = EventManager(self, available_events)``, and
    ``EventManager.__init__`` pre-populates ``callbacks`` from that same
    ``available_events`` dict, so register/unregister/trigger on an unknown
    name raise ``KeyError`` rather than accepting it. Seeding ``registered``
    from the real ``available_events`` (imported above) reproduces that: a
    typo in ``"post_run_cell"`` must fail every test below, not pass them
    silently the way a plain ``defaultdict``-style fake would.
    """

    def __init__(self):
        self.registered = {name: [] for name in available_events}

    def register(self, event, fn):
        self.registered[event].append(fn)   # KeyError on an unknown event, as IPython's does

    def unregister(self, event, fn):
        self.registered[event].remove(fn)   # raises when absent, as IPython's does

    def trigger(self, event, *a):
        for fn in list(self.registered[event]):
            fn(*a)


def test_the_expiry_hook_is_registered_on_post_run_cell_and_retires_a_snapshot():
    """Wiring, not logic: `expire` being correct is worth nothing if nothing
    calls it. Drives the registered callback through the event, the way IPython
    does, rather than calling `expire` directly."""
    shell = SimpleNamespace(events=_FakeEvents())
    assert install_expiry_hook(shell)
    assert shell.events.registered["post_run_cell"]

    handle_message({"seq": 1, "cells": CELLS})
    shell.events.trigger("post_run_cell", None)      # IPython passes the result
    assert latest_cells() is None


def test_re_installing_the_hook_does_not_stack_registrations():
    """`cash.reset_session()`, a second `Cash()`, or `%load_ext` after a reset
    all re-run the hook install. The callback is idempotent, so a duplicate
    would not corrupt anything -- but the list would grow without bound for the
    life of the kernel, which is the same leak the other hooks unregister to
    avoid."""
    shell = SimpleNamespace(events=_FakeEvents())
    for _ in range(3):
        assert install_expiry_hook(shell)
    assert len(shell.events.registered["post_run_cell"]) == 1


def test_a_shell_without_events_is_a_false_not_a_raise():
    """MockShell and embedded interpreters have no event manager. Absence is
    normal here, and harmless: nothing pushes a snapshot there either."""
    assert install_expiry_hook(SimpleNamespace()) is False


def test_install_expiry_hook_survives_a_keyerror_from_events_register():
    """CAS-274 review, item 3. `_FakeEvents` above models IPython's real
    `EventManager`, which raises `KeyError` -- not `AttributeError` or
    `TypeError` -- on an unknown event name, because `callbacks` is
    pre-populated from a closed set (`available_events`).

    Not reachable through the hardcoded "post_run_cell" with a real IPython
    shell -- that name is always in `available_events` -- so this simulates a
    still-narrower event set the way a future or embedded `EventManager`
    might have one. The production catch must be at least as wide as this
    file's own model of what can go wrong, or exactly this would propagate
    out of `install_expiry_hook`, out of `_init_session_state`, i.e. out of
    `import cash`.
    """
    shell = SimpleNamespace(events=_FakeEvents())
    del shell.events.registered["post_run_cell"]
    assert install_expiry_hook(shell) is False


def test_wiring_cashmagics_installs_both_halves_of_the_live_cell_contract():
    """The call sites, not the functions.

    `expire` and `register_target` can both be perfect while `magics.py` calls
    neither, and every test above would still pass -- the "correct logic,
    missing wiring" failure this file's other wiring tests exist to catch.
    Drives `_init_session_state`, the method that installs cash's shell hooks,
    and checks that the comm target AND the expiry hook both ended up attached.
    """
    from cash.notebook.ipython.magics import CashMagics
    from cash.notebook.live_cells import _on_post_run_cell

    shell = _FakeShell()
    shell.events = _FakeEvents()
    shell.run_cell = lambda code, *a, **k: None
    shell.run_cell_async = None

    magics = CashMagics.__new__(CashMagics)
    magics._init_session_state(shell)

    assert TARGET in shell.kernel.comm_manager.targets, "the comm target was never registered"
    assert _on_post_run_cell in shell.events.registered.get("post_run_cell", []), (
        "nothing retires a pushed snapshot, so it would outlive its execution"
    )

    # ...and the registered callback really is the one that retires a snapshot.
    handle_message({"seq": 1, "cells": CELLS})
    shell.events.trigger("post_run_cell", None)
    assert latest_cells() is None
