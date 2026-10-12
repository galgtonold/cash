"""A change made through another name is rebuilt where a run from the top
makes it: after a restart, after an edit that makes an alias, and not after
one that removes it.

Each statement keeps the edges it linked (``share_edges_key``), and the
simulation walks them in notebook order, from what each statement linked
when it ran, or, for one no run has recorded, from what its text says.
"""

from __future__ import annotations

import pytest

from cash.backends.file_backend import FileBackend
from cash.notebook import cache_key
from tests._cell_driver import run_cash_cell


def _restart(magics, names) -> None:
    """What a kernel restart leaves: no session state, none of *names*."""
    magics.tracking_state.reset_session_state()
    magics._upstream_checker.simulator.cache.reset()
    for name in names:
        magics.shell.user_ns.pop(name, None)


def _codes(magics) -> list[str]:
    return [m.get("code") for m in magics.cash_status("dict")["last_cell"]["statements"] if m.get("code")]


@pytest.fixture
def clean_backend(tmp_path):
    """A file backend: the record a later kernel reads is metadata alone,
    which only a file tier keeps."""
    backend = FileBackend(cache_dir=str(tmp_path / "cache"))
    yield backend
    backend.clear()


def test_after_a_restart_the_change_through_the_alias_is_rebuilt(cash_magics, clean_backend):
    cells = ["x = [1]", "y = x", "y.append(2)", "z = list(x)"]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert cash_magics.shell.user_ns["z"] == [1, 2], "control: the top-to-bottom run"
    record = clean_backend.get_metadata(cache_key.share_edges_key(cache_key.statement_source_hash("y.append(2)")))
    assert record and ["y", "x"] in record["edges"], record

    _restart(cash_magics, ["x", "y", "z"])
    run_cash_cell(cash_magics, cells[-1], cells=cells)

    assert "y.append(2)" in _codes(cash_magics), _codes(cash_magics)
    assert cash_magics.shell.user_ns["z"] == [1, 2]


def test_an_edit_that_makes_an_alias_reruns_the_change_through_it(cash_magics):
    cells = ["x = [1]", "y = list(x)", "y.append(2)", "z = list(x)"]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert cash_magics.shell.user_ns["z"] == [1], "control: no alias yet"

    cells[1] = "y = x"
    run_cash_cell(cash_magics, cells[-1], cells=cells)

    assert cash_magics.shell.user_ns["z"] == [1, 2], _codes(cash_magics)


def test_an_edit_that_removes_an_alias_drops_its_edge(cash_magics):
    """``y = x`` edited to ``y = [0]``: ``x = x + [5]`` below reads and
    rebinds ``x``, which the bump takes for a change in place; through the
    edge the kernel still had, it was rebuilt as a producer of ``y``."""
    cells = ["x = [1]", "y = x", "x = x + [5]", "n = len(y)"]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert cash_magics.shell.user_ns["n"] == 1, "control: the top-to-bottom run"

    cells[1] = "y = [0, 0]"
    run_cash_cell(cash_magics, cells[-1], cells=cells)

    assert cash_magics.shell.user_ns["n"] == 2
    assert "x = x + [5]" not in _codes(cash_magics), _codes(cash_magics)


matplotlib = pytest.importorskip("matplotlib")


def test_after_a_restart_a_drawing_through_pyplot_is_rebuilt(cash_magics, clean_backend):
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells = [
        "import matplotlib.pyplot as plt",
        "fig = plt.figure()",
        "plt.plot([1, 2])",
        "n = len(fig.axes[0].lines) if fig.axes else 0",
    ]
    try:
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)
        assert cash_magics.shell.user_ns["n"] == 1, "control: the top-to-bottom run"

        plt.close("all")
        _restart(cash_magics, ["plt", "fig", "n"])
        run_cash_cell(cash_magics, cells[-1], cells=cells)

        assert "plt.plot([1, 2])" in _codes(cash_magics), _codes(cash_magics)
        assert cash_magics.shell.user_ns["n"] == 1
    finally:
        plt.close("all")


def test_a_statement_that_linked_nothing_is_not_read_from_its_text(cash_magics):
    """``y = x[0]`` takes a number out of ``x``: its text would link the two,
    its run linked nothing. Taken from the text, ``x.append(3)`` moved ``y``
    in the simulation and not in the kernel, and every run rebuilt ``y``."""
    cells = ["x = [1, 2]", "y = x[0]", "x.append(3)", "z = y + 1"]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    for _ in range(2):
        run_cash_cell(cash_magics, cells[-1], cells=cells)
        assert _codes(cash_magics) == ["z = y + 1"], _codes(cash_magics)
