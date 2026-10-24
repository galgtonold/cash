"""A function's default, changed in place by a cell below, is said to be dropped.

``def draw(k, rows=BIG)`` reads ``BIG`` through its default, not by name.
Re-run on its own after a cell below ran ``BIG.append(...)``, the calling
cell gets ``draw`` and ``BIG`` as the cells above leave them, as a run from
the top does, and computes without the change. That is the result a run
from the top gives; but where the cell names ``BIG`` itself NOTEBOOK-CHANGED-
BELOW says so, and through the default it said nothing.
"""

from __future__ import annotations

import warnings

import pytest

from tests._cell_driver import run_cash_cell

pytestmark = pytest.mark.core

SETUP = (
    "import time\nBIG = [{'id': i} for i in range({N})]\n"
    "def draw(k, rows=BIG):\n    time.sleep(0.05)\n    return len(rows)\n"
)

CHANGES = {
    "append": "BIG.append({'id': -1})",
    "extend": "BIG.extend([1, 2])",
    "item-set": "BIG[0] = 'x'",
}


def _rerun_after(cash_magics, cells, reader, change):
    run_cash_cell(cash_magics, cells[0], cells=cells)
    for _ in range(3):
        run_cash_cell(cash_magics, reader, cells=cells)
    run_cash_cell(cash_magics, change, cells=cells)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_cash_cell(cash_magics, reader, cells=cells)
    return [str(w.message) for w in caught if "NOTEBOOK-CHANGED-BELOW" in str(w.message)]


@pytest.mark.parametrize("size", [3, 20_000], ids=["small", "big"])
@pytest.mark.parametrize("change", sorted(CHANGES))
def test_a_default_changed_below_is_said_to_be_dropped(cash_magics, size, change):
    setup = SETUP.replace("{N}", str(size))
    reader = "y = draw(1)\nprint(y)"
    said = _rerun_after(cash_magics, [setup, reader, CHANGES[change]], reader, CHANGES[change])
    assert cash_magics.shell.user_ns["y"] == size, "not the value a run from the top gives"
    assert said, "the change below was dropped without a word"
    assert "reads 'draw', which holds 'BIG', which cell 3 below changed in place" in said[0], said[0]


def test_an_object_a_default_holds_changed_below_is_said_to_be_dropped(cash_magics):
    setup = "import time\nclass Box:\n    pass\nB = Box()\nB.n = 1\ndef read(k, box=B):\n    time.sleep(0.05)\n    return box.n"
    reader = "y = read(1)"
    said = _rerun_after(cash_magics, [setup, reader, "B.n = 2"], reader, "B.n = 2")
    assert cash_magics.shell.user_ns["y"] == 1, "not the value a run from the top gives"
    assert said, "the change below was dropped without a word"


def test_a_result_sharing_elements_with_a_default_is_not_said_to_be_held(cash_magics):
    """``z = draw(2)`` below is a new list of ``BIG``'s own items, so it
    shares objects with ``BIG``; ``z.append(9)`` changes only ``z``, which
    ``draw`` does not hold, and nothing the calling cell reads is dropped."""
    setup = SETUP.replace("{N}", "3").replace("return len(rows)", "return list(rows[:2])")
    reader = "y = draw(1)"
    change = "z = draw(2)\nz.append(9)"
    said = _rerun_after(cash_magics, [setup, reader, change], reader, change)
    assert cash_magics.shell.user_ns["y"] == [{"id": 0}, {"id": 1}]
    assert not said, said


def test_a_default_holding_it_past_a_thousand_items_is_said_to_be_held(cash_magics):
    """``rows=[*1500 records, BIG]``: ``BIG`` sits past the first thousand
    items, and the change below is still said to be dropped."""
    setup = (
        "import time\nBIG = [{'id': i} for i in range(20000)]\n"
        "def draw(k, rows=[*[{'i': i} for i in range(1500)], BIG]):\n    time.sleep(0.05)\n    return len(rows[-1])\n"
    )
    reader = "y = draw(1)"
    change = CHANGES["append"]
    said = _rerun_after(cash_magics, [setup, reader, change], reader, change)
    assert cash_magics.shell.user_ns["y"] == 20_000, "not the value a run from the top gives"
    assert said, "the change below was dropped without a word"
    assert "reads 'draw', which holds 'BIG'" in said[0], said[0]


def test_a_walk_cut_short_still_says_it_may_be_dropped(cash_magics, monkeypatch):
    """Past the walk's limit, what ``draw`` holds is not all seen: ``BIG``,
    which shares an object with it, may be one of them, and the change below
    is said to be dropped rather than nothing said."""
    from cash.notebook.upstream import changed_below

    monkeypatch.setattr(changed_below, "_HOLD_ITEMS", 100)
    setup = (
        "import time\nBIG = [{'id': i} for i in range(3)]\n"
        "def draw(k, rows=[*[{'i': i} for i in range(150)], BIG]):\n    time.sleep(0.05)\n    return len(rows[-1])\n"
    )
    reader = "y = draw(1)"
    change = CHANGES["append"]
    said = _rerun_after(cash_magics, [setup, reader, change], reader, change)
    assert cash_magics.shell.user_ns["y"] == 3, "not the value a run from the top gives"
    assert said, "the walk ran out and the change below was dropped without a word"
    assert "reads 'draw', which may hold 'BIG'" in said[0], said[0]


def test_a_closure_cell_a_call_rebound_to_it_is_said_to_be_held(cash_magics):
    """``draw, rebind = make([0])`` then ``rebind(BIG)``: no statement binds
    ``draw`` to ``BIG``, yet its closure cell holds it, and ``BIG.append``
    below changes what it reads."""
    setup = (
        "import time\nBIG = [{'id': i} for i in range(20000)]\n"
        "def make(rows):\n    def draw(k):\n        time.sleep(0.05)\n        return len(rows)\n"
        "    def rebind(r):\n        nonlocal rows\n        rows = r\n    return draw, rebind\n"
        "draw, rebind = make([0])\nrebind(BIG)\n"
    )
    reader = "y = draw(1)"
    change = CHANGES["append"]
    said = _rerun_after(cash_magics, [setup, reader, change], reader, change)
    assert cash_magics.shell.user_ns["y"] == 20_000, "not the value a run from the top gives"
    assert said, "the change below was dropped without a word"
    assert "reads 'draw', which holds 'BIG'" in said[0], said[0]


def test_a_closure_cell_rebound_to_another_list_is_not_said_to_hold_it(cash_magics):
    """``draw`` made from ``BIG`` and rebound to a list of its own holds
    ``BIG`` no longer; a change to ``BIG`` below drops nothing it reads."""
    setup = (
        "import time\nBIG = [{'id': i} for i in range(3)]\n"
        "def make(rows):\n    def draw(k):\n        time.sleep(0.05)\n        return len(rows)\n"
        "    def rebind(r):\n        nonlocal rows\n        rows = r\n    return draw, rebind\n"
        "draw, rebind = make([0])\n"
    )
    reader = "y = draw(1)"
    change = CHANGES["append"]
    said = _rerun_after(cash_magics, [setup, reader, change], reader, change)
    assert cash_magics.shell.user_ns["y"] == 1
    assert not said, said


MAKE = (
    "import time\nBIG = [{'id': i} for i in range(3)]\n"
    "def make(rows):\n    def draw(k):\n        time.sleep(0.05)\n        return len(rows)\n"
    "    def rebind(r):\n        nonlocal rows\n        rows = r\n    return draw, rebind\n"
)


def test_a_closure_rebound_away_is_rebuilt_with_its_rebind(cash_magics):
    """``draw, rebind = make(BIG)`` then ``rebind([0])``: a run from the top
    leaves ``draw`` reading ``[0]``. Rebuilt for a change to ``BIG`` below,
    it got ``make(BIG)`` again without the ``rebind([0])`` after it and read
    ``BIG``. The call is a change to every function sharing the cell."""
    setup = MAKE + "draw, rebind = make(BIG)\nrebind([0])\n"
    reader = "y = draw(1)"
    change = CHANGES["append"]
    said = _rerun_after(cash_magics, [setup, reader, change], reader, change)
    assert cash_magics.shell.user_ns["y"] == 1, "not the value a run from the top gives"
    assert not said, said


def test_a_reader_edited_after_the_change_below_gets_the_run_from_the_top(cash_magics):
    """``rebind(BIG)`` above, ``BIG.append`` below, then the reader edited:
    the edited reader misses and runs, on ``draw`` as the cells above leave
    it, and the warning says so."""
    setup = MAKE.replace("range(3)", "range(20000)") + "draw, rebind = make([0])\nrebind(BIG)\n"
    reader, edited = "y = draw(1)", "y = draw(1) + 0"
    change = CHANGES["append"]
    cells = [setup, reader, change]
    run_cash_cell(cash_magics, setup, cells=cells)
    run_cash_cell(cash_magics, reader, cells=cells)
    run_cash_cell(cash_magics, change, cells=cells)
    cells = [setup, edited, change]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_cash_cell(cash_magics, edited, cells=cells)
    said = [str(w.message) for w in caught if "NOTEBOOK-CHANGED-BELOW" in str(w.message)]
    assert cash_magics.shell.user_ns["y"] == 20_000, "not the value a run from the top gives"
    assert said and "reads 'draw', which holds 'BIG'" in said[0], said


def test_a_walk_cut_short_with_nothing_linking_them_says_it_may_hold(cash_magics, monkeypatch):
    """``rebind({'rows': [..., BIG]})`` puts ``BIG`` past the walk's limit, and
    no statement links ``draw`` to ``BIG``: what ``draw`` holds is not all
    seen, so the change below is said to be one it may hold."""
    from cash.notebook.upstream import changed_below

    monkeypatch.setattr(changed_below, "_HOLD_ITEMS", 100)
    setup = (
        MAKE.replace("return len(rows)", "return len(rows['rows'][-1])")
        + "draw, rebind = make({'rows': [[0]]})\nrebind({'rows': [*[{'i': i} for i in range(150)], BIG]})\n"
    )
    reader = "y = draw(1)"
    change = CHANGES["append"]
    said = _rerun_after(cash_magics, [setup, reader, change], reader, change)
    assert cash_magics.shell.user_ns["y"] == 3, "not the value a run from the top gives"
    assert said, "the walk ran out and the change below was dropped without a word"
    assert "reads 'draw', which may hold 'BIG'" in said[0], said[0]


@pytest.mark.parametrize("call", ["setup()", "setup(); pass"], ids=["a_helper", "a_helper_in_a_line"])
def test_a_closure_rebound_inside_a_helper_is_rebuilt_with_its_rebind(cash_magics, call):
    """``def setup(): rebind([0])`` then ``setup()``: only ``setup`` was seen
    as the callee, so the reader re-run on its own after a Run All rebuilt
    ``draw`` from ``make(BIG)`` without the rebind and read ``BIG``."""
    setup = MAKE.replace("range(3)", "range(20000)") + "draw, rebind = make(BIG)\ndef setup():\n    rebind([0])\n"
    reader = "y = draw(1)"
    cells = [setup, call, reader, CHANGES["append"]]
    for _ in range(2):
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)
    run_cash_cell(cash_magics, reader, cells=cells)
    assert cash_magics.shell.user_ns["y"] == 1, "not the value a run from the top gives"


def _chain(n: int) -> str:
    return "def h0():\n    rebind([0])\n" + "".join(f"def h{i}():\n    h{i - 1}()\n" for i in range(1, n))


@pytest.mark.parametrize(
    ("defined", "call", "warned"),
    [
        ("def setup():\n    rebind([0])\n", "if True:\n    setup()", False),
        (_chain(300), "h299()", True),
    ],
    ids=["a_helper_called_under_an_if", "a_chain_of_helpers_past_the_cap"],
)
def test_a_closure_rebound_by_a_helper_reached_another_way_is_rebuilt_with_its_rebind(
    cash_magics, defined, call, warned
):
    """``setup()`` under an ``if``, which the branch runs as one unit, or a
    chain of 300 helpers past the 256 the search follows: neither was
    followed to ``rebind``, and the reader re-run on its own after a Run All
    read ``BIG``. Past the chain, the statement runs every time, and says so."""
    setup = MAKE.replace("range(3)", "range(20000)") + "draw, rebind = make(BIG)\n" + defined
    reader = "y = draw(1)"
    cells = [setup, call, reader, CHANGES["append"]]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(2):
            for cell in cells:
                run_cash_cell(cash_magics, cell, cells=cells)
        run_cash_cell(cash_magics, reader, cells=cells)
    assert cash_magics.shell.user_ns["y"] == 1, "not the value a run from the top gives"
    said = [str(w.message) for w in caught if "too long for cash" in str(w.message)]
    assert bool(said) is warned, said


@pytest.mark.parametrize(
    ("statement", "reached"),
    [
        ("def h1():\n    h0()", set()),
        ("h1 = lambda: h0()", set()),
        ("@h0()\ndef h1():\n    pass", {"rebind"}),
        ("def h1(x=h0()):\n    pass", {"rebind"}),
        ("class C:\n    h0()", {"rebind"}),
    ],
    ids=["a_def", "a_lambda", "a_decorator", "a_default", "a_class_body"],
)
def test_a_def_reaches_only_what_it_runs_as_it_is_made(statement, reached):
    """``def h1(): h0()`` runs nothing of its body, so it does not reach the
    closure writer ``rebind`` ``h0`` calls: following its body made each ``def`` of a
    chain of helpers read the whole chain. Its decorators and defaults run."""
    import ast

    from cash.notebook.statement.mutation_routing import callees_changing_their_closure

    ns: dict = {}
    exec(MAKE + "draw, rebind = make(BIG)\ndef h0():\n    rebind([0])\n", ns)
    assert callees_changing_their_closure(ast.parse(statement), ns) == reached
