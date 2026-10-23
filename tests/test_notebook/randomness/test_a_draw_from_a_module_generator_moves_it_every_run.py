"""A draw from a generator a module holds gives the next item on every Run
All, as in plain Python.

``from gens import G`` binds the module's generator, which a second Run All
does not make afresh. A cell drawing from it twice (``t = next(G)``, then
again) was skipped whole on later runs: the simulation of the cell reached
the lineages the first run recorded, so every output looked current and
``w`` stayed ``(1, 10)``. A helper drawing from it (``def d(): return
next(G)``) was served at the call: its key, ``d``'s source and ``G``'s
lineage, does not hold where ``G`` stands, so ``t = d()`` gave the first
item every run. Both run every time now.
"""

from __future__ import annotations

import sys

import pytest

from cash.notebook.call_key import globals_it_draws_from
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

GENS = "def gen():\n    i = 0\n    while True:\n        yield i\n        i += 1\nG = gen()\n"
SLEEP = f"time.sleep({ABOVE_PERSISTENCE_FLOOR_S})"


@pytest.fixture
def gens(tmp_path, monkeypatch):
    (tmp_path / "draw_gens.py").write_text(GENS, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    yield
    sys.modules.pop("draw_gens", None)


def _run_all(magics, cells, names):
    for code in cells:
        run_cash_cell(magics, code)
    return tuple(magics.shell.user_ns[n] for n in names)


def test_a_cell_drawing_twice_is_not_skipped(cash_magics, gens):
    cells = ["from draw_gens import G", "t = next(G)\nu = t + 1\nt = next(G)\nv = t * 10", "w = (u, v)"]
    seen = [_run_all(cash_magics, cells, ["w"])[0] for _ in range(3)]
    assert seen == [(1, 10), (3, 30), (5, 50)]


HELPER = f"import time\ndef d():\n    {SLEEP}\n    return next(G)"

DRAWS = {
    "alone": "t = d()",
    "in_an_if": "if True:\n    t = d()",
    "in_a_loop": "for _ in range(1):\n    t = d()",
    "in_a_try": "try:\n    t = d()\nexcept Exception:\n    pass",
    "in_a_with": "import contextlib\nwith contextlib.nullcontext():\n    t = d()",
}


@pytest.mark.parametrize("draw", list(DRAWS))
def test_a_helper_drawing_from_a_module_generator_draws_every_run(cash_magics, gens, draw):
    cells = ["from draw_gens import G", HELPER, DRAWS[draw], "u = (t, t + 1)"]
    seen = [_run_all(cash_magics, cells, ["u"])[0] for _ in range(3)]
    assert seen == [(0, 1), (1, 2), (2, 3)]


DRAWERS = (
    "import itertools\n"
    "G = iter(range(3))\nC = itertools.count()\nL = [1, 2]\n"
    "def direct():\n    return next(G)\n"
    "def by_loop():\n    for x in C:\n        return x\n"
    "def through():\n    return direct()\n"
    "def reads_a_list():\n    return next(iter(L))\n"
    "def inspects():\n    return type(G)\n"
)


def test_what_a_function_draws_from_is_read_off_its_source(tmp_path, monkeypatch):
    (tmp_path / "draw_kinds.py").write_text(DRAWERS, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    import draw_kinds

    try:
        assert globals_it_draws_from(draw_kinds.direct) == ("G",)
        assert globals_it_draws_from(draw_kinds.by_loop) == ("C",)
        assert globals_it_draws_from(draw_kinds.through) == ("G",)
        assert globals_it_draws_from(draw_kinds.reads_a_list) == ()
        assert globals_it_draws_from(draw_kinds.inspects) == ()
    finally:
        sys.modules.pop("draw_kinds", None)
