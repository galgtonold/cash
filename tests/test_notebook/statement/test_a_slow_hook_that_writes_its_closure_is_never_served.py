"""A call of a function that changes what it closes over (``lst.append(v)``
in ``make``'s lambda) must run each time: served from the cache, the write
never happens, and the list stays empty on the next Run All.

cash read a lambda's closure writes from its source, which is the line it
is on (``return lambda v: ...``) and not a ``def``: it found none, and a
hook slow enough to be worth storing was served. A quick hook hid it,
until a loaded machine made one slow. The hook is read where the notebook
holds it too (``hooks['f'](5)``, ``h.f(5)``); the ``def`` shapes are
controls.
"""

from __future__ import annotations

import pytest

from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

SLEEP = f"time.sleep({ABOVE_PERSISTENCE_FLOOR_S})"
LAMBDA = f"import time\nstore = []\ndef make(lst):\n    return lambda v: ({SLEEP}, lst.append(v))"
NESTED = (
    f"import time\nstore = []\ndef make(lst):\n    def add(v):\n        {SLEEP}\n        lst.append(v)\n    return add"
)
SHAPES = {
    "a lambda by name": (LAMBDA, ["f = make(store)", "f(5)"]),
    "a lambda in a dict": (LAMBDA, ["hooks = {'f': make(store)}", "hooks['f'](5)"]),
    "a lambda in a list": (LAMBDA, ["hooks = [make(store)]", "hooks[0](5)"]),
    "a lambda on an object": (LAMBDA, ["class H:\n    pass\nh = H()\nh.f = make(store)", "h.f(5)"]),
    "a def in a dict": (NESTED, ["hooks = {'f': make(store)}", "hooks['f'](5)"]),
    "a def on an object": (NESTED, ["class H:\n    pass\nh = H()\nh.f = make(store)", "h.f(5)"]),
}


@pytest.mark.parametrize("shape", list(SHAPES))
def test_a_hook_writing_its_closure_runs_on_every_run_all(cash_magics, shape):
    setup, cells = SHAPES[shape]
    cells = [setup, *cells, "r = list(store)"]
    for run in ("first", "second", "third"):
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)
        assert cash_magics.shell.user_ns["r"] == [5], f"{run} Run All"


@pytest.mark.parametrize("shape", list(SHAPES))
def test_a_hook_whose_source_is_gone_runs_on_every_run_all(cash_magics, monkeypatch, shape):
    """A hook whose source cannot be read is taken to write its closure,
    not to write nothing: in a long run of the suite a lambda's cell had
    gone from where its source is read, and the hook was served."""
    import cash.notebook.call_key as call_key

    def gone(fn):
        raise OSError("could not get source code")

    monkeypatch.setattr(call_key, "own_source", gone)
    call_key._GLOBAL_MUTATION_CACHE.clear()
    test_a_hook_writing_its_closure_runs_on_every_run_all(cash_magics, shape)
