"""A function decorated with ``@wrapt.decorator``.

wrapt's wrapper answers ``__code__`` with a proxy of the code, whose
instructions and names come from different functions, so reading it for
the globals the call reads raised IndexError, and every later cell stepped
aside with NOTEBOOK-BAILOUT. The call is now keyed by the code the wrapper
holds: the decorated function's and the decorator's.
"""

from __future__ import annotations

import warnings

import pytest

from tests._cell_driver import run_cash_cell

pytest.importorskip("wrapt")

_DECO = (
    "import wrapt\n@wrapt.decorator\ndef deco(wrapped, instance, args, kwargs):\n"
    "    return [SCALE * v for v in wrapped(*args, **kwargs)]"
)
# Slow enough to be stored and served.
_F = "import time\n@deco\ndef f(x):\n    time.sleep(0.2)\n    return [x + G]"


def _run_all(magics, cells):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for cell in cells:
            run_cash_cell(magics, cell, cells=cells)
    bailouts = [str(w.message) for w in caught if "NOTEBOOK-BAILOUT" in str(w.message)]
    return magics.shell.user_ns["y"], bailouts


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        (lambda cells: cells, [11]),
        (lambda cells: ["G = 20", *cells[1:]], [21]),
        (lambda cells: [*cells[:2], _F.replace("x + G", "x + G + 100"), *cells[3:]], [111]),
        (lambda cells: [cells[0], cells[1].replace("SCALE = 1", "SCALE = 3"), *cells[2:]], [33]),
        (lambda cells: [cells[0], _DECO.replace("SCALE * v", "SCALE * v - 1") + "\nSCALE = 1", *cells[2:]], [10]),
    ],
    ids=[
        "unchanged",
        "a global the function reads",
        "the function's body",
        "a global the decorator reads",
        "the decorator's body",
    ],
)
def test_a_wrapt_decorated_call_is_cached_and_follows_what_it_runs(cash_magics, change, expected):
    cells = ["G = 10", _DECO + "\nSCALE = 1", _F, "y = f(1)"]
    first, bailouts = _run_all(cash_magics, cells)
    assert first == [11]
    assert not bailouts
    again, bailouts = _run_all(cash_magics, change(cells))
    assert again == expected
    assert not bailouts
