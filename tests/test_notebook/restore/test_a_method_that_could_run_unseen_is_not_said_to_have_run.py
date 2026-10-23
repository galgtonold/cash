"""Without a record of what ran, the reason names a method that could run.

With no trace (Python 3.10 and 3.11, ``CASH_TRACE=0``, a debugger holding
``sys.monitoring``) the objects a statement reads are looked through for a
special method that writes outside them, and a value too big to look
through counts every class alive with one. The reason said such a method
"ran in this statement" when nothing showed it did.
"""

from __future__ import annotations

import pytest

from cash.notebook import shadow
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S as F

HEAD = (
    "import time, types\n"
    "LOG = []\n"
    "class Vec:\n"
    "    def __add__(self, other):\n"
    "        LOG.append(other)\n"
    "        return 1\n"
)


@pytest.fixture
def no_trace(monkeypatch):
    monkeypatch.setenv("CASH_TRACE", "0")
    monkeypatch.setattr(shadow, "SHADOW", None)


def _reasons(statement_processor, code):
    return " ".join(statement_processor.process_statement(code)["uncacheable_reasons"])


def test_a_method_reached_could_run(no_trace, cash_magics, statement_processor):
    run_cash_cell(cash_magics, HEAD + "import numpy as np\narr = np.empty(1, dtype=object)\narr[0] = Vec()")
    reasons = _reasons(statement_processor, f"w = (arr + 1, time.sleep({F}))[0]")
    assert "Vec.__add__() could run in this statement (no record of what runs" in reasons, reasons
    assert "ran in this statement" not in reasons


def test_a_value_too_big_to_look_through_says_so(no_trace, cash_magics, statement_processor):
    # 100,001 objects to look at: a cell making 300,000 took cash 11-21 s.
    run_cash_cell(cash_magics, HEAD + "v = Vec()\nbig = [types.SimpleNamespace() for _ in range(100_001)]")
    reasons = _reasons(statement_processor, f"w = (len(big), time.sleep({F}))[0]")
    assert "Vec.__add__() could run in this statement: a value it reads is too big to look through" in reasons, reasons
    assert "ran in this statement" not in reasons


#: Values with nothing of the user's in them, past the bounds by count: a
#: numeric array or frame (items not read), plain items (read at C speed),
#: rows of plain fields or of containers of them (one looked at per class,
#: none for a list, tuple or dict).
BIG_BUT_PLAIN = {
    "floats": "np.zeros((110_000, 10))",
    "frame": "pd.DataFrame(np.zeros((200_000, 10)))",
    "ints": "list(range(2_100_000))",
    "strings": "pd.Series([str(i) for i in range(2_100_000)], dtype=object)",
    "string_columns": "pd.DataFrame({c: np.array([str(i) for i in range(700_000)], dtype=object) for c in 'abc'})",
    "rows": "[Row(i, str(i)) for i in range(25_000)]",
    "rows_with_a_list": "[Tagged(i, [1.0]) for i in range(25_000)]",
    "json": "[{'id': i, 'city': f'c{i % 40}', 'tags': ['a', 'b'][: i % 3]} for i in range(30_000)]",
    "tuples": "[(i, str(i)) for i in range(25_000)]",
    "dict_of_lists": "{i: [i, i + 1] for i in range(25_000)}",
}


@pytest.mark.parametrize("value", list(BIG_BUT_PLAIN))
def test_a_big_value_of_plain_items_is_looked_through(no_trace, cash_magics, statement_processor, value):
    # Counted as too big, any class with a writing method and an object
    # alive (``v``) refused every statement reading it, the method never
    # used on it. A notebook's namespace is ``__main__``: a dataclass made
    # in a cell is the user's on every Python.
    cash_magics.shell.user_ns["__name__"] = "__main__"
    run_cash_cell(
        cash_magics,
        HEAD + "v = Vec()\nimport dataclasses\nimport numpy as np\nimport pandas as pd\n"
        "@dataclasses.dataclass\nclass Row:\n    i: int\n    s: str\n"
        "@dataclasses.dataclass\nclass Tagged:\n    i: int\n    tags: list",
    )
    run_cash_cell(cash_magics, f"big = {BIG_BUT_PLAIN[value]}")
    reasons = _reasons(statement_processor, f"w = (len(big), time.sleep({F}))[0]")
    assert "Vec.__add__" not in reasons, reasons


def test_a_class_with_no_object_alive_is_not_counted(no_trace, cash_magics, statement_processor):
    # Too big to look through, and ``Vec`` has no object alive: none can
    # be inside it.
    run_cash_cell(cash_magics, HEAD + "big = [types.SimpleNamespace() for _ in range(100_001)]")
    reasons = _reasons(statement_processor, f"w = (len(big), time.sleep({F}))[0]")
    assert "Vec.__add__" not in reasons, reasons


def test_a_users_array_is_not_asked_its_length(no_trace, cash_magics):
    # Its ``__len__`` ran, and an error in it failed the cell after it ran.
    run_cash_cell(
        cash_magics,
        HEAD + "import pandas as pd\nclass MyArr(pd.api.extensions.ExtensionArray):\n"
        "    def __init__(self, n):\n        self.n = n\n        self.boom = False\n"
        "    def __len__(self):\n        if self.boom:\n            raise RuntimeError('boom')\n        return self.n\n"
        "bad = MyArr(3)",
    )
    run_cash_cell(cash_magics, "bad.boom = True")
    run_cash_cell(cash_magics, f"w = (bad.n, time.sleep({F}))[0]")
    assert cash_magics.shell.user_ns["w"] == 3


def test_a_walk_that_fails_runs_the_statement_and_says_so(no_trace, cash_magics, statement_processor, monkeypatch):
    from cash.notebook.statement import processor

    def broken(*_args):
        raise RuntimeError("a bug of cash's")

    monkeypatch.setattr(processor, "writers_without_a_trace", broken)
    run_cash_cell(cash_magics, HEAD + "import numpy as np\narr = np.empty(1, dtype=object)\narr[0] = Vec()")
    reasons = _reasons(statement_processor, f"w = (len(arr), time.sleep({F}))[0]")
    assert "could not look through what this statement reads" in reasons, reasons


def test_a_method_set_on_a_class_later_is_named_by_it(cash_magics, statement_processor):
    from cash.decorator import code_trace

    if not code_trace.on():
        pytest.skip("reads the trace of the run (3.12+)")
    run_cash_cell(
        cash_magics,
        "import time\ncalls = []\ndef _add(self, o):\n    calls.append(1)\n    return 1\n"
        "class Vec:\n    pass\nVec.__add__ = _add\nd = {'a': [Vec()]}\nk = 'a'",
    )
    reasons = _reasons(statement_processor, f"w = (d[k][0] + 1, time.sleep({F}))[0]")
    assert "Vec.__add__() ran in this statement" in reasons, reasons
