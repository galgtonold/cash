"""A function of a local module that keeps state in the module sets it again
when it is handed to a call, as when it is called by name.

``helpers.record(v)`` appending to ``helpers.SEEN`` runs every time; handed to
``s.apply(helpers.record)``, or to ``s.map(count)`` after ``from helpers
import count``, the statement was served from the cache on the next run and
the module state the cell above reset stayed empty.
"""

import os
import sys
import warnings

import pytest

from cash.notebook.callee_reach import module_state_writes
from tests._cell_driver import run_cash_cell

HELPERS = """import time
SEEN = []
STATS = {"n": 0}
def record(v):
    time.sleep(0.07)
    SEEN.append(v)
    return v * 2
def count(v):
    time.sleep(0.07)
    STATS["n"] += 1
    return v
def double(v):
    return v * 2
"""


@pytest.fixture
def helpers(tmp_path, monkeypatch):
    name = f"_handed_helpers_{os.getpid()}_{id(tmp_path)}"
    (tmp_path / f"{name}.py").write_text(HELPERS, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    yield name
    sys.modules.pop(name, None)


@pytest.mark.parametrize(
    ("statement", "check"),
    [
        ("r = s.apply({m}.record)", "len({m}.SEEN)"),
        ("r = s.map(count)", "{m}.STATS['n']"),
    ],
    ids=["module-attribute", "from-imported"],
)
def test_every_run_sets_the_state_again(cash_magics, mock_shell, helpers, statement, check):
    cells = [
        f"import pandas as pd\nimport {helpers}\nfrom {helpers} import count\ns = pd.Series(range(4))",
        f"{helpers}.SEEN.clear()\n{helpers}.STATS['n'] = 0",
        statement.format(m=helpers),
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(3):
            for cell in cells:
                run_cash_cell(cash_magics, cell, cells=cells)
            assert eval(check.format(m=helpers), mock_shell.user_ns) == 4


def test_a_handed_module_function_is_followed(helpers):
    import importlib

    module = importlib.import_module(helpers)
    ns = {"m": module, "count": module.count, "double": module.double, "ops": {"rec": module.record}}
    assert module_state_writes("r = s.apply(m.record)", ns) == {helpers}
    assert module_state_writes("r = s.map(count)", ns) == {helpers}
    assert module_state_writes("r = s.apply(ops['rec'])", ns) == {helpers}
    # Control: a function of the module that sets nothing on it.
    assert module_state_writes("r = s.apply(m.double)", ns) == frozenset()
