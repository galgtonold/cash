"""A call that only reads a memo on a class of the user's is served only
while the memo holds what it read.

``r = fit(1)`` returns a ``Model`` whose class holds a ``functools.cache``
method. When an earlier cell already filled it, the call only reads it; the
entry was stored, then served after a restart with the memo empty, so the
call's fill never happened: ``Model.lookup`` ran again later, printing where
plain Python prints nothing. Without a trace the call was not stored at all.
The entry now records the memo's answers and is served only while the memo
holds them again; after a restart the call runs and fills it.
"""

from __future__ import annotations

import sys

import pytest

from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S as SLEEP
from tests.test_notebook.restore.test_review_memofill4_e2e import NAMES, _clean, _plain, _restart, _run_all

MODEL = """
import functools

class Model:
    @staticmethod
    @functools.cache
    def lookup(n):
        print("LOOKUP", n)
        return n * 2
"""

FIT = f"""
import time

def fit(n):
    import class_memo_read_model  # only here: the notebook does not import it
    m = class_memo_read_model.Model()
    m.lookup(n)
    time.sleep({SLEEP})
    return m
"""

NAME = "class_memo_read_fit"
MODS = (NAME, "class_memo_read_model")


def _drop():
    for mod in MODS:
        sys.modules.pop(mod, None)


@pytest.fixture
def user_module(tmp_path, monkeypatch):
    (tmp_path / f"{NAME}.py").write_text(FIT, encoding="utf-8")
    (tmp_path / "class_memo_read_model.py").write_text(MODEL, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    _drop()
    yield
    _drop()


@pytest.mark.parametrize("trace", ["on", "off"])
def test_a_call_that_read_a_filled_memo_runs_after_a_restart(cash_magics, capsys, user_module, monkeypatch, trace):
    # ``r`` reads what ``a`` filled. After a restart ``a`` was served, so
    # the memo stayed empty, and ``r`` was served too: the reader filled it,
    # printing LOOKUP where plain Python prints it in ``a``.
    if trace == "off":
        monkeypatch.setenv("CASH_TRACE", "0")
    cells = [
        f"import {NAME}",
        f"a = {NAME}.fit(1)",
        f"r = {NAME}.fit(1)",
        "y = type(r).lookup(1)",
    ]
    _, plain_out = _plain(cells)
    _drop()
    _run_all(cash_magics, cells, capsys)
    _restart(cash_magics, NAMES + [NAME, "a", "r", "y"])
    _drop()

    _, out = _run_all(cash_magics, cells, capsys)
    assert [_clean(o) for o in out] == plain_out == ["", "LOOKUP 1\n", "", ""]


def test_a_call_that_read_a_filled_memo_is_served_while_it_is_filled(cash_magics, capsys, user_module):
    cells = [f"import {NAME}", f"a = {NAME}.fit(1)", f"r = {NAME}.fit(1)"]
    _run_all(cash_magics, cells, capsys)
    seen, _ = _run_all(cash_magics, cells[2:], capsys)
    assert seen == [["MIXED"]], "the call inside was served"
