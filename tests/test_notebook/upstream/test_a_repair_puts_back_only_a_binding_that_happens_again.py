"""A repair puts back a name a later cell bound only when that binding happens again.

``n, m = [5], [8]`` re-run to bring ``m`` up to date binds ``n`` too. When a
cell between it and the cell being run binds ``n`` unconditionally
(``n = [2]``), a run from the top has that ``n`` where the cell starts, and
plain Jupyter has it too: the live one is put back
(``UpstreamChecker._keep_live_bindings``). When the binding is on a branch
(``if m[0] > 7: n = [2]``), it may not happen again, and the cell reading
``n`` by a string (``globals()['n']``) reads it unseen: there the repaired
value stands, as a run from the top has it.
"""

from __future__ import annotations

import pytest

from cash.notebook.upstream.checker import UpstreamChecker
from tests._cell_driver import run_cash_cell


@pytest.fixture
def magics(cash_magics):
    cash_magics.cash_on("")
    cash_magics.badges.mode = "off"
    return cash_magics


@pytest.fixture
def no_put_back(monkeypatch):
    monkeypatch.setattr(UpstreamChecker, "_keep_live_bindings", lambda self, *a, **k: None)


def _run(m, cells, *idx):
    for i in idx:
        run_cash_cell(m, cells[i], cells=cells)


def _rebound_between(magics):
    ns = magics.shell.user_ns
    v1 = ["n, m = [1], [7]", "n = [2]", "seen_m = list(m)"]
    _run(magics, v1, 0, 1, 2)
    keep = ns["n"]
    v2 = ["n, m = [5], [8]", *v1[1:]]
    _run(magics, v2, 2)
    return ns, keep


def test_a_name_bound_between_keeps_its_live_value(magics):
    ns, keep = _rebound_between(magics)
    assert ns["seen_m"] == [8]
    assert ns["n"] is keep  # a run from the top has [2] here, and so does plain Jupyter


@pytest.mark.usefixtures("no_put_back")
def test_without_the_put_back_the_repair_leaves_the_name_from_above(magics):
    """Guard for the test above: without the put-back the repair's ``n``
    stays, so that test is what proves it."""
    ns, keep = _rebound_between(magics)
    assert ns["n"] == [5]


@pytest.fixture(params=["put back", "no put back"])
def either(request, magics, monkeypatch):
    if request.param == "no put back":
        monkeypatch.setattr(UpstreamChecker, "_keep_live_bindings", lambda self, *a, **k: None)
    return magics


def test_a_cell_below_binding_on_a_branch_is_not_put_back(either):
    ns = either.shell.user_ns
    below = "if m[0] > 7:\n    n = [2]\nflag = m[0] > 7"
    v1 = ["n, m = [1], [9]", "seen_m = list(m)", below, "seen_n = list(n)"]
    _run(either, v1, 0, 1, 2, 3)
    assert ns["seen_n"] == [2]
    v2 = ["n, m = [5], [3]", *v1[1:]]
    _run(either, v2, 1, 2, 3)
    assert ns["flag"] is False
    assert ns["seen_n"] == [5]  # what a run from the top gives


def test_the_cell_binding_on_a_branch_is_not_put_back(either):
    ns = either.shell.user_ns
    cur = "if m[0] > 7:\n    n = [2]\nseen_m = list(m)"
    v1 = ["n, m = [1], [9]", cur, "seen_n = list(n)"]
    _run(either, v1, 0, 1, 2)
    v2 = ["n, m = [5], [3]", cur, "seen_n = list(n)"]
    _run(either, v2, 1, 2)
    assert ns["seen_n"] == [5]


def test_a_cell_reading_it_by_a_string_gets_the_repaired_value(either):
    ns = either.shell.user_ns
    v1 = ["n, m = [1], [7]", "seen = (list(m), list(globals()['n']))", "n = [2]"]
    _run(either, v1, 0, 1, 2)
    v2 = ["n, m = [5], [8]", *v1[1:]]
    _run(either, v2, 1)
    assert ns["seen"] == ([8], [5])
