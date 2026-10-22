"""What a ``from helpers import NAME`` adds to the lineage of ``NAME``: its
value's digest (`lineage_formula._imported_value_component`).

A value too big to read each time is decided by its size, not by the time
reading it took: a 3-million entry dict was read in full three times per
Run All until a time budget ran out, and a 1-million entry one was taken
as new or not by how loaded the machine was. A value only its identity
hashes (an object holding a lock) does not move when it changes in place,
so it is new each time too; a partial is read by its parts.
"""

from __future__ import annotations

import functools
import threading

from cash import value_hash
from cash.notebook import lineage_formula


def _component(value):
    return lineage_formula._imported_value_component("_no_such_module_", "VALUE", value)


def test_a_value_past_the_size_is_new_each_time_without_being_read(monkeypatch):
    def unread(*_args, **_kwargs):
        raise AssertionError("read")

    monkeypatch.setattr(lineage_formula, "FOLLOWED_ITEMS", 10)
    monkeypatch.setattr(lineage_formula, "compute_hash", unread)
    monkeypatch.setattr(lineage_formula, "budgeted_hash", unread)
    big = {i: str(i) for i in range(11)}

    assert _component(big) != _component(big)


def test_a_value_within_the_size_is_read_whatever_the_time_budget(monkeypatch):
    # The budget gone: a dict near it was new one run in four.
    monkeypatch.setattr(value_hash, "CHANGE_CHECK_BUDGET_S", 0.0)
    table = {i: str(i) for i in range(1000)}

    assert _component(table) == _component(dict(table))
    changed = dict(table)
    changed[0] = "x"
    assert _component(changed) != _component(table)


def test_a_partial_is_read_by_its_parts():
    def inc(x, k=1):
        return x + k

    assert _component(functools.partial(inc, k=1)) == _component(functools.partial(inc, k=1))
    assert _component(functools.partial(inc, k=1)) != _component(functools.partial(inc, k=2))


class _Service:
    def __init__(self):
        self.lock = threading.Lock()
        self.k = 1

    def __call__(self, x):
        return x * self.k


def test_a_value_only_its_identity_hashes_is_new_each_time():
    # ``helpers.SVC.k = 5`` changed what ``SVC(1)`` gives; its digest, of
    # its identity, did not move, and ``t = SVC(1) + 1`` was served as 2.
    service = _Service()

    assert _component(service) != _component(service)


def test_what_reads_a_value_past_the_size_runs_again(cash_magics, cash_instance, tmp_path, monkeypatch):
    import os
    import sys

    from tests._cell_driver import run_cash_cell

    name = f"_big_{os.getpid()}_{id(tmp_path)}"
    (tmp_path / f"{name}.py").write_text("BIG = {i: i for i in range(20)}\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(lineage_formula, "FOLLOWED_ITEMS", 10)
    cash_instance.config.min_execution_time_to_cache_seconds = 0.0
    cells = [f"from {name} import BIG", "t = BIG[3] + 1"]
    try:
        for _ in range(2):
            for cell in cells:
                run_cash_cell(cash_magics, cell, cells=cells)
        assert cash_magics.shell.user_ns["t"] == 4
        assert str(cash_magics._last_cell_metrics["statements"][-1]["status"]) == "COMPUTED"
    finally:
        sys.modules.pop(name, None)
