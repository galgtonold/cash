"""What a statement was seen to change through the objects it met is rebuilt
after a restart, when a cell below is run alone.

``f = lst.append`` then ``f(4)``: that the call changes ``lst`` is read from
the object ``f`` holds, which is gone after a restart. Each run keeps the
variables it changed that its text does not bind (``statement_effects_key``),
and the simulation adds them to what the text says.
"""

from __future__ import annotations

import pytest

from cash.backends.file_backend import FileBackend
from cash.notebook import cache_key
from tests._cell_driver import run_cash_cell

CASES = {
    "a bound method kept in a variable": (["lst = [1, 2, 3]", "f = lst.append\nf(4)"], [1, 2, 3, 4]),
    "a bound method handed to map": (["lst = [1, 2, 3]", "list(map(lst.append, [4, 5]))"], [1, 2, 3, 4, 5]),
    "a partial": (["import functools\nlst = [1, 2, 3]", "p = functools.partial(lst.append, 4)\np()"], [1, 2, 3, 4]),
    "a method looked up by getattr": (["lst = [1, 2, 3]", "getattr(lst, 'append')(4)"], [1, 2, 3, 4]),
    "a bound method handed to a helper": (
        ["lst = [1]\ndef each(fn, xs):\n    for x in xs: fn(x)", "each(lst.append, [2, 3])"],
        [1, 2, 3],
    ),
    "a closure a factory returned": (
        ["lst = [1, 2, 3]\ndef mk():\n    def inner():\n        lst.append(4)\n    return inner", "g = mk()", "g()"],
        [1, 2, 3, 4],
    ),
    "a decorator that registers": (
        [
            "lst = []\ndef register(f):\n    lst.append(f.__name__)\n    return f",
            "@register\ndef a(): pass",
            "@register\ndef b(): pass",
        ],
        ["a", "b"],
    ),
    "a method of an object of the notebook's": (
        ["lst = []\nclass T:\n    def go(self, v):\n        lst.append(v)\n        return v\nt = T()", "r = t.go(5)"],
        [5],
    ),
}
READER = "out = list(lst)"


@pytest.fixture
def clean_backend(tmp_path):
    """A file backend: the record a later kernel reads is metadata alone,
    which only a file tier keeps."""
    backend = FileBackend(cache_dir=str(tmp_path / "cache"))
    yield backend
    backend.clear()


def _restart(magics) -> None:
    """What a kernel restart leaves: no session state, none of the notebook's names."""
    magics.tracking_state.reset_session_state()
    magics._upstream_checker.simulator.cache.reset()
    magics._upstream_checker.simulator.probe.reset()
    user_ns = magics.shell.user_ns
    for name in [n for n in user_ns if not n.startswith("_") and n not in ("In", "Out", "get_ipython", "exit", "quit")]:
        del user_ns[name]


def _record(backend, statement: str):
    from cash.notebook.effects_record import EffectsRecord

    key = cache_key.statement_effects_key(cache_key.statement_source_hash(statement))
    return EffectsRecord.from_dict(backend.get_metadata(key))


@pytest.mark.parametrize("case", sorted(CASES))
def test_a_reader_run_alone_after_a_restart_sees_the_change(cash_magics, clean_backend, case):
    above, expected = CASES[case]
    cells = [*above, READER]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert cash_magics.shell.user_ns["out"] == expected, "control: the top-to-bottom run"

    _restart(cash_magics)
    run_cash_cell(cash_magics, READER, cells=cells)

    assert cash_magics.shell.user_ns["out"] == expected


def test_a_run_keeps_what_it_changed_that_its_text_does_not_bind(cash_magics, clean_backend):
    cells = ["lst = [1]\nf = lst.append", "f(4)", "x = len(lst)", "lst.append(5)"]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)

    record = _record(clean_backend, "f(4)")
    assert record is not None and record.changed() == {"lst"}, record
    # Control: what the text says is not kept a second time.
    assert _record(clean_backend, "x = len(lst)") is None
    assert _record(clean_backend, "lst.append(5)") is None
    assert _record(clean_backend, "lst = [1]") is None


def test_the_names_are_those_of_the_run_with_the_same_inputs(cash_magics, clean_backend):
    """``f(4)`` ran with ``f`` holding one list's method, then another's:
    asked with the first inputs, the record says the first list."""
    from cash.notebook.effects_record import inputs_digest

    cells = ["a = [1]\nb = [1]", "f = a.append", "f(4)"]
    lineage = cash_magics.tracking_state.variable_lineage
    for cell in cells[:2]:
        run_cash_cell(cash_magics, cell, cells=cells)
    at_first = inputs_digest({"f": lineage["f"]})
    run_cash_cell(cash_magics, cells[2], cells=cells)
    cells[1] = "f = b.append"
    run_cash_cell(cash_magics, cells[1], cells=cells)
    at_second = inputs_digest({"f": lineage["f"]})
    run_cash_cell(cash_magics, cells[2], cells=cells)
    assert at_first != at_second

    record = _record(clean_backend, "f(4)")

    assert record.changed(at_first) == {"a"}
    assert record.changed(at_second) == {"b"}
    assert record.changed("inputs no run had") == {"a", "b"}


def test_a_record_keeps_a_few_runs_and_reads_back_as_written():
    from cash.notebook.effects_record import RUNS_KEPT, EffectsRecord

    record = EffectsRecord()
    for i in range(RUNS_KEPT + 3):
        record = record.with_run(f"run{i}", {f"v{i}"})
    record = record.with_process({"environ"})

    assert len(record.runs) == RUNS_KEPT
    assert record.changed("run0") == record.changed(), "the oldest run is gone: every name kept answers"
    assert record.changed(f"run{RUNS_KEPT + 2}") == {f"v{RUNS_KEPT + 2}"}
    assert EffectsRecord.from_dict(record.to_dict()) == record
    assert EffectsRecord.from_dict({"mutation_verdict": True}) is None
    assert EffectsRecord.from_dict(None) is None
