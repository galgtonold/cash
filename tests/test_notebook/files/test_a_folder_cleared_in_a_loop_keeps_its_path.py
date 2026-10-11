"""A loop clearing a folder keeps the name of the folder bound to its path.

``for old in OUT.glob('*.txt'): old.unlink()`` lists ``OUT`` among what it
changes when the simulation cannot see that ``OUT`` holds a path (after a
restart, before the cell has run). Taken for a second binding of ``OUT``,
the writers' paths no longer resolved, and every cell below re-ran the
chart cell first. Nothing changes a path in place, so only a binding counts.
"""

from __future__ import annotations

from cash.notebook.upstream._types import TraceEntry
from cash.notebook.upstream.file_writers import _literal_path_bindings


def _entry(code: str, outputs: set[str]) -> TraceEntry:
    return TraceEntry(code, outputs, set(), {}, {}, False)


def test_a_loop_changing_the_name_in_place_keeps_its_path():
    trace = [
        _entry("OUT = Path('charts')", {"OUT"}),
        _entry("for old in OUT.glob('*.txt'):\n    old.unlink()", {"OUT"}),
    ]
    assert _literal_path_bindings(trace) == {"OUT": "charts"}


def test_a_second_binding_still_leaves_the_name_out():
    trace = [
        _entry("OUT = Path('charts')", {"OUT"}),
        _entry("for OUT in dirs:\n    pass", {"OUT"}),
    ]
    assert _literal_path_bindings(trace) == {}


def test_what_can_bind_without_an_assignment_still_leaves_it_out():
    for code in ("from cfg import OUT", "exec('OUT = 1')", "globals().update(OUT=1)"):
        trace = [_entry("OUT = Path('charts')", {"OUT"}), _entry(code, {"OUT"})]
        assert _literal_path_bindings(trace) == {}, code
