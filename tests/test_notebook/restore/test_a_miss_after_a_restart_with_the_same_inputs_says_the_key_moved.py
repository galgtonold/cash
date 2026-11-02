"""A statement stored before under another key, though every input it
reads has the lineage it had then -- its key moved for something else it
reads -- says so as its miss reason, where it said nothing:
``grid_search.fit(...)`` ran again after a restart for a quarter of a
minute with no reason given."""

from __future__ import annotations

from cash.notebook.statement.input_change import stored_input_change_reason


class _Backend:
    """One version kept, built from *input_lineages*."""

    def __init__(self, input_lineages):
        self._metadata = {"input_lineages": input_lineages}

    def previous_version(self, slot, besides):
        return "stmt:before"

    def peek_metadata(self, key):
        return self._metadata if key == "stmt:before" else None


def _reason(was, now):
    return stored_input_change_reason(_Backend(was), "stmt:now", "src", ["X", "y"], ["model"], now)


def test_the_same_inputs_name_something_else_it_reads():
    reason = _reason({"X": "a", "y": "b"}, {"X": "a", "y": "b"})

    assert reason is not None
    assert reason.startswith("stored before under another key, from the same inputs")


def test_an_input_that_moved_is_named_instead():
    assert _reason({"X": "a", "y": "b"}, {"X": "a", "y": "c"}) == "input changed since it was last stored: y"


def test_nothing_is_said_without_a_version_to_compare():
    assert _reason({}, {"X": "a", "y": "b"}) is None
