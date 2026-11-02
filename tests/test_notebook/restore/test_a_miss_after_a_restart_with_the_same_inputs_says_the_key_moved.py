"""A statement stored before under another key, though every input it
reads has the lineage it had then -- its key moved for something else it
reads -- says so as its miss reason, where it said nothing:
``grid_search.fit(...)`` ran again after a restart for a quarter of a
minute with no reason given."""

from __future__ import annotations

from cash.notebook.statement.input_change import stored_input_change_reason
from cash.tracking.randomness import rng_virtual_var

NP = rng_virtual_var("numpy.random")


class _Backend:
    """One version kept, built from *input_lineages* and keyed on the random
    streams *rng_reads*."""

    def __init__(self, input_lineages, rng_reads=None):
        self._metadata = {"input_lineages": input_lineages}
        if rng_reads is not None:
            self._metadata["rng_reads"] = rng_reads

    def previous_version(self, slot, besides):
        return "stmt:before"

    def peek_metadata(self, key):
        return self._metadata if key == "stmt:before" else None


def _reason(was, now, rng_reads=None):
    return stored_input_change_reason(_Backend(was, rng_reads), "stmt:now", "src", ["X", "y"], ["model"], now)


def test_the_same_inputs_name_something_else_it_reads():
    reason = _reason({"X": "a", "y": "b"}, {"X": "a", "y": "b"})

    assert reason is not None
    assert reason.startswith("stored before under another key, from the same inputs")


def test_an_input_that_moved_is_named_instead():
    assert _reason({"X": "a", "y": "b"}, {"X": "a", "y": "c"}) == "input changed since it was last stored: y"


def test_nothing_is_said_without_a_version_to_compare():
    assert _reason({}, {"X": "a", "y": "b"}) is None


def test_a_random_stream_seeded_since_is_named():
    """``gs.fit(X, y)`` stored from the stream a seed below left on a second
    Run All, then run again after a restart, where it draws unseeded."""
    was = {"X": "a", "y": "b"}
    assert (
        _reason(was, {"X": "a", "y": "b"}, {NP: "seeded"})
        == "random state changed since it was last stored: numpy.random (seeded then, unseeded now)"
    )
    assert (
        _reason(was, {"X": "a", "y": "b", NP: "seeded"}, {NP: ""})
        == "random state changed since it was last stored: numpy.random (unseeded then, seeded now)"
    )
    assert _reason(was, {"X": "a", "y": "b", NP: "two"}, {NP: "one"}) == (
        "random state changed since it was last stored: numpy.random"
    )


def test_a_random_stream_where_it_was_names_something_else():
    reason = _reason({"X": "a", "y": "b"}, {"X": "a", "y": "b", NP: "one"}, {NP: "one"})

    assert reason is not None
    assert reason.startswith("stored before under another key, from the same inputs")
