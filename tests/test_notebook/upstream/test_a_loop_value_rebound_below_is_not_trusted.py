"""Which statement above last binds a loop-derived name memory took from an
earlier one (``_rebound_below``): only a later fresh binding moves it."""

from __future__ import annotations

from cash.notebook.upstream._types import TraceEntry
from cash.notebook.upstream.mismatch_classifier import _rebound_below

LOOP = "for i in range(3):\n    rows.append(i)"
BUILD = "df = frame(rows)"


def _entry(code: str, outputs: set[str], inputs: set[str]) -> TraceEntry:
    return TraceEntry(code, outputs, inputs, {}, {}, False)


def test_a_fresh_binding_below_rebinds_it():
    trace = [_entry(BUILD, {"df"}, {"frame", "rows"}), _entry("df = example()", {"df"}, {"example"})]
    assert _rebound_below("df", BUILD, trace)


def test_the_last_binding_being_its_own_does_not():
    trace = [_entry("df = example()", {"df"}, {"example"}), _entry(BUILD, {"df"}, {"frame", "rows"})]
    assert not _rebound_below("df", BUILD, trace)


def test_a_statement_below_reading_it_does_not():
    trace = [_entry(BUILD, {"df"}, {"frame", "rows"}), _entry("df = df.dropna()", {"df"}, {"df"})]
    assert not _rebound_below("df", BUILD, trace)


def test_a_body_statement_of_the_loop_counts_as_the_loop():
    trace = [_entry("rows = []", {"rows"}, set()), _entry(LOOP, {"rows", "i"}, {"rows", "range"})]
    assert not _rebound_below("rows", "rows.append(i)", trace)
