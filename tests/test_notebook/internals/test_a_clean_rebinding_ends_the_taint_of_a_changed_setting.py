"""A name bound anew from clean inputs is not stale for a setting changed below.

``np.set_printoptions(precision=6)`` in a cell below, run on an earlier pass
and followed there by ``import numpy as np``, leaves ``np`` with another
lineage than the cells above give it. The check before a cell took what was
built from ``np`` above for stale and followed that through every statement
after, even past one that binds the name anew from nothing ``np`` touched
(``tfidf = TfidfVectorizer(...)`` after a demo ``tfidf =
TfidfTransformer(...)``). Running the cells below an edit then built the
pipeline and the grid search on that ``tfidf`` again.
"""

from __future__ import annotations

from cash.notebook.tracking_state import TrackingState
from cash.notebook.upstream._types import TraceEntry
from cash.notebook.upstream.unsaved_edits import UnsavedEdits

IMPORT = "import numpy as np"


def _tainted(*statements: tuple[str, set[str], set[str]]) -> set[str]:
    state = TrackingState()
    state.lineage.record("np", "np-as-the-cell-below-left-it")
    state.executed_cell_codes["np"] = IMPORT
    trace = [TraceEntry(code, outputs, inputs, {}, {}, False) for code, outputs, inputs in statements]
    return UnsavedEdits(state, statements=None).compute_tainted_vars_from_unsaved_edits(
        {"np": "np-as-the-cells-above-leave-it"},
        trace,
        {entry.stmt_code for entry in trace},
        current_cell_idx=len(trace),
        notebook_cells=[code for code, _, _ in statements],
    )


def test_a_clean_rebinding_ends_the_taint():
    tainted = _tainted(
        (IMPORT, {"np"}, set()),
        ("tfidf = np.array([0.5])", {"tfidf"}, {"np"}),
        ("tfidf = TfidfVectorizer()", {"tfidf"}, {"TfidfVectorizer"}),
        ("pipe = Pipeline(tfidf)", {"pipe"}, {"Pipeline", "tfidf"}),
    )
    assert tainted == set()


def test_a_rebinding_that_reads_a_tainted_name_passes_it_on():
    tainted = _tainted(
        (IMPORT, {"np"}, set()),
        ("tfidf = np.array([0.5])", {"tfidf"}, {"np"}),
        ("tfidf = list(tfidf)", {"tfidf"}, {"tfidf"}),
        ("pipe = Pipeline(tfidf)", {"pipe"}, {"Pipeline", "tfidf"}),
    )
    assert tainted == {"tfidf", "pipe"}


def test_a_change_in_place_does_not_end_the_taint():
    tainted = _tainted(
        (IMPORT, {"np"}, set()),
        ("tfidf = np.array([0.5])", {"tfidf"}, {"np"}),
        ("fill(tfidf)", {"tfidf"}, {"fill"}),
        ("pipe = Pipeline(tfidf)", {"pipe"}, {"Pipeline", "tfidf"}),
    )
    assert tainted == {"tfidf", "pipe"}
