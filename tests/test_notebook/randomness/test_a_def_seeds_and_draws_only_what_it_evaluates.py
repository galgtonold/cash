"""A ``def`` runs only its decorators and defaults; its body runs when the
function is called, and the statement calling it seeds or draws there.

Read whole, ``@functools.cache def cities(n): random.seed(n); ...`` was a
seed and a draw: it gave the stream a lineage of its own, a later check keyed
it on that very lineage (and so had it set another), and every ``def`` whose
body draws carried the stream's lineage where it was defined, so a seed set
by a call below moved it and the definition was run again.
"""

from __future__ import annotations

from cash.notebook.lineage_formula import UNSEEDED, take_rng_reads_at_last_draw
from cash.notebook.tracking_state import TrackingState
from cash.source_norm import exact_source_digest
from cash.tracking.randomness import (
    drawn_rng_vars,
    hidden_lineage_reads,
    hidden_lineage_writes,
    rng_virtual_var,
)
from tests._cell_driver import run_cash_cell

RANDOM = rng_virtual_var("random")
MEMO_HELPER = "@functools.cache\ndef cities(n):\n    random.seed(n)\n    return random.random()"


def test_a_def_neither_reads_nor_writes_the_stream_its_body_uses():
    assert hidden_lineage_reads(MEMO_HELPER) == set()
    assert hidden_lineage_writes(MEMO_HELPER) == set()
    assert hidden_lineage_reads("def rep(c):\n    return random.sample(c, len(c))") == set()


def test_what_a_def_evaluates_is_read_whole():
    """Positive control: a default or a decorator runs as the def does, and
    a class body runs where the class is defined."""
    assert hidden_lineage_reads("def f(x=random.random()):\n    return x") == {RANDOM}
    assert hidden_lineage_writes("@pick(random.seed(0))\ndef f():\n    return 1") == {RANDOM}
    assert hidden_lineage_reads("class C:\n    x = random.random()") == {RANDOM}
    assert hidden_lineage_writes("random.seed(0)") == {RANDOM}
    assert hidden_lineage_reads("x = cities(3) + random.random()") == {RANDOM}


def test_defining_a_seeding_helper_gives_the_stream_no_lineage(cash_magics):
    cells = ["import functools, random", MEMO_HELPER]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert RANDOM not in cash_magics.tracking_state.variable_lineage
    # Control: calling it seeds the stream, which takes a lineage then.
    cells.append("v = cities(3)")
    run_cash_cell(cash_magics, cells[-1], cells=cells)
    assert cash_magics.tracking_state.variable_lineage.get(RANDOM)


def test_a_stream_a_statement_drew_unseeded_is_simulated_unseeded():
    """The simulation keys a draw that found the stream unseeded without a
    lineage for it, not with the one the kernel's stream has now (which a
    seed inside a helper the draw called gave it)."""
    state = TrackingState()
    code = "a = sum(cities(5))"
    state.rng_unseeded_at_last_draw[exact_source_digest(code)] = frozenset({RANDOM})
    state.lineage.record(RANDOM, "set-by-its-own-seed")
    virtual: dict[str, str] = {}
    take_rng_reads_at_last_draw(state, code, {RANDOM}, virtual)
    assert virtual == {RANDOM: UNSEEDED}
    assert drawn_rng_vars({RANDOM}, code, virtual) == set()
    # Control: a lineage the simulation set above is kept, and moves on.
    virtual = {RANDOM: "seeded-above"}
    take_rng_reads_at_last_draw(state, code, {RANDOM}, virtual)
    assert virtual == {RANDOM: "seeded-above"}
    assert drawn_rng_vars({RANDOM}, code, virtual) == {RANDOM}
