"""``%time train(model)`` changes ``model``, as ``train(model)`` does.

A ``%time``/``%timeit``/``%prun`` line runs Python that may hand a variable to
a call that changes it in place. Only the receiver of a method call counted
as changed by the magic (``%time model.fit()``), so a function argument kept
the lineage of the statement that made it, and the cell below was served or
rebuilt without the change. What the magic's Python hands to a call is now
fingerprinted around it, as a plain statement's is, and a name it changed
gets the magic's lineage.
"""

from __future__ import annotations

import pytest

from cash.notebook.magic_effects import simulation_cell
from tests._cell_driver import run_cash_cell

HELPERS = "def train(m, f=10):\n" "    m['w'] = m['k'] * f\n" "def show(m):\n" "    return m['w']"
MODEL = "model = {'k': 2, 'w': None}"


def _arguments(cell: str, ns: dict):
    from cash.notebook.magic_effects import magic_call_arguments  # the name this fix adds

    return magic_call_arguments(simulation_cell(cell)[1].body[-1], ns)


@pytest.fixture
def ns():
    namespace: dict = {}
    exec(HELPERS, namespace)
    np = pytest.importorskip("numpy")
    namespace.update(model={"k": 2, "w": None}, np=np, arr=np.zeros(3), k=2)
    return namespace


@pytest.mark.parametrize(
    ("cell", "watched"),
    [
        ("%time train(model)", {"model"}),
        ("%timeit -n1 -r1 train(model)", {"model"}),
        ("%time out = train(model)", {"model"}),
        ("%time np.copyto(arr, k)", {"arr"}),
        ("%time m = np.mean(arr)", set()),
        ("files = !ls", set()),
    ],
)
def test_what_the_magics_python_hands_to_a_call_is_watched(ns, cell, watched):
    assert _arguments(cell, ns)[0] == watched


def _run_magic(processor, cell: str, run) -> None:
    """What the magics wrapper does around a cell of magics IPython runs on
    its own: fingerprints before, *run* standing in for IPython, the record after."""
    before = dict(processor.tracking_state.variable_lineage)
    snapshots = processor.magic_cell_snapshots(cell)
    run()
    processor.record_magic_cell(cell, before, snapshots)


def test_a_timed_call_that_changes_its_argument_moves_its_lineage(cash_magics, statement_processor):
    run_cash_cell(cash_magics, HELPERS)
    run_cash_cell(cash_magics, MODEL)
    ns = cash_magics.shell.user_ns
    made = statement_processor.tracking_state.variable_lineage["model"]
    _run_magic(statement_processor, "%time train(model)", lambda: ns["train"](ns["model"]))
    assert statement_processor.tracking_state.variable_lineage["model"] != made


def test_a_timed_call_that_only_reads_leaves_the_lineage(cash_magics, statement_processor):
    """Control: ``%time show(model)`` changes nothing."""
    run_cash_cell(cash_magics, HELPERS)
    run_cash_cell(cash_magics, MODEL)
    ns = cash_magics.shell.user_ns
    made = statement_processor.tracking_state.variable_lineage["model"]
    _run_magic(statement_processor, "%time show(model)", lambda: ns["show"](ns["model"]))
    assert statement_processor.tracking_state.variable_lineage["model"] == made
