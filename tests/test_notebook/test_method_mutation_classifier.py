"""Receiver-mutation classifier: pin both CAS-194 and CAS-196 directions.

``_classify_method_mutations`` (runtime) and ``_mutation_receivers`` (simulation)
must agree, and must:

* CAS-194 — treat any method call on a live matplotlib Axes/Figure as an
  in-place draw (``ax.hist()`` returns a data tuple but mutates the Axes), and
* CAS-196 — NOT treat ``df.to_csv(path)`` / ``fig.savefig(path)`` as a mutation
  of the receiver (they read it and write a file),

without loosening the genuine cases (``lst.append``, ``df.sort_values(inplace=
True)`` still mutate; ``df.head()`` still does not).
"""
import ast
import hashlib

import pytest
from unittest.mock import MagicMock
from traitlets.config import Configurable

from cash import Cash
from cash.backends import InMemoryBackend
from cash.notebook.ipython.magics import CashMagics


class MockShell(Configurable):
    def __init__(self):
        super().__init__()
        self.user_ns = {}
        self.input_transformers_cleanup = []
        self.run_cell = MagicMock()
        self.events = MagicMock()
        self.ast_transformers = []
        self.user_global_ns = self.user_ns


@pytest.fixture
def classifiers():
    backend = InMemoryBackend()
    cash = Cash(backend=backend, register_magic=False)
    shell = MockShell()
    magics = CashMagics(shell, cash)
    proc = magics._statement_processor
    yield proc, shell
    backend.clear()
    shell.user_ns.clear()


def _routes_mutation(proc, shell, code):
    """True if *code*'s receiver is routed as an in-place mutation."""
    tree = ast.parse(code)
    h = hashlib.sha256(code.encode()).hexdigest()
    pre_route, observe, assumed, _ = proc._classify_method_mutations(tree, h, set())
    return pre_route | assumed


def test_axes_hist_is_a_mutation_but_dataframe_writes_are_not(classifiers):
    pd = pytest.importorskip("pandas")
    plt = pytest.importorskip("matplotlib.pyplot")
    proc, shell = classifiers

    fig, ax = plt.subplots()
    shell.user_ns.update({
        'df': pd.DataFrame({'x': [1, 2, 3]}),
        'ax': ax,
        'fig': fig,
        'lst': [1, 2, 3],
        'data': [1, 2, 3, 4, 5],
    })

    # CAS-194: ax.hist(...) draws on the Axes -> mutation, despite the data tuple.
    assert 'ax' in _routes_mutation(proc, shell, "ax.hist(data)")
    assert 'ax' in _routes_mutation(proc, shell, "ax.plot(data)")

    # CAS-196: df.to_csv reads the frame and writes a file -> NOT a receiver
    # mutation, so it never bumps df's lineage and cannot become a spurious
    # producer of df that re-fires the write during reconstruction.
    assert 'df' not in _routes_mutation(proc, shell, "df.to_csv('out.csv')")
    assert 'df' not in _routes_mutation(proc, shell, "df.to_parquet('out.pq')")

    # fig.savefig() is DELIBERATELY still a mutation: fig is identity-coupled
    # (never cached, re-derived as a unit) so bumping it is idempotent, and the
    # savefig->fig edge is load-bearing for CAS-175 chart-coherence re-derivation.
    assert 'fig' in _routes_mutation(proc, shell, "fig.savefig('out.png')")

    # Controls — genuine mutations must STILL route (no under-invalidation).
    assert 'lst' in _routes_mutation(proc, shell, "lst.append(4)")
    assert 'df' in _routes_mutation(proc, shell, "df.sort_values('x', inplace=True)")

    # Control — a genuine receiver-pure read must STILL not route (no over-invalidation).
    assert 'df' not in _routes_mutation(proc, shell, "df.head()")

    plt.close(fig)
