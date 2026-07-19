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


@pytest.fixture(autouse=True)
def _force_agg_backend():
    """Pin matplotlib to the headless Agg backend for this module's tests.

    The two ``plt.subplots()`` tests below don't set a backend, so under ``-n16``
    matplotlib defaults to the interactive Tk backend, which cannot create a
    canvas in a parallel worker with no display and raises. Whether an earlier
    test in the same worker had already switched to Agg was pure worksteal-order
    luck; force it so the tests are isolation-robust regardless of ordering.
    """
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg", force=True)
    yield


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


def test_captured_return_draw_routes_on_identity_coupled_receiver_only(classifiers):
    """CAS-199: a draw whose return is CAPTURED into an assignment must route too.

    ``standalone_method_call_receivers`` sees only bare-``Expr`` calls, so the
    captured form (``counts, bins, _ = ax.hist(...)`` -- an ``ast.Assign``)
    slipped the CAS-194 routing and was cached, blanking the chart on a warm
    re-run. The extension keys on the RECEIVER (live Axes/Figure), never the
    statement shape: a captured pure read on an ordinary receiver still caches.
    """
    pd = pytest.importorskip("pandas")
    plt = pytest.importorskip("matplotlib.pyplot")
    proc, shell = classifiers

    fig, ax = plt.subplots()
    shell.user_ns.update({
        'df': pd.DataFrame({'x': [1, 2, 3]}),
        'ax': ax,
        'fig': fig,
        'data': [1, 2, 3, 4, 5],
        'sizes': [30, 20, 50],
    })

    # Captured-return draws on a live Axes -> mutation, whatever the return type.
    assert 'ax' in _routes_mutation(proc, shell, "counts, bins, patches = ax.hist(data)")
    assert 'ax' in _routes_mutation(proc, shell, "h = ax.hist(data, bins=11)")          # single target
    assert 'ax' in _routes_mutation(proc, shell, "wedges, texts = ax.pie(sizes)")       # sibling: pie
    assert 'ax' in _routes_mutation(proc, shell, "ml, sl, bl = ax.stem(data)")          # sibling: stem
    # Nested in a larger RHS expression is still caught (whole RHS is walked).
    assert 'ax' in _routes_mutation(proc, shell, "n = int((ax.hist(data)[0] > 0).sum())")

    # No over-invalidation: a captured pure read on an ORDINARY (non-Axes)
    # receiver must NOT route -- the discriminator is the receiver, not the shape.
    assert 'df' not in _routes_mutation(proc, shell, "m = df.mean()")
    assert 'df' not in _routes_mutation(proc, shell, "s = df.describe()")
    assert 'df' not in _routes_mutation(proc, shell, "top = df.head()")

    plt.close(fig)


def test_call_expression_receiver_is_not_attributed_to_the_callee(classifiers):
    """CAS-210: ``open(p, 'a').write(x)`` must not record a mutation of ``open``.

    The receiver of ``.write`` is a Call, not a name. Resolving it walked through
    the Call to the CALLEE and returned ``open`` -- but the callee is not the
    receiver: the call builds a NEW object that no variable is bound to, so there
    is no receiver lineage to bump. Booking it as a mutation of ``open`` made the
    writer statement re-execute during upstream reconstruction, and because the
    write is a ``mode='a'`` append, re-execution DUPLICATED the line on disk.

    This pins the misattribution only. Whether a non-idempotent write may be
    re-fired at all is the separate defence-in-depth half of CAS-210.
    """
    proc, shell = classifiers
    shell.user_ns.update({
        'p': 'audit.log',
        'payload': 42,
        'groups': {},
        'key': 'k',
        'val': 1,
    })

    # A constructor/factory call as the receiver has NO variable to mutate.
    assert 'open' not in _routes_mutation(proc, shell, "open(p, 'a').write('x\\n')")
    assert 'open' not in _routes_mutation(proc, shell, "open(p, 'w').writelines(['x'])")
    assert 'Path' not in _routes_mutation(proc, shell, "Path(p).write_text('x')")

    # The documented chained-call intent still resolves to the real variable:
    # this descends through the Attribute branch, not the callee branch.
    assert 'groups' in _routes_mutation(proc, shell, "groups.setdefault(key, []).append(val)")

    # And an ordinary named receiver is untouched by the guard.
    assert 'groups' in _routes_mutation(proc, shell, "groups.update({'a': 1})")
