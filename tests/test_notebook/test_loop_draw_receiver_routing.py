"""CAS-220: in-loop draws are routed to skip-cache by receiver identity.

A control-structure body statement carries an injected marker comment, and
``process_statement`` skips method-mutation classification for such statements
wholesale -- the control structure owns its body's mutation lineage. That is
correct for ordinary receivers and wrong for a live Axes: ``ax.bar(...)`` has no
outputs, so it is cached as an ordinary no-output call and restored as a no-op
on warm runs, while the sibling ``fig.savefig(...)`` still executes because it
writes a file. Result: a byte-blank chart, with the cell printing normally.

``_identity_coupled_call_receivers`` is the narrow exemption. These tests pin
both directions -- that a draw IS caught, and that ordinary receivers are NOT,
since widening it would silently disable per-iteration loop caching.

The end-to-end bug is not reproducible under ``NotebookClient`` (see
``tests/test_notebook_integration/test_loop_draw_not_cached.py``), so the
mechanism is pinned here and the behaviour by the real-server reproducer.
"""
from __future__ import annotations

import ast

import pytest

from cash.notebook.statement import StatementProcessor


class _Shell:
    def __init__(self, ns):
        self.user_ns = ns


class _Stub:
    """Minimal stand-in: the method under test only reads ``self.shell.user_ns``."""

    def __init__(self, ns):
        self.shell = _Shell(ns)


def _receivers(code: str, ns: dict):
    return StatementProcessor._identity_coupled_call_receivers(_Stub(ns), ast.parse(code))


@pytest.fixture
def axes():
    plt = pytest.importorskip("matplotlib.pyplot")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    fig, ax = plt.subplots()
    yield fig, ax
    plt.close(fig)


def test_a_bare_draw_on_an_axes_is_caught(axes):
    """The regression: the exact statement that went missing inside the loop."""
    _fig, ax = axes
    assert _receivers("ax.bar(['a'], [1])", {'ax': ax}) == {'ax'}


def test_a_figure_receiver_is_caught(axes):
    """``fig.savefig(...)`` is identity-coupled too, per CAS-194's reasoning."""
    fig, _ax = axes
    assert _receivers("fig.savefig('x.png')", {'fig': fig}) == {'fig'}


def test_a_captured_return_draw_is_caught(axes):
    """``counts, bins, patches = ax.hist(...)`` draws AND binds (CAS-199 shape)."""
    _fig, ax = axes
    assert _receivers("counts, bins, patches = ax.hist([1, 2])", {'ax': ax}) == {'ax'}


def test_a_dataframe_receiver_is_not_caught():
    """The discriminator that keeps ordinary loops caching: ``df.head()`` is pure."""
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({'v': [1, 2, 3]})
    assert _receivers("df.head()", {'df': df}) == set()


def test_a_list_accumulator_is_not_caught():
    """``out.append(rec)`` is the commonest loop body there is; it must be untouched."""
    assert _receivers("out.append(3)", {'out': []}) == set()


def test_a_module_call_is_not_caught():
    """``plt.savefig()`` is a module function call, not a receiver draw."""
    plt = pytest.importorskip("matplotlib.pyplot")
    assert _receivers("plt.savefig('x.png')", {'plt': plt}) == set()


def test_an_unbound_receiver_is_not_caught():
    """A name absent from the namespace cannot be proven coupled; do not guess."""
    assert _receivers("ax.bar(['a'], [1])", {}) == set()


def test_no_tree_is_handled():
    assert StatementProcessor._identity_coupled_call_receivers(_Stub({}), None) == set()
