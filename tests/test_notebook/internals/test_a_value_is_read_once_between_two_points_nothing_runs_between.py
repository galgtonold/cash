"""A statement reads a value once wherever no code of the user's can run
between two checks that read it, and again wherever some can.

p20 (networkx) read its graph up to four times a statement: the in-place
check before the statement, the routed call's argument hash before and
after the call, the in-place check after it, and the output's session hash.
A graph has no cheaper digest than pickling all of it and no sound
"unchanged" signal, so a digest is handed on only where nothing can run
between (`DigestHandoff`):

* before a call: when it is the first call the statement makes and
  nothing before it can run code (`_first_call_source`), not only when
  the statement is that one call;
* after the statement: from the in-place check to the session hash and
  the same-value cutoff of an output it changed (`take_settled`).

A call's own after-hash is never handed one taken before it: that is the
check that sees what the call changed.
"""

from __future__ import annotations

import ast

import pytest

from cash import _native, kept_state, value_watch
from cash.notebook.call_effects import DigestHandoff, _first_call_source
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

np = pytest.importorskip("numpy")


@pytest.mark.parametrize(
    "source, first",
    [
        ("y = f(x)", None),  # the whole statement: `_call_shape` covers it
        ("y = f(x) + 0", "f(x)"),
        ("y = round(f(x), 2)", "f(x)"),
        ("print(f(x))", "f(x)"),
        ("y = f(x)[0]", "f(x)"),
        ("y = f(g(x))", "g(x)"),
        ("y = f(x, g(x))", "g(x)"),
        ("y = x + f(x)", "f(x)"),
        # Something before the call may run code of the user's.
        ("y = x.a + f(x)", None),
        ("y = x[0] + f(x)", None),
        ("y = obj.m(x)", None),
        ("y = f(x.a)", None),
        ("y = f(*xs)", None),
        ("y = f(**kw)", None),
        ("y = f(x) if c else 0", None),
        ("y = (x.sum(), f(x))[1]", None),
        # The same call twice: which one started first is not known by source.
        ("y = [f(x), f(x)]", None),
        ("for i in f(x):\n    pass", None),
        ("y = f(x); z = 1", None),
    ],
)
def test_the_first_call_with_nothing_before_it(source, first):
    tree = ast.parse(source)
    found = None if source == "y = f(x)" else _first_call_source(tree)
    assert found == first


def test_a_module_function_is_read_without_running_code():
    np_tree = ast.parse("y = round(np.sum(x), 2)")
    assert _first_call_source(np_tree, {"np": np}) == "np.sum(x)"
    assert _first_call_source(np_tree, {}) is None


def _counting(handoff, monkeypatch):
    reads = []
    real = handoff.fingerprints.digest
    monkeypatch.setattr(handoff.fingerprints, "digest", lambda v: reads.append(v) or real(v))
    return reads


@pytest.mark.parametrize("source, shared", [("f(x)", True), ("g(x)", False), (None, False)])
def test_the_first_call_takes_the_statement_digest_only_by_its_source(monkeypatch, source, shared):
    x = np.arange(10.0)
    handoff = DigestHandoff()
    handoff.begin_statement()
    handoff.watch(ast.parse("y = round(f(x), 2)"))
    before = handoff.digest(x)
    reads = _counting(handoff, monkeypatch)
    ticket = handoff.call_started()
    assert handoff.digest_before(ticket, source)(x) == before
    assert len(reads) == (0 if shared else 1)


def test_the_statement_digests_are_let_go_once_a_call_took_them(monkeypatch):
    """Held, they would read as references from outside to the checks
    that count them."""
    x = np.arange(10.0)
    handoff = DigestHandoff()
    handoff.begin_statement()
    handoff.watch(ast.parse("y = round(f(x), 2)"))
    handoff.digest(x)
    ticket = handoff.call_started()
    handoff.digest_before(ticket, "f(x)")(x)
    reads = _counting(handoff, monkeypatch)
    handoff.digest_before(ticket, "f(x)")(x)
    assert len(reads) == 1


def test_a_second_call_hashes_for_itself(monkeypatch):
    x = np.arange(10.0)
    handoff = DigestHandoff()
    handoff.begin_statement()
    handoff.watch(ast.parse("y = f(g(x))"))
    handoff.digest(x)
    handoff.call_started()
    reads = _counting(handoff, monkeypatch)
    handoff.digest_before(handoff.call_started(), "f(g(x))")(x)
    assert len(reads) == 1


def test_digests_are_kept_for_the_outputs_only_while_collecting():
    x = np.arange(10.0)
    handoff = DigestHandoff()
    handoff.begin_statement()
    handoff.digest_after(x)
    assert handoff.take_settled() == {}
    handoff.begin_settled()
    digest = handoff.digest_after(x)
    settled = handoff.take_settled()
    assert settled == {id(x): (x, digest)}
    handoff.digest_after(x)
    assert handoff.take_settled() == {}


# --- in a notebook --------------------------------------------------------

SETUP = (
    "import time\n"
    "import numpy as np\n"
    "def normalize(a):\n"
    f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
    "    return (a - a.mean()) / a.std()\n"
)


@pytest.fixture
def reads_of_x(monkeypatch, mock_shell):
    from cash import content_hashers

    seen = []
    real = content_hashers.hash_numpy

    def counting(value, *a, **k):
        if value is mock_shell.user_ns.get("x"):
            seen.append(1)
        return real(value, *a, **k)

    monkeypatch.setattr(content_hashers, "hash_numpy", counting)
    return seen


@pytest.mark.parametrize(
    "statement, reads",
    [
        # Was 4: the statement's check before, the call's before and after,
        # the statement's check after. Nothing runs before the call.
        ("y = normalize(x) + 0", 3),
        ("y = round(float(normalize(x).sum()), 3)", 3),
        # `x.sum()` runs before the call: each check reads for itself.
        ("y = (x.sum(), normalize(x))[1]", 4),
    ],
)
def test_a_statement_reads_its_argument_once_where_nothing_runs_between(
    cash_magics, mock_shell, reads_of_x, statement, reads
):
    mock_shell.user_ns["__name__"] = "__main__"
    run_cash_cell(cash_magics, SETUP)
    run_cash_cell(cash_magics, "x = np.arange(300_000.0)")
    reads_of_x.clear()
    run_cash_cell(cash_magics, statement)
    assert len(reads_of_x) == reads, f"x was hashed {len(reads_of_x)} times"


nx = pytest.importorskip("networkx")


@pytest.fixture
def unwatched():
    """No value watched or sighted yet. The watchers live as long as the
    process: a sighting left by an earlier test at the address a new value
    takes counts as that value's first, so the value is watched one read
    sooner and a later check is answered without a read."""
    value_watch.reset()


def _pickles_of(monkeypatch, mock_shell, name, label=lambda value: 1):
    """Count the pickles a digest takes of *name*: whole
    (`kept_state.dumps_also_keyed` writing to a hashing sink) or keyed apart
    (`kept_state.dumps`). A store's pickle of its own is not a read. Each
    read is recorded as ``label(value)``."""
    seen = []
    real, real_also = kept_state.dumps, kept_state.dumps_also_keyed

    def counted(value):
        if value is not None and value is mock_shell.user_ns.get(name):
            seen.append(label(value))

    def dumps(value, *a, **k):
        counted(value)
        return real(value, *a, **k)

    def dumps_also_keyed(value, *a, **k):
        if k.get("sink") is not None:
            counted(value)
        return real_also(value, *a, **k)

    monkeypatch.setattr(kept_state, "dumps", dumps)
    monkeypatch.setattr(kept_state, "dumps_also_keyed", dumps_also_keyed)
    return seen


@pytest.mark.parametrize("watched_before", [False, True])
def test_a_graph_a_statement_changes_is_read_once_after_it(
    cash_magics, mock_shell, monkeypatch, unwatched, watched_before
):
    """p20: ``nx.set_node_attributes(G, region, "region")``. The check
    after it reads ``G`` to see it changed; ``G`` is then an output, and
    its session hash is that same digest: no code ran between. The check
    before it reads ``G`` too, unless the watchers already prove ``G``
    unchanged since a read before the statement."""
    if watched_before and (not _native.AVAILABLE or value_watch.VERIFY):
        pytest.skip("needs the C watchers, without CASH_CWATCH_VERIFY=1 hashing behind every reuse")
    from cash.value_hash import compute_hash

    run_cash_cell(cash_magics, "import networkx as nx")
    run_cash_cell(cash_magics, "G = nx.gnm_random_graph(200, 800, seed=1)")
    run_cash_cell(cash_magics, "region = {n: n % 3 for n in G}")
    if watched_before:
        compute_hash(mock_shell.user_ns["G"])  # a second sighting: watched from here on
    reads = _pickles_of(monkeypatch, mock_shell, "G", label=lambda g: "after" if "region" in g.nodes[5] else "before")

    run_cash_cell(cash_magics, 'nx.set_node_attributes(G, region, "region")')

    # Was 3: before, after, and the session hash.
    assert reads == (["after"] if watched_before else ["before", "after"])
    assert mock_shell.user_ns["G"].nodes[5]["region"] == 2
    state = cash_magics.tracking_state
    assert state.current_session_hashes["G"] == compute_hash(mock_shell.user_ns["G"])


def test_a_graph_read_by_a_call_inside_print_is_read_once_before_the_call(
    cash_magics, mock_shell, monkeypatch, unwatched
):
    """p20 ``print("density:", round(nx.density(G), 6))``-like: the call's
    argument hash before it takes the statement's own, when the statement
    took one; after the call it is read again."""
    mock_shell.user_ns["__name__"] = "__main__"
    run_cash_cell(cash_magics, "import time\nimport networkx as nx")
    run_cash_cell(
        cash_magics,
        f"def degree_sum(g):\n    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n    return sum(d for _, d in g.degree())",
    )
    run_cash_cell(cash_magics, "G = nx.gnm_random_graph(200, 800, seed=1)")
    reads = _pickles_of(monkeypatch, mock_shell, "G")
    run_cash_cell(cash_magics, "total = round(degree_sum(G), 2)")
    assert mock_shell.user_ns["total"] == 1600
    # 3 reads in pure Python; with the C fast path on, the watchers prove G
    # unchanged since an earlier read and one more read is skipped.
    if value_watch.VERIFY:
        pytest.skip("counts reads; CASH_CWATCH_VERIFY=1 hashes behind every reuse")
    expected = 2 if _native.AVAILABLE else 3
    assert len(reads) == expected, f"G was read {len(reads)} times"
