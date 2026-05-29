"""Regression: %cash_provenance --graph must show actual dependencies.

The StatementProcessor's metrics dict needs to carry the 'inputs' field
through to provenance/audit consumers. Without it, every ProvenanceRecord
ends up with `inputs=[]`, and the dependency graph the magic prints reads
"(no dependencies)" for every variable — even ones with clear inputs.
The bug was originally reported via the financial_analysis_demo notebook:
the user added `%cash_provenance df --graph` and got an empty graph even
though `df` had been computed from prior cells.
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock
from traitlets.config.configurable import Configurable

from cash.core import Cash
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
def magics_fixture():
    backend = InMemoryBackend()
    cash = Cash(backend=backend, register_magic=False)
    shell = MockShell()
    magics = CashMagics(shell, cash)
    yield magics, shell, backend
    backend.clear()


def test_metrics_carries_inputs_on_cache_miss(magics_fixture):
    """A fresh compute populates metrics['inputs'] with the analyzed inputs."""
    magics, shell, _ = magics_fixture
    shell.user_ns["a"] = 1
    shell.user_ns["b"] = 2
    metrics = magics._statement_processor.process_statement("c = a + b")
    assert str(metrics["status"]).upper().endswith("COMPUTED")
    assert set(metrics.get("inputs", [])) >= {"a", "b"}


def test_metrics_carries_inputs_on_second_run(magics_fixture):
    """A second run — whatever path it takes — still surfaces metrics['inputs'].

    With the trivial `a + b` example the cost-model gate may or may not cache
    (the timing is below floor). What we want to prove is that whichever
    branch fires (RESTORED or COMPUTED again), the inputs are still carried
    on the metrics dict.
    """
    from cash.notebook.annotations import CacheAnnotation
    magics, shell, _ = magics_fixture
    shell.user_ns["a"] = 1
    shell.user_ns["b"] = 2
    annotation = CacheAnnotation(persist=True)
    magics._statement_processor.process_statement("c = a + b", annotation=annotation)
    shell.user_ns.pop("c", None)
    metrics2 = magics._statement_processor.process_statement("c = a + b", annotation=annotation)
    assert set(metrics2.get("inputs", [])) >= {"a", "b"}, (
        f"inputs missing on second-run metrics; status={metrics2['status']}, "
        f"got inputs={metrics2.get('inputs')}"
    )


def test_provenance_dependency_graph_is_populated(magics_fixture):
    """End-to-end: after a compute, %cash_provenance shows a real graph."""
    magics, shell, _ = magics_fixture
    shell.user_ns["a"] = 1
    shell.user_ns["b"] = 2
    # Compute c from a and b, then drain into provenance via the magic flow.
    metrics = magics._statement_processor.process_statement("c = a + b")
    magics._record_observability([metrics])
    deps = magics._session.provenance.get_dependencies("c")
    assert "a" in deps, f"expected 'a' in dependencies of c, got {deps}"
    assert "b" in deps, f"expected 'b' in dependencies of c, got {deps}"


def test_graph_renders_as_tree_with_transitive_deps():
    """The graph block must show indentation that reflects the dep chain.

    Three-level chain: c depends on b, b depends on a. The rendered block
    should indent 'a' under 'b', not put both at the same level.
    """
    from cash.notebook.provenance import ProvenanceTracker
    pv = ProvenanceTracker()
    pv.record("a", code="a = 1", inputs=[])
    pv.record("b", code="b = a + 1", inputs=["a"])
    pv.record("c", code="c = b * 2", inputs=["b"])

    block = pv._format_graph_section("c")
    text = "\n".join(block)
    # 'b' appears as a direct child; 'a' appears underneath, more indented.
    b_idx = next(i for i, line in enumerate(block) if "─ b " in line)
    a_idx = next(i for i, line in enumerate(block) if "─ a " in line)
    assert b_idx < a_idx, f"'b' should appear before 'a' in the tree:\n{text}"
    b_indent = len(block[b_idx]) - len(block[b_idx].lstrip(" │"))
    a_indent = len(block[a_idx]) - len(block[a_idx].lstrip(" │"))
    assert a_indent > b_indent, (
        f"'a' should be more indented than 'b' (a child of b in the tree):\n{text}\n"
        f"b_indent={b_indent}, a_indent={a_indent}"
    )


def test_graph_marks_external_inputs_as_leaves():
    """Inputs without their own provenance record render as `(external)`.

    AST-extracted 'inputs' can include imports and built-ins (np, len, …)
    that aren't real upstream variables. Those should appear as leaves with
    an `(external)` tag rather than being expanded further.
    """
    from cash.notebook.provenance import ProvenanceTracker
    pv = ProvenanceTracker()
    pv.record("f", code="def f(x): return np.mean(x)", inputs=["np"])
    pv.record("y", code="y = f(data)", inputs=["f", "data"])

    block = pv._format_graph_section("y")
    text = "\n".join(block)
    assert "(external)" in text, f"expected external marker for np/data:\n{text}"
    # np must not be expanded — it has no record, so it's a leaf.
    np_lines = [line for line in block if " np" in line]
    assert any("(external)" in line for line in np_lines), \
        f"np should be marked external:\n{text}"


def test_graph_walks_union_of_history_records():
    """When a variable is mutated multiple times, the graph must show inputs
    from EVERY history record — not just the latest.

    This is the financial-demo failure: df was created via pd.read_csv (record 1),
    then mutated by `df['SMA'] = df.groupby(...)` (record 2). The latest record's
    only input is df itself (self-skip). Without walking the first record too,
    the graph shows nothing.
    """
    from cash.notebook.provenance import ProvenanceTracker
    pv = ProvenanceTracker()
    pv.record("df", code="df = pd.read_csv('x.csv')", inputs=["pd"])
    # Later: df is mutated; its inputs only list df (self-reference)
    pv.record("df", code="df['SMA'] = df.groupby('g').transform(...)", inputs=["df"])

    block = pv._format_graph_section("df")
    text = "\n".join(block)
    # The creation record had pd as an input — must surface in the graph.
    assert "pd" in text and "(external)" in text, (
        f"Expected 'pd (external)' from the creation record (record 1).\n"
        f"Got:\n{text}"
    )


def test_graph_breaks_self_reference_cycle():
    """A variable that lists itself as an input must not cause infinite recursion."""
    from cash.notebook.provenance import ProvenanceTracker
    pv = ProvenanceTracker()
    # df depends on df (the financial-demo case: df['SMA'] = df.groupby(...))
    pv.record("df", code="df = pd.read_csv('x.csv')", inputs=[])
    pv.record("df", code="df['SMA'] = df.groupby(...).transform(...)", inputs=["df"])

    block = pv._format_graph_section("df")
    text = "\n".join(block)
    # Self-ref should NOT render df under df. The graph should be empty or
    # contain only external inputs.
    assert "    └─ df " not in text and "    ├─ df " not in text, \
        f"self-reference to df should be suppressed:\n{text}"
