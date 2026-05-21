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
from cash.backends.backend import InMemoryBackend
from cash.notebook.magics import CashMagics


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
