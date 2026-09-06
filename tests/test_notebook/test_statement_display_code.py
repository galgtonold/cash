"""The metrics dict carries a display-only copy of the statement's source."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from traitlets.config import Configurable

from cash import Cash
from cash.backends import InMemoryBackend
from cash.notebook.annotations import CacheAnnotation
from cash.notebook.cache_status import CacheStatus
from cash.notebook.ipython.magics import CashMagics

pytest.importorskip("IPython")

# Force caching regardless of the 10 ms min-execution-time floor -- same
# convention as test_already_executed_optimization.py.
_PERSIST = CacheAnnotation(persist=True)


class MockShell(Configurable):
    """Mock IPython shell for testing."""
    def __init__(self):
        super().__init__()
        self.user_ns = {}
        self.input_transformers_cleanup = []
        self.run_cell = MagicMock()
        self.events = MagicMock()
        self.ast_transformers = []
        self.user_global_ns = self.user_ns
        self.display_pub = type('MockDisplayPub', (), {'publish': MagicMock()})()


@pytest.fixture
def processor_fixture():
    """Provide StatementProcessor instance for testing."""
    backend = InMemoryBackend()
    cash = Cash(backend=backend, register_magic=False)
    shell = MockShell()
    magics = CashMagics(shell, cash)
    processor = magics._statement_processor
    yield processor, shell, backend, magics
    backend.clear()
    shell.user_ns.clear()


def test_display_code_is_recorded_when_supplied(processor_fixture):
    processor, shell, backend, magics = processor_fixture
    shell.user_ns["a"] = 1

    metrics = processor.process_statement(
        "x = a + 1", display_code="x = (\n    a + 1\n)",
    )

    assert metrics["display_code"] == "x = (\n    a + 1\n)"
    assert metrics["code"] == "x = a + 1", "the keyed text must not change"


def test_display_code_defaults_to_none(processor_fixture):
    """An older caller that does not pass it must behave exactly as before."""
    processor, shell, backend, magics = processor_fixture
    shell.user_ns["a"] = 1

    metrics = processor.process_statement("x = a + 1")

    assert metrics.get("display_code") is None


def test_display_code_does_not_change_the_cache_key(processor_fixture):
    """The whole point: it is display-only.

    Same code, different display text -- the second call must still hit.

    Two fixture gates have nothing to do with display_code but block a hit
    on their own, so both must be cleared for this test to mean anything:
    a raw ``shell.user_ns[...] =`` write (as used by the other two tests in
    this file, and by the sibling fixture files) leaves the variable without
    cash-tracked lineage, which alone makes ANY statement reading it
    uncacheable -- so 'a' is established via process_statement instead.
    And ``x = a + 1`` runs in well under a millisecond, under the 10ms
    "too cheap to cache" floor -- so both calls force persistence.
    """
    processor, shell, backend, magics = processor_fixture
    processor.process_statement("a = 1")

    first = processor.process_statement("x = a + 1", display_code="x = a + 1", annotation=_PERSIST)
    second = processor.process_statement(
        "x = a + 1", display_code="x = (\n    a + 1\n)", annotation=_PERSIST,
    )

    assert first["status"] == CacheStatus.COMPUTED
    assert second["status"] in (CacheStatus.SKIPPED, CacheStatus.RESTORED), (
        f"display text changed the key: second call was {second['status']}"
    )


def test_a_control_body_statement_has_no_display_code(processor_fixture):
    """Capture is top-level only, by design.

    A control body, a loop-split iteration and a statement cash rewrote all
    keep showing what actually RAN. Their source is not what executed, so
    showing it would mislead rather than help.

    This drives a REAL ``for`` loop through the full ``%%cash`` pipeline
    (``CashMagics.cash`` -> ``CellExecutor`` -> ``ControlStructureProcessor``)
    rather than calling ``process_statement`` directly, because the thing
    being pinned is a fact about WIRING, not about ``process_statement``
    itself: ``process_statement`` happily accepts a ``display_code`` kwarg
    from any caller. What actually keeps a loop body's row ``None`` is that
    ``for_handler.py``'s ``_execute_loop_body`` calls
    ``self.statement_processor.process_statement(modified_code, ttl, silent,
    annotation=annotation)`` with no ``display_code`` kwarg at all -- for
    every loop-body statement, every iteration. Only the top-level dispatch
    loop in ``cell_executor.py`` computes ``_statement_source`` and threads
    it through. A test that calls ``process_statement("y = 2")`` directly
    (as this one used to) never touches that wiring at all: it would keep
    passing even if someone threaded ``display_code`` all the way into
    ``for_handler.py`` / ``if_handler.py`` / ``try_handler.py`` tomorrow.

    Scope note: this only drives a ``for`` loop (the control structure
    ``for_handler.py`` implements), not ``if``/``try``. The three handlers
    share the same ``process_statement(...)`` call shape with no
    ``display_code`` kwarg (verified by reading each), so the same argument
    applies to all three, but this test does not itself exercise ``if``/
    ``try`` bodies.
    """
    processor, shell, backend, magics = processor_fixture
    magics._badge_mode = 'html'
    shell.user_ns['xs'] = [1, 2, 3]
    cell = (
        "for i in xs:\n"
        "    y = i + 1\n"
        "z = 99\n"
    )

    with patch.object(magics, '_render_interactive_badge') as mock_badge:
        magics.cash("", cell)

    # Premises: both the loop body and the sibling statement actually ran,
    # so a false pass can't hide behind a cell that silently did nothing.
    # `y` is overwritten each iteration, so it holds the LAST one (i=3).
    assert shell.user_ns.get('y') == 4
    assert shell.user_ns.get('z') == 99

    all_metrics = mock_badge.call_args_list[-1][0][0]
    loop_rows = [m for m in all_metrics if 'loop_vars' in m]
    sibling_rows = [m for m in all_metrics if m.get('code') == 'z = 99']

    assert loop_rows, "premise: the loop produced its own per-iteration rows"
    assert len(loop_rows) == 3, "premise: one row per iteration (3 elements in xs)"
    assert all(row.get('display_code') is None for row in loop_rows)

    assert sibling_rows, "premise: the sibling top-level statement has its own row"
    assert all(row.get('display_code') is not None for row in sibling_rows)
