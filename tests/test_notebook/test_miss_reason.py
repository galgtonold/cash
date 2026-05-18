"""Tests for the cache-miss attribution feature.

When a cell is COMPUTED (cache miss), the runtime should stamp a one-line
``miss_reason`` on the metric dict so the badge can answer "why did this
cell re-run?" in its row-detail drawer.

See ``StatementProcessor._diagnose_miss`` and the per-invalidator
``self._last_miss_reason`` writes in ``statement_processor.py``.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from traitlets.config import Configurable

from cash.backends.backend import InMemoryBackend
from cash.core import Cash
from cash.notebook.cache_status import CacheStatus
from cash.notebook.magics import CashMagics


class _MockShell(Configurable):
    def __init__(self):
        super().__init__()
        self.user_ns = {}
        self.input_transformers_cleanup = []
        self.run_cell = MagicMock()
        self.events = MagicMock()
        self.ast_transformers = []
        self.user_global_ns = self.user_ns
        self.display_pub = type("MockDisplayPub", (), {"publish": MagicMock()})()


@pytest.fixture
def magics_fixture():
    backend = InMemoryBackend()
    cash = Cash(backend=backend, register_magic=False)
    shell = _MockShell()
    magics = CashMagics(shell, cash)
    magics._auto_cache_enabled = True
    yield magics, shell, backend, cash
    backend.clear()
    shell.user_ns.clear()


def _last_metric(shell, magics, code: str) -> dict:
    """Run one cell and return the last metric the processor recorded."""
    captured: list[dict] = []
    real_render = magics._render_interactive_badge

    def capture(metrics, **kw):
        captured.append(list(metrics))
        return real_render(metrics, **kw)

    magics._render_interactive_badge = capture  # type: ignore[assignment]
    try:
        magics.cash("", code)
    finally:
        magics._render_interactive_badge = real_render  # type: ignore[assignment]
    assert captured, "no metrics captured"
    return captured[-1][-1]


class TestMissReasonAttribution:
    """End-to-end: the metric dict on a COMPUTED first-run carries a reason."""

    def test_first_run_attributes_first_time_seeing_this_code(self, magics_fixture):
        magics, shell, _backend, _cash = magics_fixture
        m = _last_metric(shell, magics, "x = 21")
        assert m["status"] == CacheStatus.COMPUTED
        assert m.get("miss_reason") == "first time seeing this code"

    def test_unchanged_re_run_is_restored_and_has_no_miss_reason(self, magics_fixture):
        magics, shell, _backend, _cash = magics_fixture
        magics.cash("", "x = 21")               # first run, populates cache
        m = _last_metric(shell, magics, "x = 21")  # re-run, expect hit
        assert m["status"] == CacheStatus.RESTORED
        # RESTORED rows don't have a miss to attribute.
        assert m.get("miss_reason") is None

    def test_changed_input_lineage_is_attributed_to_input(self, magics_fixture):
        magics, shell, _backend, _cash = magics_fixture
        # Run the producer cell once, then the consumer once (both miss).
        magics.cash("", "a = 1")
        magics.cash("", "b = a + 1")
        # Edit the producer so its lineage changes. The consumer was cached
        # with the old `a` lineage; re-running it should now miss, and the
        # reason should mention input lineage.
        magics.cash("", "a = 2")
        m = _last_metric(shell, magics, "b = a + 1")
        assert m["status"] == CacheStatus.COMPUTED
        reason = m.get("miss_reason") or ""
        assert "lineage" in reason or "input" in reason, (
            f"expected an input-related miss reason, got: {reason!r}"
        )


class TestDiagnoseMissUnit:
    """Direct unit tests for ``StatementProcessor._diagnose_miss``."""

    def test_empty_backend_returns_first_time(self, magics_fixture):
        magics, _shell, _backend, _cash = magics_fixture
        proc = magics._statement_processor
        reason = proc._diagnose_miss(source_hash="never-seen", inputs=set())
        assert reason == "first time seeing this code"

    def test_same_code_different_input_set_names_the_diff(self, magics_fixture):
        magics, _shell, backend, _cash = magics_fixture
        # Plant a fake entry with one input.
        backend.set(
            "stmt:abcd",
            {"variables": {}, "stdout": "", "stderr": "", "rich_outputs": []},
            {
                "timestamp": 1.0,
                "inputs": ["a"],
                "outputs": ["b"],
                "execution_time": 0.01,
                "source_hash": "src-hash-xyz",
                "code": "b = a + 1",
                "key": "stmt:abcd",
                "output_lineages": {"b": "lh1"},
            },
        )
        proc = magics._statement_processor
        # Same code (source_hash matches) but our current inputs include `c`
        # which the prior didn't.
        reason = proc._diagnose_miss(source_hash="src-hash-xyz", inputs={"a", "c"})
        assert "added input(s): c" in (reason or "")
