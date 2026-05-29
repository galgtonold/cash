"""Tests for the cache-miss attribution feature.

When the cache-check path detects TTL expiry or a file-dependency change,
the runtime stamps a short ``miss_reason`` on the metric dict so the
badge's row-detail drawer can answer "why did this cell re-run?".

The earlier backend-walking fallback (``_diagnose_miss``) was removed
because it was O(N²) in cache size and dominated cold-run wall time
(see the 2026-05-18 overhead analysis). The remaining miss-reason path
covers only the *cheap* attributions that the cache-check already
computes as a side effect — TTL and file invalidations. On the empty-key
"first time seeing this code" path, no miss reason is attached.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from traitlets.config import Configurable

from cash.backends import InMemoryBackend
from cash.core import Cash
from cash.notebook.cache_status import CacheStatus
from cash.notebook.ipython.magics import CashMagics


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
    def test_first_run_has_no_miss_reason_attached(self, magics_fixture):
        """The first execution of a statement is a cache miss with no cheap
        reason available (no TTL, no file dep). Since the backend-walking
        fallback was removed, miss_reason stays unset."""
        magics, shell, _backend, _cash = magics_fixture
        m = _last_metric(shell, magics, "x = 21")
        assert m["status"] == CacheStatus.COMPUTED
        assert m.get("miss_reason") is None

    def test_unchanged_re_run_is_restored_and_has_no_miss_reason(self, magics_fixture):
        magics, shell, _backend, _cash = magics_fixture
        # @cash:persist forces caching regardless of the 10 ms min-execution-time floor
        magics.cash("", "# @cash:persist\nx = 21")               # first run, populates cache
        m = _last_metric(shell, magics, "# @cash:persist\nx = 21")  # re-run, expect hit
        assert m["status"] == CacheStatus.RESTORED
        # RESTORED rows don't have a miss to attribute.
        assert m.get("miss_reason") is None

    def test_changed_input_has_no_miss_reason_attached(self, magics_fixture):
        """Changing an upstream value invalidates the consumer's cache, but
        without the O(N²) diagnostic we no longer label why."""
        magics, shell, _backend, _cash = magics_fixture
        magics.cash("", "a = 1")
        magics.cash("", "b = a + 1")
        magics.cash("", "a = 2")
        m = _last_metric(shell, magics, "b = a + 1")
        assert m["status"] == CacheStatus.COMPUTED
        # No cheap reason fired for this kind of miss.
        assert m.get("miss_reason") is None
