"""Tests for the cache-miss attribution feature.

When the cache-check path detects TTL expiry or a file-dependency change,
the runtime stamps a short ``miss_reason`` on the metric dict so the
badge's row-detail drawer can answer "why did this cell re-run?".

The earlier backend-walking fallback (``_diagnose_miss``) was removed
because it was O(N²) in cache size and dominated cold-run wall time
(see the 2026-05-18 overhead analysis). Everything here must stay on the
cheap side of that line: no attribution may probe the backend.

Covered, and how each stays cheap:

* **TTL and file invalidations** — the cache-check already computes them
  as a side effect, so the reason is free.
* **A changed input** — compared against ``executed_input_lineages``,
  which records what the statement last ran with. An O(inputs) dict walk,
  never a cache scan. This is the most common reason a statement re-runs
  and used to be the one the badge could not name.

Deliberately NOT covered: the first run of a statement. Proving the
absence of an entry is what made the old fallback expensive, and "first
time" is self-evident to someone running a cell for the first time. The
test below pins that absence on purpose — it is a decision, not a gap.
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

    def test_changed_input_is_named_on_the_badge(self, magics_fixture):
        """Changing an upstream value invalidates the consumer -- and says so.

        The reason comes from ``executed_input_lineages`` (what the statement
        last ran with) versus the current lineage, so naming ``a`` costs an
        O(inputs) dict walk and no backend access.
        """
        magics, shell, _backend, _cash = magics_fixture
        magics.cash("", "a = 1")
        magics.cash("", "b = a + 1")
        magics.cash("", "a = 2")
        m = _last_metric(shell, magics, "b = a + 1")
        assert m["status"] == CacheStatus.COMPUTED
        assert m.get("miss_reason") == "input changed: a", m.get("miss_reason")

    def test_only_the_input_that_actually_changed_is_named(self, magics_fixture):
        """Control arm: the reason must discriminate, not list every input.

        Without this, an implementation that named all inputs of any re-run
        statement would pass the test above while telling the user nothing.
        """
        magics, shell, _backend, _cash = magics_fixture
        magics.cash("", "a = 1")
        magics.cash("", "c = 100")
        magics.cash("", "b = a + c")
        magics.cash("", "a = 2")            # only `a` moves; `c` is untouched
        m = _last_metric(shell, magics, "b = a + c")
        assert m["status"] == CacheStatus.COMPUTED
        assert m.get("miss_reason") == "input changed: a", m.get("miss_reason")

    def test_a_statement_whose_inputs_are_unchanged_is_not_labelled(self, magics_fixture):
        """Control arm: re-running with nothing changed must stay silent.

        Guards the direction the feature could fail in without any test
        noticing -- attributing a change that did not happen is worse than
        attributing nothing, because it sends the reader after the wrong
        variable.
        """
        magics, shell, _backend, _cash = magics_fixture
        magics.cash("", "a = 1")
        magics.cash("", "# @cash:no-cache\nb = a + 1")
        m = _last_metric(shell, magics, "# @cash:no-cache\nb = a + 1")
        assert m.get("miss_reason") is None, m.get("miss_reason")
