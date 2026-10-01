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

from cash.notebook.cache_status import CacheStatus
from tests._cell_driver import run_cash_cell


def _last_metric(shell, magics, code: str) -> dict:
    """Run one cell and return the last metric the processor recorded."""
    captured: list[dict] = []
    real_render = magics.badges.render

    def capture(metrics, **kw):
        captured.append(list(metrics))
        return real_render(metrics, **kw)

    magics.badges.render = capture  # type: ignore[assignment]
    try:
        run_cash_cell(magics, code)
    finally:
        magics.badges.render = real_render  # type: ignore[assignment]
    assert captured, "no metrics captured"
    return captured[-1][-1]


class TestMissReasonAttribution:
    def test_first_run_has_no_miss_reason_attached(self, cash_magics, mock_shell):
        """The first execution of a statement is a cache miss with no cheap
        reason available (no TTL, no file dep). Since the backend-walking
        fallback was removed, miss_reason stays unset."""
        m = _last_metric(mock_shell, cash_magics, "x = 21")
        assert m["status"] == CacheStatus.COMPUTED
        assert m.get("miss_reason") is None

    def test_unchanged_re_run_is_restored_and_has_no_miss_reason(self, cash_magics, mock_shell):
        # @cash:persist forces caching regardless of the 10 ms min-execution-time floor
        run_cash_cell(cash_magics, "# @cash:persist\nx = 21")  # first run, populates cache
        m = _last_metric(mock_shell, cash_magics, "# @cash:persist\nx = 21")  # re-run, expect hit
        assert m["status"] == CacheStatus.RESTORED
        # RESTORED rows don't have a miss to attribute.
        assert m.get("miss_reason") is None

    def test_changed_input_is_named_on_the_badge(self, cash_magics, mock_shell):
        """Changing an upstream value invalidates the consumer -- and says so.

        The reason comes from ``executed_input_lineages`` (what the statement
        last ran with) versus the current lineage, so naming ``a`` costs an
        O(inputs) dict walk and no backend access.
        """
        run_cash_cell(cash_magics, "a = 1")
        run_cash_cell(cash_magics, "b = a + 1")
        run_cash_cell(cash_magics, "a = 2")
        m = _last_metric(mock_shell, cash_magics, "b = a + 1")
        assert m["status"] == CacheStatus.COMPUTED
        assert m.get("miss_reason") == "input changed: a", m.get("miss_reason")

    def test_only_the_input_that_actually_changed_is_named(self, cash_magics, mock_shell):
        """Control arm: the reason must discriminate, not list every input.

        Without this, an implementation that named all inputs of any re-run
        statement would pass the test above while telling the user nothing.
        """
        run_cash_cell(cash_magics, "a = 1")
        run_cash_cell(cash_magics, "c = 100")
        run_cash_cell(cash_magics, "b = a + c")
        run_cash_cell(cash_magics, "a = 2")  # only `a` moves; `c` is untouched
        m = _last_metric(mock_shell, cash_magics, "b = a + c")
        assert m["status"] == CacheStatus.COMPUTED
        assert m.get("miss_reason") == "input changed: a", m.get("miss_reason")

    def test_a_statement_whose_inputs_are_unchanged_is_not_labelled(self, cash_magics, mock_shell):
        """Control arm: re-running with nothing changed must stay silent.

        Guards the direction the feature could fail in without any test
        noticing -- attributing a change that did not happen is worse than
        attributing nothing, because it sends the reader after the wrong
        variable.
        """
        run_cash_cell(cash_magics, "a = 1")
        run_cash_cell(cash_magics, "# @cash:no-cache\nb = a + 1")
        m = _last_metric(mock_shell, cash_magics, "# @cash:no-cache\nb = a + 1")
        assert m.get("miss_reason") is None, m.get("miss_reason")
