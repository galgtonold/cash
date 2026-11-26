"""A library setting a statement's run read is checked before its entry is
served, and the badge says which one moved.

``s = df.to_string()`` names ``df`` alone: a ``pd.set_option`` made since
moves no lineage it reads. On Python 3.12+ the run records the settings the
library code it ran read (``library_reads``, as for a traced ``@cash.cache``
call), and a hit is served only while each holds the value it was stored
under. Below 3.12, or with ``CASH_TRACE=0``, nothing records them: cash says
so once (KEY-LIBRARY-STATE-UNSEEN).
"""

from __future__ import annotations

import warnings

import pytest

from cash import CashImpurityWarning
from cash.decorator import code_trace
from cash.notebook.cache_status import CacheStatus
from tests._cell_driver import run_cash_cell

pd = pytest.importorskip("pandas")

SETUP = "import pandas as pd\ndf = pd.DataFrame({'a': [1.123456789, 2.5]})"
SHOW = "# @cash:persist\ns = df.to_string()"


def _last_metric(magics, code: str) -> dict:
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
    return captured[-1][-1]


@pytest.fixture
def options_restored():
    yield
    pd.reset_option("display.precision")
    pd.reset_option("display.max_rows")


@pytest.mark.skipif(not code_trace.on(), reason="records what a statement ran: Python 3.12+, CASH_TRACE on")
class TestRecorded:
    def test_a_changed_option_the_run_read_reruns_it_and_names_the_option(
        self, cash_magics, mock_shell, options_restored
    ):
        run_cash_cell(cash_magics, SETUP)
        run_cash_cell(cash_magics, SHOW)
        assert _last_metric(cash_magics, SHOW)["status"] == CacheStatus.RESTORED

        run_cash_cell(cash_magics, "pd.set_option('display.precision', 2)")
        m = _last_metric(cash_magics, SHOW)

        assert m["status"] == CacheStatus.COMPUTED
        assert mock_shell.user_ns["s"] == df_text(mock_shell, 2)
        reason = m.get("miss_reason") or ""
        assert "['display']['precision']" in reason, reason
        assert "a library setting the run read, changed" in reason, reason

    def test_an_option_the_run_did_not_read_leaves_it_served(self, cash_magics, mock_shell, options_restored):
        run_cash_cell(cash_magics, SETUP)
        run_cash_cell(cash_magics, SHOW)

        run_cash_cell(cash_magics, "pd.set_option('display.max_rows', 7)")

        assert _last_metric(cash_magics, SHOW)["status"] == CacheStatus.RESTORED

    def test_the_option_set_back_serves_the_first_entry_again_after_the_rerun(
        self, cash_magics, mock_shell, options_restored
    ):
        run_cash_cell(cash_magics, SETUP)
        run_cash_cell(cash_magics, SHOW)
        run_cash_cell(cash_magics, "pd.set_option('display.precision', 2)")
        run_cash_cell(cash_magics, SHOW)

        run_cash_cell(cash_magics, "pd.reset_option('display.precision')")
        m = _last_metric(cash_magics, SHOW)

        # Stored again under precision 2: computed once more, never served the 2-digit text.
        assert m["status"] == CacheStatus.COMPUTED
        assert mock_shell.user_ns["s"] == df_text(mock_shell, 6)


def df_text(shell, precision: int) -> str:
    with pd.option_context("display.precision", precision):
        return shell.user_ns["df"].to_string()


def test_an_unrecorded_statement_reaching_a_library_warns_once(cash_magics, mock_shell, monkeypatch):
    """Below 3.12 or with ``CASH_TRACE=0`` (the notebook's traces off), a
    stored statement reaching a library says once that a setting changed
    since is not seen, as an unrecorded ``@cash.cache`` call does."""
    from cash.decorator import library_reads

    monkeypatch.setattr(library_reads, "UNSEEN_SAID", False)
    monkeypatch.setenv("CASH_TRACE", "0")
    from cash.notebook import shadow

    monkeypatch.setattr(shadow, "SHADOW", None)
    run_cash_cell(cash_magics, SETUP)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_cash_cell(cash_magics, SHOW)
        run_cash_cell(cash_magics, "# @cash:persist\nt = df.to_string()")
        run_cash_cell(cash_magics, "# @cash:persist\nu = 1 + 2")
    said = [w for w in caught if "KEY-LIBRARY-STATE-UNSEEN" in str(w.message)]
    assert len(said) == 1, [str(w.message) for w in caught]
    assert "pd.set_option" in str(said[0].message)
    assert said[0].category is CashImpurityWarning
