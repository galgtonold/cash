"""CAS-173: a SyntaxError in an upstream cell must be DISCLOSED, not swallowed.

Before this fix, one unparseable upstream cell (a half-written cell the user
had merely SAVED, not run) made the upstream simulator re-raise a SyntaxError
that aborted the whole simulation and silently dropped the current cell into an
uncached fallback path — while ``auto_cache_enabled`` and the badge still said
caching was on. cash logged ``[UPSTREAM] Syntax error in cell N`` at debug level
and told the user nothing.

These in-process tests pin the new contract:

* a broken upstream cell emits a visible ``CashUpstreamSyntaxWarning`` naming
  the offending cell (1-based);
* the warning is deduped per distinct break (not once per downstream run) but
  re-fires when the break changes;
* the current cell is NOT dropped into the silent uncached fallback — the
  caching pipeline still runs, so ``auto_cache_enabled`` is not a lie;
* a VALID cell (including a multi-line ``%``-format print — CAS-163) never
  triggers the warning and never poisons.

Correctness containment (a downstream cell that does not depend on the broken
cell still RESTORES from cache) is proven end-to-end against a real kernel in
``tests/test_notebook_integration/test_cas173_upstream_syntax_error.py``.
"""
from __future__ import annotations

import json
import warnings
from unittest.mock import MagicMock, patch

import pytest
from traitlets.config.configurable import Configurable

from cash import CashUpstreamSyntaxWarning
from cash.backends import InMemoryBackend
from cash.core import Cash
from cash.notebook.ipython.magics import CashMagics


class MockShell(Configurable):
    """Mock IPython shell (real Configurable, not a sys.modules mock)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.user_ns = {}
        self.user_ns["_ih"] = []
        self.run_cell = MagicMock()
        self.input_transformers_cleanup = []
        self.display_pub = type("MockDisplayPub", (), {"publish": MagicMock()})()
        self.ast_transformers = []
        self.events = MagicMock()
        self.events.register = MagicMock(return_value=None)


@pytest.fixture
def harness(tmp_path):
    """Return ``(magics, shell, write_cells, run)``.

    ``write_cells`` rewrites the on-disk notebook (the SAVE step); ``run``
    executes a cell through the ``%cash_on`` hook pipeline while the checker
    reads the saved notebook.
    """
    backend = InMemoryBackend()
    backend.clear()
    cash = Cash(backend=backend, register_magic=False)
    shell = MockShell()
    magics = CashMagics(shell, cash)
    magics._auto_cache_enabled = True

    nb_path = tmp_path / "test.ipynb"

    def write_cells(cells):
        nb = {
            "cells": [
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [c],
                }
                for c in cells
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 4,
        }
        nb_path.write_text(json.dumps(nb), encoding="utf-8")

    def get_cells(_path=None):
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        return [c["source"][0] for c in nb["cells"]]

    def run(code):
        with patch(
            "cash.notebook.upstream.checker.get_notebook_cells",
            side_effect=get_cells,
        ):
            return magics._execute_cell(code)

    return magics, shell, write_cells, run


def _capture(run, code):
    """Run *code* capturing every warning raised during the upstream check."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        run(code)
    return rec


def _syntax_warnings(records):
    return [w for w in records if issubclass(w.category, CashUpstreamSyntaxWarning)]


def test_broken_upstream_cell_emits_named_warning(harness):
    """A saved-but-broken upstream cell emits a CashUpstreamSyntaxWarning that
    names the offending cell (1-based)."""
    magics, shell, write_cells, run = harness

    write_cells(["x = 10", "result = x * 2"])
    run("x = 10")
    run("result = x * 2")
    assert shell.user_ns["result"] == 20

    # Break cell 1 by SAVING only (never execute it); then run cell 2.
    write_cells(["x = 10 +", "result = x * 2"])
    rec = _capture(run, "result = x * 2")

    sw = _syntax_warnings(rec)
    assert sw, f"no CashUpstreamSyntaxWarning emitted; got {[w.category for w in rec]}"
    msg = str(sw[0].message)
    assert "cell 1" in msg, msg
    # The snippet of the offending source is included so the cell is unambiguous.
    assert "x = 10 +" in msg, msg


def test_no_warning_and_no_fallback_when_all_cells_valid(harness):
    """A notebook with only valid cells never warns and never drops into the
    silent uncached fallback (guards against over-eager flagging / CAS-163)."""
    magics, shell, write_cells, run = harness

    write_cells(["x = 10", "result = x * 2"])
    run("x = 10")
    magics._original_run_cell.reset_mock()
    rec = _capture(run, "result = x * 2")

    assert not _syntax_warnings(rec)
    # Normal completion delegates original_run_cell("pass"); a silent fallback
    # would instead call it with the raw cell code. The raw cell must never be
    # the argument.
    called_with = [c.args[0] for c in magics._original_run_cell.call_args_list if c.args]
    assert "result = x * 2" not in called_with, called_with


def test_multiline_percent_print_upstream_not_flagged(harness):
    """CAS-163 guard: a VALID multi-line ``%``-format print upstream cell must
    NOT be reported as broken and must NOT poison the downstream cell."""
    magics, shell, write_cells, run = harness

    pct_print = 'print("a=%.2f\\n" "b=%.2f" % (1.0, 2.0))'
    write_cells([pct_print, "result = 40 + 2"])
    run(pct_print)
    magics._original_run_cell.reset_mock()
    rec = _capture(run, "result = 40 + 2")

    assert not _syntax_warnings(rec), [str(w.message) for w in _syntax_warnings(rec)]
    called_with = [c.args[0] for c in magics._original_run_cell.call_args_list if c.args]
    assert "result = 40 + 2" not in called_with, called_with


def test_broken_upstream_does_not_use_silent_uncached_fallback(harness):
    """Honesty: with a broken upstream cell present, the downstream cell still
    runs through the caching pipeline instead of the silent uncached fallback,
    so ``auto_cache_enabled`` stays truthful."""
    magics, shell, write_cells, run = harness

    write_cells(["x = 10", "result = x * 2"])
    run("x = 10")
    run("result = x * 2")

    write_cells(["x = 10 +", "result = x * 2"])
    magics._original_run_cell.reset_mock()
    run("result = x * 2")

    # The bug routed the cell through original_run_cell(raw_cell) (uncached).
    # After the fix the pipeline completes and only delegates "pass".
    called_with = [c.args[0] for c in magics._original_run_cell.call_args_list if c.args]
    assert "result = x * 2" not in called_with, (
        "downstream cell fell into the silent uncached fallback while a broken "
        f"upstream cell was present (CAS-173). original_run_cell calls: {called_with}"
    )
    assert magics._auto_cache_enabled is True


def test_persistent_break_warns_once_then_re_warns_on_change(harness):
    """Dedup: an unchanged break warns once across repeated downstream runs; a
    CHANGED break (or a fixed-then-rebroken cell) warns again."""
    magics, shell, write_cells, run = harness

    write_cells(["x = 10", "result = x * 2"])
    run("x = 10")
    run("result = x * 2")

    # First break -> warns.
    write_cells(["x = 10 +", "result = x * 2"])
    assert _syntax_warnings(_capture(run, "result = x * 2")), "first break did not warn"

    # Same break, run downstream again -> deduped (silent).
    assert not _syntax_warnings(_capture(run, "result = x * 2")), "duplicate break re-warned"

    # Change the break -> re-warns.
    write_cells(["x = 10 + +", "result = x * 2"])
    assert _syntax_warnings(_capture(run, "result = x * 2")), "changed break did not re-warn"

    # Fix the cell -> no warning; re-break -> warns again.
    write_cells(["x = 10", "result = x * 2"])
    assert not _syntax_warnings(_capture(run, "result = x * 2")), "fixed cell warned"
    write_cells(["x = 10 +", "result = x * 2"])
    assert _syntax_warnings(_capture(run, "result = x * 2")), "re-broken cell did not warn"
