"""After a restart, a statement that calls a notebook function is restored, not re-run.

Round 23 (r23s3, 2026-09-14): a restart and one run of the last cell re-ran the
whole notebook -- 350 s against 370 s uncached. The upstream simulation meets
``summary = slow_summary(raw)`` before ``def slow_summary`` has run again, so
the function is not in ``user_ns``, and the two key components that come from
the live function -- its source digest, and the globals its code reads -- were
missing from the simulated key. It never matched the entry the runtime wrote,
so the statement was re-run, and its inputs were rebuilt for it: every
producer above it that called a function too.

Counted, not timed: a tee on ``StatementProcessor.process_statement`` in the
new kernel counts the statements that ran, by their text.
"""
import ast

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(600)]

_TEE = '''
import cash.notebook.statement.processor as _p
C = _p.StatementProcessor
if not hasattr(C, "_test_orig"):
    C._test_orig = C.process_statement
    def _tee(self, code, *a, **k):
        r = C._test_orig(self, code, *a, **k)
        try:
            s = r.get("status")
            s = str(getattr(s, "value", s))
            if s == "COMPUTED":
                C._test_ran.append(str(code).strip().splitlines()[-1][:40])
        except Exception:
            pass
        return r
    C.process_statement = _tee
C._test_ran = []
'''
_UNTEE = '''
import cash.notebook.statement.processor as _p
C = _p.StatementProcessor
if hasattr(C, "_test_orig"):
    C.process_statement = C._test_orig
    del C._test_orig
'''
_RAN = "__import__('cash.notebook.statement.processor', fromlist=['_']).StatementProcessor._test_ran"

#: ``load`` is quick, so ``raw`` lives in RAM only and is gone after the
#: restart -- the shape that makes a re-run expensive: rebuilding an input
#: the restored statement never needed (in r23s2, a loop over 1,312 files).
SETUP = ("import time\n"
         "SCALE = 3\n"
         "def load(n):\n"
         "    return [i * SCALE for i in range(n)]\n"
         "def slow_summary(xs):\n"
         "    time.sleep(0.4)\n"
         "    return sum(xs) + OFFSET")
OFFSET = "OFFSET = 0"
LOAD = "raw = load(1000)"
SUMMARY = "summary = slow_summary(raw)"
REPORT = "print('S', summary)"
CELLS = ["import cash\n%cash_on", SETUP, OFFSET, LOAD, SUMMARY, REPORT]


@pytest.fixture
def _teed(nb_runner):
    yield
    try:
        nb_runner.peek(f"exec({_UNTEE!r}, {{}})")
    except Exception:  # noqa: BLE001 - a kernel that never started has nothing to undo
        pass


def _restart_and_report(nb_runner) -> list[str]:
    """Restart, run the last cell alone; return the statements that ran."""
    nb_runner.restart()
    nb_runner._inject_notebook_path()
    nb_runner.run_cell(1)
    nb_runner.peek(f"exec({_TEE!r}, {{}})")
    # The peek's own upstream check simulated the notebook and cached it
    # before the tee existed; %cash_on drops that cache.
    nb_runner.peek("get_ipython().run_line_magic('cash_on', '')")
    nb_runner.run_cell(len(CELLS))
    return [code for code in ast.literal_eval(nb_runner.peek(_RAN)) if "__CASH_PEEK__" not in code]


def test_a_restart_restores_a_call_instead_of_rerunning_its_producers(nb_runner, _teed):
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "S 1498500" in nb_runner.get_output(len(CELLS))

    ran = _restart_and_report(nb_runner)

    assert "S 1498500" in nb_runner.get_output(len(CELLS))
    # Measured before the fix: the setup cell and `raw = load(1000)` ran
    # again to rebuild `raw`, and only then did `summary` restore.
    assert ran == [REPORT], f"ran after the restart: {ran}"


@pytest.mark.parametrize("cell, edit, want", [
    # The callee's own source: its digest, and so the key, moves.
    (2, SETUP.replace("sum(xs) + OFFSET", "sum(xs) + OFFSET + 1"), "S 1498501"),
    # A global only the callee reads: the callee component, at the call's
    # position, moves.
    (3, "OFFSET = 7", "S 1498507"),
])
def test_an_edit_before_the_restart_is_not_served_the_old_value(nb_runner, _teed, cell, edit, want):
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.set_cell_source(cell, edit)

    ran = _restart_and_report(nb_runner)

    assert want in nb_runner.get_output(len(CELLS)), nb_runner.get_output(len(CELLS))
    assert SUMMARY in ran
