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


def test_a_call_on_a_module_does_not_change_it_after_a_restart(nb_runner, _teed):
    """``pd.set_option(...)`` is a module function call, and the runtime leaves
    ``pd``'s lineage alone. After a restart ``pd`` was not imported yet and the
    runtime's verdict on the call was gone, so the simulation read it as an
    unknown method that mutates its receiver and bumped ``pd`` -- the key of
    every statement reading ``pd`` moved, and nothing restored (r23s2)."""
    pytest.importorskip("pandas")
    cells = ["import cash\n%cash_on",
             "import time\nimport pandas as pd\npd.set_option('display.width', 160)\n"
             "def slow_total(t):\n    time.sleep(0.4)\n    return int(t['v'].sum())",
             "raw = pd.DataFrame({'v': range(1000)})",
             "total = slow_total(raw)",
             "print('T', total)"]
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "T 499500" in nb_runner.get_output(5)

    nb_runner.restart()
    nb_runner._inject_notebook_path()
    nb_runner.run_cell(1)
    nb_runner.peek(f"exec({_TEE!r}, {{}})")
    nb_runner.peek("get_ipython().run_line_magic('cash_on', '')")
    nb_runner.run_cell(5)
    ran = [c for c in ast.literal_eval(nb_runner.peek(_RAN)) if "__CASH_PEEK__" not in c]

    assert "T 499500" in nb_runner.get_output(5)
    assert ran == ["print('T', total)"], f"ran after the restart: {ran}"


def _restores_only_the_report(nb_runner, cells: list[str]) -> list[str]:
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    nb_runner.run_all()
    want = nb_runner.get_output(len(cells))
    nb_runner.restart()
    nb_runner._inject_notebook_path()
    nb_runner.run_cell(1)
    nb_runner.peek(f"exec({_TEE!r}, {{}})")
    nb_runner.peek("get_ipython().run_line_magic('cash_on', '')")
    nb_runner.run_cell(len(cells))
    assert nb_runner.get_output(len(cells)) == want
    return [c for c in ast.literal_eval(nb_runner.peek(_RAN)) if "__CASH_PEEK__" not in c]


SLOW_LEN = ("import time\nfrom pathlib import Path\n"
            "def slow_len(p):\n    time.sleep(0.4)\n    return len(str(p)) * 1000")


def test_a_value_built_by_an_imported_class_is_restored_after_a_restart(nb_runner, _teed):
    """``DATA = Path(...)``: the runtime folds ``Path``'s source digest into
    ``DATA``'s lineage. After a restart ``Path`` was not imported yet, the
    simulation left the digest out, and nothing built from ``DATA`` restored
    (r23s2: ``EXPORTS``, and so every table in the notebook)."""
    ran = _restores_only_the_report(nb_runner, [
        "import cash\n%cash_on", SLOW_LEN, "DATA = Path('data_dir')",
        "total = slow_len(DATA)", "print('T', total)"])
    assert ran == ["print('T', total)"], f"ran after the restart: {ran}"


def test_a_path_a_directory_is_made_from_is_restored_after_a_restart(nb_runner, _teed):
    """r23s2's ``PACK = Path('pack'); PACK.mkdir(exist_ok=True)``: ``Path``'s
    digest in ``OUT``'s lineage, and a method called on ``OUT`` that is decided
    without a verdict (it writes the filesystem, not the object)."""
    ran = _restores_only_the_report(nb_runner, [
        "import cash\n%cash_on", SLOW_LEN, "OUT = Path('out_dir')\nOUT.mkdir(exist_ok=True)",
        "total = slow_len(OUT)", "print('T', total)"])
    assert ran == ["print('T', total)"], f"ran after the restart: {ran}"


def test_names_from_a_module_not_loaded_yet_are_restored_after_a_restart(nb_runner, _teed):
    """``from zipapp import get_interpreter`` (a function) and ``from wave import
    Wave_read`` (a class), read by a helper. The runtime folds both digests into
    the def's lineage; after a restart neither module is loaded when the
    simulation meets the import, so it took both for modules, had no digest,
    and every call of the helper got a lineage the runtime never gave it
    (r23s2: ``from statsmodels... import ExponentialSmoothing``, a 45 s cell)."""
    cells = ["import cash\n%cash_on",
             "import time\nfrom zipapp import get_interpreter\nfrom wave import Wave_read, Wave_write\n"
             "def slow_hue(r):\n    time.sleep(0.4)\n    assert Wave_read and Wave_write\n    return round(r * 2, 4) if get_interpreter else None",
             "hue = slow_hue(0.3)",
             "print('H', hue)"]
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    nb_runner.run_all()
    want = nb_runner.get_output(4)
    nb_runner.restart()
    nb_runner._inject_notebook_path()
    nb_runner.run_cell(1)
    assert nb_runner.peek("[m for m in ('zipapp', 'wave') if m in __import__('sys').modules]") == "[]", \
        "the modules are loaded already: this would not exercise the recorded bindings"
    nb_runner.peek(f"exec({_TEE!r}, {{}})")
    nb_runner.peek("get_ipython().run_line_magic('cash_on', '')")
    nb_runner.run_cell(4)
    ran = [c for c in ast.literal_eval(nb_runner.peek(_RAN)) if "__CASH_PEEK__" not in c]

    assert nb_runner.get_output(4) == want
    assert ran == ["print('H', hue)"], f"ran after the restart: {ran}"


def test_a_call_the_runtime_saw_leave_its_receiver_alone_does_not_change_it_after_a_restart(nb_runner, _teed):
    """``x.limit_denominator(10)`` is a bare call to a method cash does not
    know, so the runtime watches the receiver, sees it unchanged, and records
    that verdict. The record died with the kernel; after a restart the
    simulation assumed the unknown method mutates ``x`` and bumped it."""
    ran = _restores_only_the_report(nb_runner, [
        "import cash\n%cash_on",
        "import time\nfrom fractions import Fraction\n"
        "def slow_num(x):\n    time.sleep(0.4)\n    return x.numerator * 1000",
        "x = Fraction(1, 3)\nx.limit_denominator(10)",
        "total = slow_num(x)", "print('T', total)"])
    assert ran == ["print('T', total)"], f"ran after the restart: {ran}"


def test_a_value_built_by_a_notebook_class_is_restored_after_a_restart(nb_runner, _teed):
    """``b = Box(3)``: the runtime folds ``Box``'s source digest into ``b``'s
    lineage, and after a restart ``class Box`` has not run again."""
    ran = _restores_only_the_report(nb_runner, [
        "import cash\n%cash_on",
        "import time\nclass Box:\n    def __init__(self, n):\n        self.n = n\n"
        "def slow_n(b):\n    time.sleep(0.4)\n    return b.n * 1000",
        "b = Box(3)",
        "total = slow_n(b)", "print('T', total)"])
    assert ran == ["print('T', total)"], f"ran after the restart: {ran}"


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
