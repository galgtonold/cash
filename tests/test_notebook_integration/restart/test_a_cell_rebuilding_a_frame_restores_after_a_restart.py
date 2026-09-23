"""A cell that rebuilds one frame in steps restores after a restart, not re-runs.

The cleaning cell rebuilds ``sales`` through a dozen
``sales[...] = ...`` steps. Only the last version is written to disk -- the
others would be copies of a 500 MB frame nothing restores -- and after a
restart Run All re-ran all of them: 6.6-8 s, "saved 0.33s". The cell now jumps
to the last version on disk, and runs only what that version does not cover.
"""

import ast

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(600)]

_TEE = """
import cash.notebook.statement.processor as _p
C = _p.StatementProcessor
if not hasattr(C, "_test_orig"):
    C._test_orig = C.process_statement
    def _tee(self, code, *a, **k):
        r = C._test_orig(self, code, *a, **k)
        try:
            s = r.get("status")
            if str(getattr(s, "value", s)) == "COMPUTED":
                C._test_ran.append(str(code).strip().splitlines()[-1][:40])
        except Exception:
            pass
        return r
    C.process_statement = _tee
C._test_ran = []
"""
_RAN = "__import__('cash.notebook.statement.processor', fromlist=['_']).StatementProcessor._test_ran"

CLEAN = (
    "sales = raw[raw['a'] >= 0]\n"
    "sales = (time.sleep(0.5), sales.drop_duplicates())[1]\n"
    "sales['t'] = (time.sleep(0.4), sales['a'] * 2)[1]\n"
    "is_big = sales['a'] > 50_000\n"
    "sales['big'] = is_big.astype(int)\n"
    "sales['c'] = (time.sleep(0.15), sales['t'] + 1)[1]\n"
    "print('S', len(sales), int(sales['big'].sum()), int(is_big.sum()), int(sales['c'].iloc[-1]))"
)
CELLS = [
    "import cash\n%load_ext cash\n%cash_badge print\n%cash_on",
    "import time\nimport numpy as np\nimport pandas as pd",
    "raw = (time.sleep(0.3), pd.DataFrame({'a': np.arange(400_000), 'b': np.arange(400_000) % 7}))[1]",
    CLEAN,
]
EXPECTED = "S 400000 349999 349999 799999"


def _restart_run_all(nb_runner):
    nb_runner.restart()
    nb_runner._inject_notebook_path()
    nb_runner.run_cell(1)
    nb_runner.peek(f"exec({_TEE!r}, {{}})")
    nb_runner.peek("get_ipython().run_line_magic('cash_on', '')")
    for i in range(2, len(CELLS) + 1):
        nb_runner.run_cell(i)
    return [c for c in ast.literal_eval(nb_runner.peek(_RAN)) if "__CASH_PEEK__" not in c]


def test_the_expensive_steps_are_not_re_run_after_a_restart(nb_runner):
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert EXPECTED in nb_runner.get_output(4)

    ran = _restart_run_all(nb_runner)
    out = nb_runner.get_output(4)
    assert EXPECTED in out, out
    assert not [c for c in ran if "drop_dup" in c or "sales['t']" in c], f"re-ran after the restart: {ran}\n{out}"

    # And once more in the same kernel: still nothing expensive.
    nb_runner.peek(f"exec({_TEE!r}, {{}})")
    nb_runner.run_cell(4)
    ran = [c for c in ast.literal_eval(nb_runner.peek(_RAN)) if "__CASH_PEEK__" not in c]
    out = nb_runner.get_output(4)
    assert EXPECTED in out, out
    assert not [c for c in ran if "drop_dup" in c or "sales['t']" in c], f"re-ran on a second run: {ran}\n{out}"


def test_an_edited_step_still_recomputes_after_a_restart(nb_runner):
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.set_cell_source(4, CLEAN.replace("sales['t'] + 1)", "sales['t'] + 2)"))
    _restart_run_all(nb_runner)
    assert "S 400000 349999 349999 800000" in nb_runner.get_output(4), nb_runner.get_output(4)


VERSIONS = (
    "sales = raw[raw['a'] >= 0]\n"
    "sales = (time.sleep(0.5), sales.drop_duplicates())[1]\n"
    "sales['t'] = (time.sleep(0.4), sales['a'] * 2)[1]\n"
    "is_big = sales['t'] > 500_000\n"
    "sales['t'] = (time.sleep(0.15), sales['t'] * 10)[1]\n"
    "print('V', int(is_big.sum()), int(sales['t'].iloc[-1]))"
)


def test_a_step_that_must_run_reads_the_version_before_it_not_the_restored_one(nb_runner):
    """``is_big`` reads ``t`` before it is multiplied. With its own entry gone,
    it must re-run on the ``sales`` of its place in the cell -- not on the final
    ``sales`` a restore just put back, which counts 374999 rows."""
    from pathlib import Path

    from cash.backends.entry_format import read_entry

    cells = CELLS[:3] + [VERSIONS]
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "V 149999 7999980" in nb_runner.get_output(4)
    nb_runner.restart()
    removed = 0
    for entry in (Path(nb_runner.work_dir) / ".cash").glob("*.entry"):
        meta, _ = read_entry(str(entry), with_payload=False)
        if meta.get("outputs") == ["is_big"]:
            entry.unlink()
            removed += 1
    assert removed, "precondition: is_big had an entry on disk"
    nb_runner._inject_notebook_path()
    for i in range(1, len(cells) + 1):
        nb_runner.run_cell(i)
    out = nb_runner.get_output(4)
    assert "V 149999 7999980" in out, out
