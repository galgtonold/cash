"""An expensive call wrapped in a builtin or a class is still call-cached.

Round 22 (r22s4, found again by the tester-session tests): a backtest built
its table with ``rows.append(dict(cutoff=c, alpha=a, wape=score(df, c, a)))``.
``dict(...)`` was taken as the outermost call to cache, a class is never
wrapped at runtime, and ``score`` inside it was never considered -- so an
unchanged re-run refitted every model. ``rows.append(score(...))`` and
``w = score(...)`` were always reused.
"""

from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration]

SETUP = (
    "import time\n"
    "def score(c, a):\n"
    "    open('calls.log', 'a').write(f'score {c} {a}\\n')\n"
    "    time.sleep(0.05)\n"
    "    return c * a\n"
    "CUTOFFS = (1, 2, 3)\n"
    "GRID = (10, 20)"
)
LOOP = (
    "rows = []\n"
    "for c in CUTOFFS:\n"
    "    for a in GRID:\n"
    "        rows.append(dict(cutoff=c, alpha=a, err=score(c, a)))\n"
    "print(sum(r['err'] for r in rows))"
)


def _calls(runner) -> list[str]:
    log = Path(runner.work_dir) / "calls.log"
    return log.read_text().splitlines() if log.exists() else []


def test_an_unchanged_rerun_reuses_every_call(nb_runner):
    nb_runner.create_notebook(["import cash\n%cash_on", SETUP, LOOP])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "180" in nb_runner.get_output(3)
    before = len(_calls(nb_runner))
    nb_runner.run_cell(3)
    assert "180" in nb_runner.get_output(3)
    assert _calls(nb_runner)[before:] == [], "an unchanged re-run recomputed the calls"

    # A new grid value costs only its own calls.
    nb_runner.set_cell_source(2, SETUP.replace("GRID = (10, 20)", "GRID = (10, 20, 30)"))
    nb_runner.run_all()
    assert "360" in nb_runner.get_output(3)
    assert sorted(_calls(nb_runner)[before:]) == ["score 1 30", "score 2 30", "score 3 30"]
