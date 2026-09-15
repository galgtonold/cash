"""A cached call is keyed on the values it receives, not on where they came from.

Round 23's r23s3 fitted one model per machine and window::

    for W in WINDOWS:
        sc = {mid: fit_score(make_features(cleaned[mid], W)) for mid in ids}

The call's key held the lineage of all of ``cleaned`` and the text of the
statement around it. So fixing the data of 35 machines out of 200 re-fitted
every model -- 495 of 600 on features identical to the last run, "THE
repeated move in this project" -- and renaming ``sc`` to keep each window's
scores re-fitted all 180 of them.

A call whose inputs are all plain data now keys on the features it receives
and leaves the statement out, as long as nothing it reads can change while
its key stays put (CAS-256's ``fetch_next(conn)`` keeps its statement).
Written here as a ``for`` loop: a call inside a comprehension is cached only
when a notebook variable shares the comprehension's variable name (r23s3 had
one), which is a matter of its own. Counted with ``os.write``: a cached
callee's own writes to a variable would be restored on a hit and count the
same either way.
"""
from pathlib import Path

import pytest

pytest.importorskip("pandas")

pytestmark = [pytest.mark.integration, pytest.mark.timeout(120)]

SETUP = (
    "import os, time\n"
    "import pandas as pd\n"
    "def make_features(h, w):\n"
    "    return h.rolling(w, min_periods=1).mean()\n"
    "def fit_score(f):\n"
    "    fd = os.open('fits.log', os.O_WRONLY | os.O_CREAT | os.O_APPEND)\n"
    "    os.write(fd, b'fit|')\n"
    "    os.close(fd)\n"
    "    time.sleep(0.02)\n"
    "    return round(float(f['v'].sum()), 6)"
)
DATA = "raw = {m: pd.DataFrame({'v': [float(m * 100 + i) for i in range(30)]}) for m in range(6)}"
CLEAN = "SWAP = {swap}\ncleaned = {{m: (h * 2 if m in SWAP else h) for m, h in raw.items()}}"
SWEEP = (
    "rows = []\n"
    "for W in [3, 5]:\n"
    "    {target} = {{}}\n"
    "    for mid in sorted(cleaned):\n"
    "        {target}[mid] = fit_score(make_features(cleaned[mid], W))\n"
    "    rows.append((W, round(sum({target}.values()), 6)))\n"
    "print('ROWS', rows)"
)


def _cells(swap="[]", target="sc"):
    return ["import cash\n%cash_on", SETUP, DATA, CLEAN.format(swap=swap), SWEEP.format(target=target)]


def _fits(runner) -> int:
    log = Path(runner.work_dir) / "fits.log"
    return log.read_text().count("fit|") if log.exists() else 0


def _expected_rows(swap):
    import pandas as pd
    rows = []
    for w in [3, 5]:
        total = 0.0
        for m in range(6):
            h = pd.DataFrame({'v': [float(m * 100 + i) for i in range(30)]})
            h = h * 2 if m in swap else h
            total += round(float(h.rolling(w, min_periods=1).mean()['v'].sum()), 6)
        rows.append((w, round(total, 6)))
    return f"ROWS {rows}"


def test_fixing_some_elements_upstream_refits_only_those(nb_runner):
    nb_runner.create_notebook(_cells())
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert _expected_rows([]) in nb_runner.get_output(5)
    assert _fits(nb_runner) == 12

    nb_runner.set_cell_source(4, CLEAN.format(swap="[1, 4]"))
    nb_runner.run_all()

    assert _expected_rows([1, 4]) in nb_runner.get_output(5), nb_runner.get_output(5)
    # Two machines changed, in two windows each; the other eight fits are
    # served. Measured before: 12, every one of them.
    assert _fits(nb_runner) - 12 == 4


def test_renaming_what_the_sweep_assigns_refits_nothing(nb_runner):
    nb_runner.create_notebook(_cells())
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert _fits(nb_runner) == 12

    nb_runner.set_cell_source(5, SWEEP.format(target="scores"))
    nb_runner.run_cell(5)

    assert _expected_rows([]) in nb_runner.get_output(5), nb_runner.get_output(5)
    assert _fits(nb_runner) == 12, "the same fits on the same features ran again"


def test_the_same_call_in_another_cell_is_served(nb_runner):
    again = ("for W in [3]:\n"
             "    check = {}\n"
             "    for mid in sorted(cleaned):\n"
             "        check[mid] = fit_score(make_features(cleaned[mid], W))\n"
             "print('CHECK', round(sum(check.values()), 6))")
    nb_runner.create_notebook(_cells() + [again])
    nb_runner.start_kernel()
    nb_runner.run_all()

    assert _fits(nb_runner) == 12
    assert "CHECK" in nb_runner.get_output(6), nb_runner.get_output(6)
