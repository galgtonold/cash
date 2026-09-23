"""A cached call is keyed on the values it receives, not on where they came from.

One user fitted one model per machine and window::

    for W in WINDOWS:
        sc = {mid: fit_score(make_features(cleaned[mid], W)) for mid in ids}

The call's key held the lineage of all of ``cleaned`` and the text of the
statement around it. So fixing the data of 35 machines out of 200 re-fitted
every model -- 495 of 600 on features identical to the last run, "THE
repeated move in this project" -- and renaming ``sc`` to keep each window's
scores re-fitted all 180 of them.

A call whose inputs are all plain data now keys on the features it receives
and leaves the statement out, as long as nothing it reads can change while
its key stays put (``fetch_next(conn)`` keeps its statement).
Written here as a ``for`` loop; the comprehension form keys the same way.
Counted with ``os.write``: a cached
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
            h = pd.DataFrame({"v": [float(m * 100 + i) for i in range(30)]})
            h = h * 2 if m in swap else h
            total += round(float(h.rolling(w, min_periods=1).mean()["v"].sum()), 6)
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
    again = (
        "for W in [3]:\n"
        "    check = {}\n"
        "    for mid in sorted(cleaned):\n"
        "        check[mid] = fit_score(make_features(cleaned[mid], W))\n"
        "print('CHECK', round(sum(check.values()), 6))"
    )
    nb_runner.create_notebook(_cells() + [again])
    nb_runner.start_kernel()
    nb_runner.run_all()

    assert _fits(nb_runner) == 12
    assert "CHECK" in nb_runner.get_output(6), nb_runner.get_output(6)


# Another fitted a forecast per store and item:
#     cutoff = work["date"].max() - pd.Timedelta(days=28)
#     models = {key: fit_series(g, PARAMS, cutoff) for key, g in work.groupby(...)}
# Fixing ONE store's data re-fitted all 360. ``cutoff`` and ``PARAMS`` are
# passed by name, and a bare name was still keyed on where it came from: the
# cutoff is recomputed from the frame, so its lineage moves although its
# value does not.
STORES = (
    "import os, time\n"
    "import pandas as pd\n"
    "def fit_series(g, params, cutoff):\n"
    "    fd = os.open('fits.log', os.O_WRONLY | os.O_CREAT | os.O_APPEND)\n"
    "    os.write(fd, b'fit|')\n"
    "    os.close(fd)\n"
    "    time.sleep(0.02)\n"
    "    train = g[g['day'] <= cutoff]\n"
    "    return round(float(train['v'].mean() * params['alpha']), 6)"
)
WORK = (
    "FIX = {fix}\n"
    "work = pd.DataFrame({{'store': [s for s in range(6) for _ in range(10)],\n"
    "                      'day': [d for _ in range(6) for d in range(10)],\n"
    "                      'v': [float(s * 10 + d + (5 if s in FIX else 0)) for s in range(6) for d in range(10)]}})"
)
FIT = (
    "PARAMS = {'alpha': 0.5}\n"
    "cutoff = work['day'].max() - 3\n"
    "models = {key: fit_series(g, PARAMS, cutoff) for key, g in work.groupby('store')}\n"
    "print('MODELS', round(sum(models.values()), 6))"
)


def test_fixing_one_group_refits_only_it_when_settings_are_passed_by_name(nb_runner):
    nb_runner.create_notebook(["import cash\n%cash_on", STORES, WORK.format(fix="[]"), FIT])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "MODELS 84.0" in nb_runner.get_output(4), nb_runner.get_output(4)
    assert _fits(nb_runner) == 6

    nb_runner.set_cell_source(3, WORK.format(fix="[2]"))
    nb_runner.run_all()

    assert "MODELS 86.5" in nb_runner.get_output(4), nb_runner.get_output(4)
    # Measured before: 6, every store.
    assert _fits(nb_runner) - 6 == 1


def test_a_setting_passed_by_name_still_refits_when_its_value_changes(nb_runner):
    nb_runner.create_notebook(["import cash\n%cash_on", STORES, WORK.format(fix="[]"), FIT])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert _fits(nb_runner) == 6

    nb_runner.set_cell_source(4, FIT.replace("'alpha': 0.5", "'alpha': 1.0"))
    nb_runner.run_cell(4)

    assert "MODELS 168.0" in nb_runner.get_output(4), nb_runner.get_output(4)
    assert _fits(nb_runner) - 6 == 6


# Another swept windows in a loop -- `for win in WINDOWS: sc = {mid:
# fit_score(make_features(cleaned[mid], win)) ...}` -- then scored every machine
# with the chosen window in the next cell, `{... make_features(cleaned[mid],
# BEST_WIN) ...}`: all 200 fits ran again. The sweep's keys carried the loop's
# variable; the pick, outside any loop, had none to match.
PICK = (
    "BEST_W = 5\n"
    "best = {mid: fit_score(make_features(cleaned[mid], BEST_W)) for mid in sorted(cleaned)}\n"
    "print('BEST', round(sum(best.values()), 6))"
)


def test_the_chosen_setting_is_served_from_the_sweep(nb_runner):
    nb_runner.create_notebook(_cells() + [PICK])
    nb_runner.start_kernel()
    nb_runner.run_all()

    assert "BEST" in nb_runner.get_output(6), nb_runner.get_output(6)
    assert _fits(nb_runner) == 12, "the pick re-fitted what the sweep had just fitted"
