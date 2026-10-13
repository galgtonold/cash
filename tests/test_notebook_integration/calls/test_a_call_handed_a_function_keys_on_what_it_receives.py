"""A call handed a module's function is keyed on what it receives.

``ets_bt = {s: fu.backtest(daily[s], fu.fit_ets, cuts, H) for s in store_ids}``
refit every store after the list widened or one store's history was fixed:
handed a function, the call kept the statement and the lineage of ``daily``
in its key. The unit arm is
``tests/test_notebook/calls/test_a_call_handed_a_function_keys_on_what_it_receives.py``.
"""

from pathlib import Path

import pytest

pytest.importorskip("pandas")

pytestmark = [pytest.mark.integration, pytest.mark.timeout(180)]

MODULE = (
    "import os, time\n"
    "def fit_a(y, h):\n"
    "    return float(y.iloc[-1]) * h\n"
    "def fit_b(y, h):\n"
    "    return float(y.iloc[0]) * h\n"
    "def backtest(y, fit_fn, cuts, h):\n"
    "    fd = os.open('fits.log', os.O_WRONLY | os.O_CREAT | os.O_APPEND)  # @cash:assume-safe\n"
    "    os.write(fd, b'x')\n"
    "    os.close(fd)\n"
    "    time.sleep(0.3)\n"
    "    return [fit_fn(y.loc[:c], h) for c in cuts]\n"
)


def _cells(fix, stores):
    return [
        "import cash\n%cash_on",
        "import numpy as np\nimport pandas as pd\nimport fcutils as fu",
        "idx = pd.date_range('2024-01-01', periods=60, freq='D')\n"
        "daily = pd.DataFrame({s: np.arange(60.0) * s for s in range(1, 6)}, index=idx)\n"
        f"daily.loc[idx[5], 3] = {fix}.0",
        "cuts = [daily.index[k] for k in (30, 40, 50)]\nH = 7\nstore_ids = list(daily.columns)",
        f"bt = {{s: fu.backtest(daily[s], fu.fit_a, cuts, H) for s in {stores}}}\nprint(sorted(bt))",
    ]


def _fits(runner) -> int:
    log = Path(runner.work_dir) / "fits.log"
    return len(log.read_bytes()) if log.exists() else 0


def test_a_widened_comprehension_and_a_fixed_store_rerun_only_what_changed(nb_runner):
    (Path(nb_runner.work_dir) / "fcutils.py").write_text(MODULE, encoding="utf-8")
    nb_runner.create_notebook(_cells(0, "store_ids[:2]"))
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert _fits(nb_runner) == 2

    for n, source in enumerate(_cells(0, "store_ids"), start=1):
        nb_runner.set_cell_source(n, source)
    nb_runner.run_all()
    assert "[1, 2, 3, 4, 5]" in nb_runner.get_output(5)
    assert _fits(nb_runner) == 5, "the stores already done were fitted again"

    nb_runner.set_cell_source(3, _cells(1, "store_ids")[2])
    nb_runner.run_all()
    assert _fits(nb_runner) == 6, "stores whose history did not change were fitted again"
    assert nb_runner.peek("bt[3][0]") == nb_runner.peek("fu.fit_a(daily[3].loc[:cuts[0]], 7)")
