"""A call handed a function (``backtest(y, fu.fit_ets, cuts, H)``) is keyed on
what it receives, as a call handed only data is.

A function argument is code, not data: its pickled bytes are its name. So such
a call was never keyed on what it receives, and kept the enclosing statement
and the lineage of every name its arguments read. In a demand-forecasting
notebook, ``ets_bt = {s: fu.backtest(daily[s], fu.fit_ets, cuts, HORIZON) for
s in store_ids[:4]}`` widened to ``store_ids`` refit the 4 stores already
done, and a data fix to one store refit all 16 (16 s), while ``fu.fit_ets(daily[s],
HORIZON)`` in another comprehension was served; the SARIMAX ``for`` loop over
the same call refit every store after the fix (49 s), since ``cuts`` was rebuilt
from the fixed frame with the same dates.

Keyed on what it receives, the function's name stays in the key as the
callee's does, so an edit to the function it names moves the key, and an edit
to another function of the module does not.
"""

from __future__ import annotations

import os
import sys
import warnings

import pytest

from tests._cell_driver import run_cash_cell

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

MODULE = """import os
import time
LOG = {log!r}
def fit_a(y, h):
    return float(y.iloc[-1]) * h
def fit_b(y, h):
    return float(y.iloc[0]) * h
def backtest(y, fit_fn, cuts, h):
    fd = os.open(LOG, os.O_WRONLY | os.O_APPEND | os.O_CREAT)  # @cash:assume-safe
    os.write(fd, b"x")
    os.close(fd)
    time.sleep(0.25)
    return [fit_fn(y.loc[:c], h) for c in cuts]
"""


@pytest.fixture
def forecast(tmp_path, monkeypatch):
    name = f"_handed_fit_{os.getpid()}_{id(tmp_path)}"
    log = tmp_path / "runs.log"
    path = tmp_path / f"{name}.py"
    path.write_text(MODULE.format(log=str(log)), encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))

    def runs() -> int:
        return len(log.read_bytes()) if log.exists() else 0

    def edit(old: str, new: str) -> None:
        text = path.read_text(encoding="utf-8")
        assert old in text
        path.write_text(text.replace(old, new), encoding="utf-8")
        stat = os.stat(path)
        os.utime(path, (stat.st_atime, stat.st_mtime + 5))

    yield name, runs, edit
    sys.modules.pop(name, None)


def _cells(module: str, fix: int, stores: str) -> list[str]:
    return [
        f"import numpy as np\nimport pandas as pd\nimport {module} as fu",
        "idx = pd.date_range('2024-01-01', periods=60, freq='D')\n"
        f"daily = pd.DataFrame({{s: np.arange(60.0) * s for s in range(1, 7)}}, index=idx)\n"
        f"daily.loc[idx[5], 3] = {fix}.0",
        "cuts = [daily.index[k] for k in (30, 40, 50)]\nH = 7\nstore_ids = list(daily.columns)",
        f"ets_bt = {{s: fu.backtest(daily[s], fu.fit_a, cuts, H) for s in {stores}}}",
    ]


def _run(magics, cells):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for cell in cells:
            run_cash_cell(magics, cell, cells=cells)


def test_a_comprehension_widened_reuses_the_calls_it_made(cash_magics, mock_shell, forecast):
    module, runs, _edit = forecast
    _run(cash_magics, _cells(module, 0, "store_ids[:2]"))
    assert runs() == 2

    _run(cash_magics, _cells(module, 0, "store_ids"))

    assert runs() == 6, "the 2 stores already done ran again"
    assert len(mock_shell.user_ns["ets_bt"]) == 6


def test_a_fix_to_one_store_reruns_that_store_only(cash_magics, mock_shell, forecast):
    module, runs, _edit = forecast
    _run(cash_magics, _cells(module, 0, "store_ids"))
    assert runs() == 6

    _run(cash_magics, _cells(module, 1, "store_ids"))

    assert runs() == 7, "stores whose history did not change ran again"
    # Control: the fixed store's new history did reach its result.
    assert mock_shell.user_ns["ets_bt"][3] == [
        mock_shell.user_ns["fu"].fit_a(mock_shell.user_ns["daily"][3].loc[:c], 7) for c in mock_shell.user_ns["cuts"]
    ]


def test_an_edit_to_the_handed_function_reruns_every_call(cash_magics, mock_shell, forecast):
    module, runs, edit = forecast
    cells = _cells(module, 0, "store_ids")
    _run(cash_magics, cells)
    assert runs() == 6

    edit("return float(y.iloc[-1]) * h", "return float(y.iloc[-1]) * h + 1000.0")
    _run(cash_magics, cells)

    assert runs() == 12, "an edit to the function the call was handed was not seen"
    assert mock_shell.user_ns["ets_bt"][1][0] > 1000


def test_an_edit_to_another_function_of_the_module_reruns_nothing(cash_magics, mock_shell, forecast):
    module, runs, edit = forecast
    cells = _cells(module, 0, "store_ids")
    _run(cash_magics, cells)
    assert runs() == 6

    edit("return float(y.iloc[0]) * h", "return float(y.iloc[0]) * h * 2")
    _run(cash_magics, cells)

    assert runs() == 6, "an edit to fit_b, which the call never reaches, re-ran it"


def test_an_edit_to_a_module_global_the_handed_function_reads_reruns_it(cash_magics, mock_shell, forecast):
    module, runs, edit = forecast
    edit("return float(y.iloc[-1]) * h", "return float(y.iloc[-1]) * h * SCALE")
    edit("LOG = ", "SCALE = 1.0\nLOG = ")
    cells = _cells(module, 0, "store_ids")
    _run(cash_magics, cells)
    assert runs() == 6

    edit("SCALE = 1.0", "SCALE = 2.0")
    _run(cash_magics, cells)

    assert runs() == 12, "an edit to a global the handed function reads was not seen"


def test_a_for_loop_over_the_call_reuses_it_after_a_fix_elsewhere(cash_magics, mock_shell, forecast):
    """``cuts`` is rebuilt from the fixed frame with the same dates: a list,
    which the value cutoff does not compare, so its lineage moved."""
    module, runs, _edit = forecast
    loop = "bt = {}\nfor s in store_ids:\n    bt[s] = fu.backtest(daily[s], fu.fit_a, cuts, H)"
    cells = [*_cells(module, 0, "store_ids")[:3], loop]
    _run(cash_magics, cells)
    assert runs() == 6

    cells = [*_cells(module, 1, "store_ids")[:3], loop]
    _run(cash_magics, cells)

    assert runs() == 7


@pytest.mark.parametrize(
    ("before", "after"),
    [
        (("def extra(v):\n    return v", "SCALE = 1.0"), ("def extra(v):\n    return v", "SCALE = 2.0")),
        (("def extra(v):\n    return v", "SCALE = 1.0"), ("def extra(v):\n    return v * 3", "SCALE = 1.0")),
    ],
    ids=["a-global-it-reads", "a-function-it-calls"],
)
def test_a_notebook_function_handed_over_runs_again_on_a_change_it_reaches(
    cash_magics, mock_shell, forecast, before, after
):
    module, runs, _edit = forecast
    fit = "def fit(y, h):\n    return extra(float(y.iloc[-1])) * h * SCALE"
    call = "bt = {s: fu.backtest(daily[s], fit, cuts, H) for s in store_ids}"
    base = _cells(module, 0, "store_ids")[:3]
    _run(cash_magics, [*base, *before, fit, call])
    first = dict(mock_shell.user_ns["bt"])
    assert runs() == 6

    _run(cash_magics, [*base, *after, fit, call])

    assert runs() == 12, "a change to what the handed function reaches was not seen"
    assert mock_shell.user_ns["bt"] != first
