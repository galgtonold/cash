"""A cell run again after attempts that raised starts from what the cells
above give, as after a run that went through, so it ends at the value
Restart & Run All gives.

``acc = acc + new`` in a loop raised part way, then raised before the loop
rebound ``acc``, then ran through. cash went on from what the failed runs
left after one of them and started again after another, and could end at a
value neither a plain kernel (each run on top of the last) nor Run All
gives.
"""

import pytest
from nbclient.exceptions import CellExecutionError

pytestmark = [pytest.mark.integration, pytest.mark.timeout(240)]

_ABOVE = "acc = []\nn = 3000"
_BODY = "for i in range(n):\n    new = [i * {f}]\n    acc = acc + new"
_PROBE = "print('len', len(acc))"
#: Runs through; raises at the last pass; raises before rebinding; runs through.
_ATTEMPTS = ["1", "(1 if i < n - 1 else None)", "factor", "1"]


def _run(nb_runner, with_cash: bool) -> str:
    nb_runner.create_notebook([_ABOVE, _BODY.format(f=_ATTEMPTS[0]), _PROBE])
    nb_runner.start_kernel(with_cash=with_cash)
    nb_runner.run_cell(1)
    for f in _ATTEMPTS:
        nb_runner.set_cell_source(2, _BODY.format(f=f))
        try:
            nb_runner.run_cell(2)
        except CellExecutionError:
            pass
    nb_runner.run_cell(3)
    out = nb_runner.get_output(3)
    nb_runner.shutdown()
    return out


def test_the_last_run_starts_from_the_cells_above(nb_runner):
    assert "len 8999" in _run(nb_runner, with_cash=False)
    assert "len 3000" in _run(nb_runner, with_cash=True)
