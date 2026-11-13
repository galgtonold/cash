"""A cell run again after an attempt that raised before it rebound its
accumulator goes on from what the accumulator holds, as plain IPython does.

``acc = acc + new`` in a loop raised part way, so the next run continues
from what it left; an attempt that then raised before the loop rebound
``acc`` left it as it found it. cash took the next run for a re-run from
the object that attempt started from, and put back the empty list the cell
above makes: the values the runs before had built were lost.
"""

import pytest
from nbclient.exceptions import CellExecutionError

pytestmark = [pytest.mark.integration, pytest.mark.timeout(240)]

_ABOVE = "acc = []\nn = 3000"
_BODY = "for i in range(n):\n    new = [i * {f}]\n    acc = acc + new"
_PROBE = "print('len', len(acc))"
#: Raises at the last pass; raises before rebinding; runs through.
_ATTEMPTS = ["(1 if i < n - 1 else None)", "factor", "1"]


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


def test_the_values_built_before_the_failed_attempt_are_kept(nb_runner):
    plain = _run(nb_runner, with_cash=False)
    assert "len 5999" in plain
    assert "len 5999" in _run(nb_runner, with_cash=True)
