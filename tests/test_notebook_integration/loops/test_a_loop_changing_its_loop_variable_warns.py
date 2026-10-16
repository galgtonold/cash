"""A loop body that changes its loop variable in place before work cash
caches for each pass warns KEY-LOOP-VAR-CHANGED, naming the cell and line.

Each pass is keyed on the loop variable as the ``for`` binds it: two passes
that were equal then share one cached result, so the second gets the first's.
"""

import pytest

pytestmark = [pytest.mark.timeout(300)]

SETUP = "import time\ndef slow(q):\n    time.sleep(0.15)\n    return list(q)"


def test_the_loop_warns_once_with_its_cell_and_line(nb_runner):
    loop = (
        "accm = []\n"
        "for q in [[1], [1]]:\n"
        "    q.append(len(accm))  # equal when bound, different now\n"
        "    y = slow(q)\n"
        "    accm.append(y)\n"
        "print(accm)"
    )
    nb_runner.create_notebook([SETUP, loop])
    nb_runner.start_kernel()
    nb_runner.run_all()
    raw = nb_runner.get_raw_output(2)
    assert "KEY-LOOP-VAR-CHANGED" in raw
    assert "cell 2, line 3" in raw and "'q'" in raw
    nb_runner.run_cell(2)
    assert "KEY-LOOP-VAR-CHANGED" not in nb_runner.get_raw_output(2)


def test_an_ordinary_loop_is_quiet(nb_runner):
    """Cell 2 is an ordinary loop; cell 3, changing its loop variable, is the
    positive control."""
    loop = "accm = []\nfor q in [[1], [2]]:\n    y = slow(q)\n    accm.append(y)\nprint(accm)"
    changing = "acc2 = []\nfor q in [[1], [1]]:\n    q.append(len(acc2))\n    y = slow(q)\n    acc2.append(y)"
    nb_runner.create_notebook([SETUP, loop, changing])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "KEY-LOOP-VAR-CHANGED" not in nb_runner.get_raw_output(2)
    assert nb_runner.peek("accm") == "[[1], [2]]"
    assert "KEY-LOOP-VAR-CHANGED" in nb_runner.get_raw_output(3)
