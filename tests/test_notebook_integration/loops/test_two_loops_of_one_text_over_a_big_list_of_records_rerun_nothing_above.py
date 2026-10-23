"""Two loops of one text over a big list of records re-run nothing above.

``for p in points: total += p.x`` over 200,000 records, then a tuple over
the same records, a change through ``p``, and the same loop again (a
comment apart). The share check of the first loop is told at C speed that
each record is held by the list alone, so it finds ``p`` shares with
``points`` only; the second finds the tuple too and gives up, linking
``total`` as well. The edges were recorded by the loop's text, so the
simulation of the first loop took the second's: ``p.x = 4343`` moved
``total`` there and not at run time, and every Run All re-ran all eight
statements above the last cell. A structure's edges are now read where the
lineages it read match, as its outcome is.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.loops, pytest.mark.timeout(600)]

CELLS = [
    "%cash_badge print\nclass P:\n    def __init__(s, i):\n        s.x = i\npoints = [P(i) for i in range(200000)]",
    "total = 0\nfor p in points:\n    total += p.x",
    "tp = tuple(points)",
    "p = tp[150000]",
    "p.x = 4343",
    "total = 0\nfor p in points:\n    total += p.x  # 2",
    "n = len(points)",
]


def test_the_last_cell_reruns_nothing_above(nb_runner):
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    outputs = []
    for _ in range(3):
        nb_runner.run_all()
        outputs.append(nb_runner.get_raw_output(7))
    nb_runner.restart()
    for _ in range(2):
        nb_runner.run_all()
        outputs.append(nb_runner.get_raw_output(7))
    assert not [out for out in outputs if "Upstream" in out], outputs
    assert nb_runner.peek("total") == str(sum(range(200000)) - 150000 + 4343)


def test_the_first_loop_is_checked_within_its_budget(nb_runner):
    """The record step runs in a kernel: the first loop's share check ends
    with the holder it found, not with its budget run out."""
    nb_runner.create_notebook(CELLS[:2])
    nb_runner.start_kernel()
    nb_runner.run_all()
    raw = nb_runner.get_raw_output(2)
    assert "holds an object another variable or container also holds" in raw, raw
    assert "would take longer" not in raw, raw
