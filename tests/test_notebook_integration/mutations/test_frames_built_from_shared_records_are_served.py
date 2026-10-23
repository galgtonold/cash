"""Frames built from a list of records, and the records themselves, are
served on the next Run All in a real kernel.

``df = pd.DataFrame({'rec': recs})`` holds the very dicts ``recs`` holds,
and ``flat = pd.json_normalize(recs)`` the very ``tags`` lists. Each was
refused as shared with "another variable or container" and recomputed on
every Run All, with no reason in the cell's metrics: the frame holding the
records in its object column did not join the others (it was linked, and
the statement refused), and two frames read together counted each record
once short. The mock shell served ``df`` all along; only a kernel, whose
own history holds objects too, showed the rest.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

SLOW = "time.sleep(0.2)"
CELLS = [
    "import cash\n%cash_on\nimport time\nimport pandas as pd",
    "recs = [{'id': i, 'tags': ['a', 'b'][: i % 3], 'meta': {'k': i % 7, 'v': i * 1.5}} for i in range(5000)]",
    f"df = ({SLOW}, pd.DataFrame({{'id': range(5000), 'rec': recs}}))[1]",
    f"flat = ({SLOW}, pd.json_normalize(recs))[1]",
    f"df['k'] = ({SLOW}, df['rec'].apply(lambda d: d['meta']['k']))[1]",
    f"df['ntags'] = ({SLOW}, df['rec'].map(lambda d: len(d['tags'])))[1]",
    "print('R', flat.shape, int(df.k.sum()), int(df.ntags.sum()), df['rec'][3] is recs[3])",
]
STATUSES = (
    "[str(s.get('status')).split('.')[-1] for s in "
    "get_ipython().magics_manager.registry['CashMagics']._last_cell_metrics.get('statements', [])]"
)


def test_the_frames_are_served_and_still_hold_the_records(nb_runner):
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    first = nb_runner.get_output(len(CELLS))
    assert "R (5000, 4) 14995 4999 True" in first, nb_runner.get_raw_output(len(CELLS))

    served = {}
    for number in range(1, len(CELLS) + 1):
        nb_runner.run_cell(number)
        served[number] = nb_runner.peek(STATUSES)
    for number in (3, 4, 5, 6):
        assert served[number] == "['RESTORED']", (CELLS[number - 1], served)
    assert nb_runner.get_output(len(CELLS)) == first
