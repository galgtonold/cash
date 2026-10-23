"""A frame built from a list of records is served on the next Run All in a
real kernel, still holding the records the list holds.

``df = pd.DataFrame({'rec': recs})`` holds the very dicts ``recs`` holds.
The mock shell served it, but in a kernel its share check ran past its
0.1 s budget over 5,000 records and it was recomputed on every Run All: the
function check, the library-object check and the holder patches each took a
Python step per record.

``flat = pd.json_normalize(recs)`` and ``df['k'] = ...`` are still refused:
``df`` holds the records only in its cells, which a restore cannot put back.
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


def test_the_frame_is_served_and_still_holds_the_records(nb_runner):
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    first = nb_runner.get_output(len(CELLS))
    assert "R (5000, 4) 14995 4999 True" in first, nb_runner.get_raw_output(len(CELLS))

    served = {}
    for number in range(1, len(CELLS) + 1):
        nb_runner.run_cell(number)
        served[number] = nb_runner.peek(STATUSES)
    assert served[3] == "['RESTORED']", (CELLS[2], served)
    assert nb_runner.get_output(len(CELLS)) == first
