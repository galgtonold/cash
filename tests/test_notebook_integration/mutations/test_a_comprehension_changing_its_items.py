"""A comprehension or ``map`` that changes the items it walks changes the list.

``[r.update(tax=1) for r in records]`` and ``list(map(lambda r:
r.update(tax=1), records))`` change what ``records`` holds. After the
statement is edited and the notebook runs again, the cell reading
``records`` must not be served the old items. Compared with plain Python.
"""

import pytest

pytestmark = [pytest.mark.timeout(300)]

SETUP = "import time\ndef summarize(recs):\n    time.sleep(0.2)\n    return sorted(set(k for r in recs for k in r))"
SHAPES = {
    "a method on the comprehension's item": "[r.update(FIELD=1) for r in records]",
    "a lambda handed to map": "list(map(lambda r: r.update(FIELD=1), records))",
}


@pytest.mark.parametrize("shape", list(SHAPES))
def test_an_edit_reaches_the_reader(nb_runner, shape):
    statement = SHAPES[shape]
    nb_runner.create_notebook(
        [
            SETUP,
            "records = [{'p': 1}, {'p': 2}]",
            statement.replace("FIELD", "tax"),
            "fields = summarize(records)\nprint(fields)",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert nb_runner.get_output(4).strip() == "['p', 'tax']"
    nb_runner.set_cell_source(3, statement.replace("FIELD", "vat"))
    nb_runner.run_all()
    assert nb_runner.get_output(4).strip() == "['p', 'vat']"
