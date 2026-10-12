"""A loop over what a function of the user's whose body cannot be read
returns (a lambda, one made by ``exec``), changing the items in place.

``for r in active(): r['v'] = r['v'] * rate`` changes the records a global
holds, and nothing says which global. Every variable the call could reach
counts as changed, so after ``rate`` is edited the cell reading ``records``
sees the new values.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

HEADERS = {
    "lambda": "active = lambda: [r for r in records if r['on']]",
    "exec": "exec('def active():\\n    return [r for r in records if r[\"on\"]]')",
}
SLOW = "import time\ndef slow(v):\n    time.sleep(0.25)\n    return v\n"


@pytest.mark.parametrize("case", list(HEADERS))
def test_an_edit_above_reaches_the_reader(nb_runner, case):
    first = SLOW + "rate = 2\n" + HEADERS[case]
    cells = [
        first,
        "records = [{'on': True, 'v': 1}, {'on': False, 'v': 2}, {'on': True, 'v': 3}]",
        "for r in active():\n    r['v'] = r['v'] * rate",
        "out = slow([r['v'] for r in records])\nprint(out)",
    ]
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert nb_runner.get_output(4).strip() == "[2, 2, 6]"
    nb_runner.set_cell_source(1, first.replace("rate = 2", "rate = 3"))
    nb_runner.run_all()
    assert nb_runner.get_output(4).strip() == "[3, 2, 9]"
