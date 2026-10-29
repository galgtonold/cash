"""A frame restored from the RAM tier still holds the very object another
variable holds, so a later change through that variable shows in the frame.

``f = pd.DataFrame({'o': [[t], [t]]})`` stores ``t`` with ``f`` (it holds
an object of ``f``'s). The RAM tier copied the frame's cells through pickle
on their own, and ``t`` apart from them: the restored ``f`` held a copy of
``t``, and ``t.tag = 2`` no longer reached it. A pandas ``Timestamp`` was
taken for a value, its object left out of the check: ``t.tag = 1`` set
after the frame was restored never reached the frame either.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pandas")

pytestmark = [pytest.mark.integration, pytest.mark.fresh_kernel, pytest.mark.timeout(600)]

FRAME = "f = pd.DataFrame({'o': [list(pd.date_range('2020-01-01', periods=50)) + [t], [t]]})"
READ = "time.sleep(0.3)\n{name} = [getattr(x, 'tag', None) for l in f['o'] for x in l][-3:]\nprint({name})"


def _plain(cells: list[str]) -> object:
    ns: dict = {}
    for cell in cells:
        exec(cell, ns)  # the notebook's own cells, as plain Python runs them
    return ns["before"], ns["after"]


@pytest.mark.parametrize(
    "made, change",
    [
        ("t = types.SimpleNamespace(tag=1)", "t.tag = 2"),
        ("t = pd.Timestamp('2021-01-01')", "t.tag = 1"),
        ("t = pd.Timestamp('2021-01-01'); t.tag = 1", "t.tag = t.tag + 1"),
    ],
    ids=["an object", "a timestamp given an attribute after the restore", "a timestamp with an attribute"],
)
def test_run_all_twice_matches_plain_python(nb_runner, made, change):
    cells = [
        f"import time, types\nimport pandas as pd\n{made}\n{FRAME}",
        READ.format(name="before"),
        change,
        READ.format(name="after"),
    ]
    expected = _plain(cells)
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    for _ in range(2):
        for number in range(1, len(cells) + 1):
            nb_runner.run_cell(number)
        assert (nb_runner.peek("before"), nb_runner.peek("after")) == tuple(map(repr, expected))


def test_an_attribute_set_after_a_restore_from_before_says_so(nb_runner):
    """Restored after a restart from an entry stored while timestamps were
    values, the frame may hold a copy of ``t``: the first attribute set on
    one says the frame may not show it until it is built again."""
    cells = [
        "import time, types\nimport pandas as pd\nt = pd.Timestamp('2021-01-01')\n"
        "f = (time.sleep(0.3), pd.DataFrame({'o': [[t], [t]]}))[1]",
        READ.format(name="before"),
        "t.tag = 1",
    ]
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    nb_runner.run_cell(1)
    nb_runner.run_cell(2)
    nb_runner.restart()
    nb_runner._init_cash()  # a restart leaves the kernel without cash
    for number in range(1, len(cells) + 1):
        nb_runner.run_cell(number)
    warned = nb_runner.get_raw_output(3)
    assert "NOTEBOOK-SHARE-UNCHECKED" in warned
    assert "'f'" in warned and "Run All" in warned
    assert "NOTEBOOK-SHARE-UNCHECKED" not in nb_runner.get_raw_output(1)
