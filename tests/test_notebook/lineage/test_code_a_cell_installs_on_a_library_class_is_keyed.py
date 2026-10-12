"""Code a cell installs on a library class is part of the key of a
statement that reaches it by attribute.

A pandas accessor defined in a cell (``@register_dataframe_accessor("fx")``)
or a function patched onto ``pd.DataFrame``: ``res = df.fx.slow()`` names
neither, so editing that code and running its cell again served the old
result. The key now holds the code found under the attribute names the
statement and the functions it calls read, and the globals that code reads.
"""

import time

import pytest

from tests._cell_driver import run_cash_cell

pytest.importorskip("pandas")

#: Long enough that the statement is stored.
SLOW = 0.6
DATA = "import time\nimport numpy as np, pandas as pd\ndf = pd.DataFrame({'a': np.arange(5.0)})\nK = 2"
ACCESSOR = (
    "@pd.api.extensions.register_dataframe_accessor('fx_{tag}')\n"
    "class FX:\n"
    "    def __init__(self, d):\n"
    "        self._d = d\n"
    "    def slow(self):\n"
    f"        time.sleep({SLOW})\n"
    "        return self._d * {k}\n"
)
PATCHED = f"def _slow(self):\n    time.sleep({SLOW})\n    return self * {{k}}\npd.DataFrame.slow_{{tag}} = _slow"


@pytest.fixture(autouse=True)
def _uninstall():
    """What a test installs on pandas' classes is gone after it."""
    import pandas as pd

    before = set(vars(pd.DataFrame))
    yield
    for name in set(vars(pd.DataFrame)) - before:
        delattr(pd.DataFrame, name)


def _run(magics, cells, start=0):
    for cell in cells[start:]:
        run_cash_cell(magics, cell, cells=cells)


@pytest.mark.parametrize(
    ("installer", "use"),
    [
        (ACCESSOR, "res = df.fx_{tag}.slow()"),
        (PATCHED, "res = df.slow_{tag}()"),
        (ACCESSOR, "def total(d):\n    return d.fx_{tag}.slow()\nres = total(df)"),
    ],
    ids=["accessor", "patched-method", "through-a-function"],
)
def test_an_edit_to_the_installed_code_runs_the_statement_again(cash_magics, mock_shell, request, installer, use):
    tag = request.node.callspec.id.replace("-", "_")
    cells = [DATA, installer.format(k=2, tag=tag), use.format(tag=tag)]
    _run(cash_magics, cells)
    assert mock_shell.user_ns["res"].a.tolist() == [0.0, 2.0, 4.0, 6.0, 8.0]

    cells[1] = installer.format(k=3, tag=tag)
    _run(cash_magics, cells, start=1)
    assert mock_shell.user_ns["res"].a.tolist() == [0.0, 3.0, 6.0, 9.0, 12.0]


def test_a_global_the_installed_code_reads_is_keyed(cash_magics, mock_shell):
    cells = [DATA, ACCESSOR.format(k="K", tag="glob"), "res = df.fx_glob.slow()"]
    _run(cash_magics, cells)
    assert mock_shell.user_ns["res"].a.tolist() == [0.0, 2.0, 4.0, 6.0, 8.0]

    cells[0] = DATA.replace("K = 2", "K = 3")
    _run(cash_magics, cells)
    assert mock_shell.user_ns["res"].a.tolist() == [0.0, 3.0, 6.0, 9.0, 12.0]


def test_the_statement_is_still_served_while_the_code_is_the_same(cash_magics, mock_shell):
    """Positive control: running the accessor cell again unchanged keeps
    the statement's entry."""
    cells = [DATA, ACCESSOR.format(k=2, tag="same"), "res = df.fx_same.slow()"]
    _run(cash_magics, cells)
    started = time.perf_counter()
    _run(cash_magics, cells, start=1)
    assert time.perf_counter() - started < SLOW
    assert mock_shell.user_ns["res"].a.tolist() == [0.0, 2.0, 4.0, 6.0, 8.0]
