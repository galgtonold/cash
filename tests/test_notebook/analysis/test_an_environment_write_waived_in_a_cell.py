"""An environment write a function defined in a cell waives is waived.

``os.putenv`` and ``os.environ[k] = v`` change the environment, which no
variable holds, so a call that makes them is not stored: a hit would not
set it again (`cash.tracking.process_registries`). Inside ``with
cash.assume_safe():``, or on a line marked ``# @cash:assume-safe``, the
user said a hit may skip it, as for a file write: the call is stored.
Both for a ``@cash.cache`` function and for a plain one the notebook
caches the call of. The same write outside the block still keeps the call
from being stored.
"""

from __future__ import annotations

import os
import warnings

import pytest

pytest.importorskip("IPython")

from cash import Cash
from cash.notebook.ipython.magics import CashMagics
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

_KEY = "CASH_TEST_CELL_ENV_WAIVER"

#: How each write is spelled, waived by the block or the comment, and not.
_WRITES = {
    "putenv-block": f"    with cash.assume_safe():\n        os.putenv({_KEY!r}, str(v))\n",
    "environ-block": f"    with cash.assume_safe():\n        os.environ[{_KEY!r}] = str(v)\n",
    "putenv-comment": f"    os.putenv({_KEY!r}, str(v))  # @cash:assume-safe\n",
    "environ-comment": f"    os.environ[{_KEY!r}] = str(v)  # @cash:assume-safe\n",
}
_UNWAIVED = {
    "putenv": f"    with cash.assume_safe():\n        pass\n    os.putenv({_KEY!r}, str(v))\n",
    "environ": f"    with cash.assume_safe():\n        pass\n    os.environ[{_KEY!r}] = str(v)\n",
}


@pytest.fixture
def cell_runner(mock_shell, tmp_path, monkeypatch):
    """The real magic over a disk cache, with that Cash bound to ``c``;
    returns the codes of the warnings a cell raised."""
    import cash

    monkeypatch.setenv(_KEY, "outer")
    c = Cash(cache_dir=str(tmp_path / "cache"), register_magic=False)
    magics = CashMagics(mock_shell, c)
    magics.cash_on("")
    magics.badges.mode = "off"
    mock_shell.user_ns.update(c=c, cash=cash)

    def run(cell: str) -> list[str]:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            run_cash_cell(magics, cell)
        return [code for w in caught if (code := getattr(w.message, "code", None))]

    yield run
    c.backend.clear()
    os.unsetenv(_KEY)


def _function_cell(decorated: bool, write: str, counter) -> str:
    """``setenv(v)`` makes *write*, and counts its runs in *counter* (a
    waived append); it takes long enough to be worth storing."""
    return (
        "import os, time\n"
        + ("@c.cache\n" if decorated else "")
        + "def setenv(v):\n"
        + f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
        + write
        + f"    fd = os.open({str(counter)!r}, os.O_WRONLY | os.O_APPEND | os.O_CREAT)  # @cash:assume-safe\n"
        + "    os.write(fd, b'x')\n"
        + "    os.close(fd)\n"
        + "    return v * 2\n"
    )


@pytest.mark.parametrize("decorated", [True, False], ids=["decorated", "plain"])
@pytest.mark.parametrize("write", sorted(_WRITES))
def test_a_waived_environment_write_is_cached(cell_runner, tmp_path, decorated, write):
    counter = tmp_path / "runs"
    cell_runner(_function_cell(decorated, _WRITES[write], counter))
    codes = []
    for _ in range(2):
        codes += cell_runner("r = setenv(7)\n")
    assert counter.read_bytes() == b"x"
    assert "IMPURE-OBSERVED-EFFECTS" not in codes
    assert os.environ[_KEY] == "7" if write.startswith("environ") else os.environ[_KEY] == "outer"


@pytest.mark.parametrize("decorated", [True, False], ids=["decorated", "plain"])
@pytest.mark.parametrize("write", sorted(_UNWAIVED))
def test_the_same_write_outside_the_block_runs_every_time(cell_runner, tmp_path, decorated, write):
    counter = tmp_path / "runs"
    cell_runner(_function_cell(decorated, _UNWAIVED[write], counter))
    codes = []
    for _ in range(2):
        codes += cell_runner("r = setenv(7)\n")
    assert counter.read_bytes() == b"xx"
    if decorated:
        assert "IMPURE-OBSERVED-EFFECTS" in codes
