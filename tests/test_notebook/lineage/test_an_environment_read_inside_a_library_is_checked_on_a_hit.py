"""A statement is checked on the environment an installed package read.

``x = tenantlib.rows(3)``, where the package reads ``os.environ["TENANT"]``:
the analyzer does not read installed code, so nothing keyed the variable and
the first tenant's rows were served to every other tenant. A run now notes
the variables installed code read while it ran, and its entry (and the
cached call's inside it) keeps a digest of each value: a lookup serves it
only while they all match.
"""

import pytest

from cash.notebook.cache_status import CacheStatus
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

LIB = (
    "import os, time\n"
    "def rows(n):\n"
    f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
    "    return f\"{os.environ.get('CASH_UT_TENANT', 'none')}:{n}\"\n"
)


@pytest.fixture
def tenantlib(tmp_path, monkeypatch):
    site = tmp_path / "venv" / "lib" / "site-packages"
    (site / "cash_ut_tenantlib").mkdir(parents=True)
    (site / "cash_ut_tenantlib" / "__init__.py").write_text(LIB, encoding="utf-8")
    monkeypatch.syspath_prepend(str(site))
    monkeypatch.setenv("CASH_UT_TENANT", "acme")


def test_another_value_recomputes(cash_magics, mock_shell, tenantlib, monkeypatch):
    cells = ["import cash_ut_tenantlib", "x = cash_ut_tenantlib.rows(3)"]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert mock_shell.user_ns["x"] == "acme:3"
    run_cash_cell(cash_magics, cells[1], cells=cells)
    assert mock_shell.user_ns["x"] == "acme:3"
    assert _status(cash_magics) == CacheStatus.RESTORED, "the same value hits"

    monkeypatch.setenv("CASH_UT_TENANT", "globex")
    run_cash_cell(cash_magics, cells[1], cells=cells)
    assert mock_shell.user_ns["x"] == "globex:3", "another tenant was served acme's rows"
    assert _status(cash_magics) != CacheStatus.RESTORED


def _status(magics) -> str:
    rows = [m for m in magics.cash_status("dict")["last_cell"]["statements"] if m.get("code")]
    return rows[-1].get("status")
