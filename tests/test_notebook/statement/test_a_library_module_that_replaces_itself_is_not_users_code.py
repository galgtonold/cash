"""A library module that puts another object in its place in ``sys.modules``
is still the library's code to the trace check, so a statement using it is
served.

``torch._VF`` replaces itself with a ``ModuleType`` subclass that has no
file. The recorder judged the code by the module ``sys.modules`` named,
found no file, and took it for the user's (as a notebook's ``__main__`` is):
the trace kept the replacement's class, which a lookup by its name never
finds again, so every hit read as stale and ``optimizer =
torch.optim.SGD(...)`` ran every time with the trace on. The code is judged
by its own file first (`code_trace._discover`, `Recorder._classify`).
"""

from __future__ import annotations

import importlib
import sys

import pytest

from cash.notebook.cache_status import CacheStatus
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

pytestmark = pytest.mark.skipif(sys.version_info < (3, 12), reason="the trace check needs sys.monitoring")

REPLACES_ITSELF = """import sys
import types


class VFModule(types.ModuleType):
    def __init__(self, name):
        super().__init__(name)
        self.vf = types.SimpleNamespace(double=lambda x: 2 * x)

    def __getattr__(self, attr):
        return getattr(self.vf, attr)


sys.modules[__name__] = VFModule(__name__)
"""


def _status(magics, code: str) -> str | None:
    statements = magics.cash_status("dict")["last_cell"]["statements"]
    return next((m.get("status") for m in statements if m.get("code") == code), None)


def test_a_statement_using_a_self_replacing_library_module_is_served(cash_magics, tmp_path, monkeypatch):
    lib = tmp_path / "site-packages"
    lib.mkdir()
    (lib / "replaces_itself_vf.py").write_text(REPLACES_ITSELF, encoding="utf-8")
    monkeypatch.syspath_prepend(str(lib))
    monkeypatch.delitem(sys.modules, "replaces_itself_vf", raising=False)
    # Imported before the notebook's first cell, as torch often is: its code
    # is first judged after the module put its replacement in place.
    importlib.import_module("replaces_itself_vf")
    cells = [
        "import time\nimport replaces_itself_vf",
        f"r = (time.sleep({ABOVE_PERSISTENCE_FLOOR_S}), replaces_itself_vf.double(21))[1]",
    ]
    for _ in range(2):
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)
    assert _status(cash_magics, cells[1]) == CacheStatus.RESTORED
    assert cash_magics.shell.user_ns["r"] == 42
