"""A statement calling a local module's function is keyed on the environment that function reads.

``m = mylib.mode()`` with ``mode`` reading ``os.environ["MODE"]``: the
statement's own text reads no environment, so editing the cell that sets
``MODE`` and running the notebook again served the old mode. The functions
of the user's own modules a statement calls are now read for the variables
they read, as the decorator reads a cached function's helpers.
"""

import os
import sys

import pytest

from cash.notebook.lineage_formula import statement_environment_reads
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

LIB = (
    "import os, time\n"
    "def _mode():\n    return os.environ.get('CASH_UT_MODE', 'none')\n"
    f"def mode():\n    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n    return _mode()\n"
)


@pytest.fixture
def env_module(tmp_path, monkeypatch):
    name = f"_envlib_{os.getpid()}_{id(tmp_path)}"
    (tmp_path / f"{name}.py").write_text(LIB, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delenv("CASH_UT_MODE", raising=False)
    yield name
    sys.modules.pop(name, None)


@pytest.mark.parametrize("call", ["{m}.mode()", "mode()"], ids=["module_attribute", "from_import"])
def test_editing_the_variable_and_running_again_recomputes(cash_magics, mock_shell, env_module, call):
    imports = f"import os\nimport {env_module}\nfrom {env_module} import mode"
    cells = [imports, "os.environ['CASH_UT_MODE'] = 'train'", "m = " + call.format(m=env_module)]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert mock_shell.user_ns["m"] == "train"

    cells[1] = "os.environ['CASH_UT_MODE'] = 'test'"
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert mock_shell.user_ns["m"] == "test"


def test_a_helper_two_calls_down_is_read(env_module):
    module = __import__(env_module)
    reads = statement_environment_reads("m = lib.mode()", {"lib": module})
    assert ("env", "CASH_UT_MODE") in reads


def test_a_statement_calling_no_function_of_the_users_reads_nothing_more(env_module):
    assert statement_environment_reads("n = len(os.sep)", {"os": os, "len": len}) == set()
