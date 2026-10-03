"""An edited module that no longer loads fails the cell, as a fresh import would.

cash reloads an edited project module before a cell runs. When the edit
does not compile, the reload raises and the kernel keeps the module's old
code: running the cell then runs that old code as if the edit had been
picked up. The cell must fail with the module's ``SyntaxError`` (its file and
line), run none of its statements, keep failing until the file loads, and run
the new code once it does.
"""

import pytest
from nbclient.exceptions import CellExecutionError

pytestmark = [pytest.mark.integration, pytest.mark.modules, pytest.mark.timeout(120)]

MODULE = """\
import os

import cash


@cash.cache
def weekly(n):
    fd = os.open(os.path.join(os.path.dirname(__file__), "runs.log"), os.O_WRONLY | os.O_APPEND | os.O_CREAT)
    os.write(fd, b"x")
    os.close(fd)
    return n - {offset}


def write_report(v):
    return f"report={{v}}"
"""

BROKEN = MODULE.replace("return n - {offset}", "return n - ({offset}")


def _runs(tmp_path):
    log = tmp_path / "runs.log"
    return len(log.read_text(encoding="utf-8")) if log.exists() else 0


def test_a_module_that_no_longer_compiles_fails_every_cell_until_it_does(nb_runner, tmp_path):
    module = tmp_path / "replen.py"
    module.write_text(MODULE.format(offset=7), encoding="utf-8")
    nb_runner.create_notebook(
        [
            "import cash\nimport replen",
            "v = replen.weekly(10)\nprint(replen.write_report(v))",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "report=3" in nb_runner.get_output(2)
    assert _runs(tmp_path) == 1

    # The edit leaves an unclosed paren, and the cell now asks for a call
    # the old code has never run: running it would show up in the log.
    module.write_text(BROKEN.format(offset=8), encoding="utf-8")
    nb_runner.set_cell_source(2, "v = replen.weekly(20)\nprint(replen.write_report(v))")

    for _ in range(2):  # the second run must not take the edit as seen
        with pytest.raises(CellExecutionError):
            nb_runner.run_cell(2)
        error = next(o for o in nb_runner.get_cell(2).outputs if o.get("output_type") == "error")
        assert error["ename"] == "SyntaxError"
        shown = "\n".join(error["traceback"])
        assert "replen.py" in shown
        assert "return n - (8" in shown
        assert "report=" not in nb_runner.get_output(2)
        assert _runs(tmp_path) == 1

    # Re-running the import cell fails the same way a fresh kernel's would.
    with pytest.raises(CellExecutionError):
        nb_runner.run_cell(1)
    assert any(o.get("ename") == "SyntaxError" for o in nb_runner.get_cell(1).outputs)

    # A cell of cash magics still runs, so cash can be switched off.
    nb_runner.add_cell("%cash_status", save=True)
    nb_runner.run_cell(3)

    module.write_text(MODULE.format(offset=9), encoding="utf-8")
    nb_runner.run_cell(2)
    assert "report=11" in nb_runner.get_output(2)
    assert nb_runner.peek("v") == "11"
    assert _runs(tmp_path) == 2
