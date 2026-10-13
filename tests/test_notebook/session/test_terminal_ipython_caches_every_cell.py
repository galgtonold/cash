"""Terminal IPython runs like a headless session: notebook not found, cells cached.

With ipykernel installed, ``get_notebook_path`` asked
``ipykernel.get_connection_file()`` for the kernel id. In terminal IPython the
running app is a ``TerminalIPythonApp``, so ``IPKernelApp.instance()`` raised
traitlets' ``MultipleInstanceError`` ("An incompatible sibling of 'IPKernelApp'
is already instantiated as singleton"). That was not in the caught tuple, so
every cell hit NOTEBOOK-BAILOUT and nothing was cached.

The test runs a real ``ipython`` process, since only a real
``TerminalIPythonApp`` holds the singleton that makes ipykernel refuse.
"""

from __future__ import annotations

import pytest

from tests._scripts import run_python

pytest.importorskip("ipykernel")

SCRIPT = """\
ip = get_ipython()
ip.run_cell("import cash")
ip.run_cell("%cash_on")
ip.run_cell("%cash_badge print")
ip.run_cell("import time\\ndef f():\\n    time.sleep(0.3)\\n    return list(range(10))")
ip.run_cell("x = f()")
ip.run_cell("x = f()")
ip.run_cell("print('VALUE', x[:3])")
"""


@pytest.mark.timeout(120)
def test_terminal_ipython_caches_cells_and_reports_notebook_not_found(tmp_path):
    (tmp_path / "script.ipy.py").write_text(SCRIPT, encoding="utf-8")
    done = run_python(
        "-m",
        "IPython",
        "--no-banner",
        "--colors=NoColor",
        "script.ipy.py",
        cwd=tmp_path,
        env={"IPYTHONDIR": str(tmp_path / "ipython")},
    )
    out = done.stdout + done.stderr
    assert "NOTEBOOK-BAILOUT" not in out, out
    assert "MultipleInstanceError" not in out, out
    assert "NOTEBOOK-NOT-FOUND" in out, out
    assert "CACHED: x = f()" in out, out
    assert "VALUE [0, 1, 2]" in out, out
