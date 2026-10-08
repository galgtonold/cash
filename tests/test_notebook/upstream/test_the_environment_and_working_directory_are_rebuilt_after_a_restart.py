"""After a restart, a cell run alone runs in the environment and directory the cells above set.

``def setup(): os.environ["MODE"] = "b"`` and ``setup()`` above, ``x =
mylib.mode()`` below: after a kernel restart, running only the reader
rebuilt what it needs from the cells above and left ``setup()`` out, so the
variable was not set and the reader got the shell's value, silently. The
same for ``os.chdir(d)`` in a cell, or in a function of the user's module:
the reader read the file of the directory the kernel started in.

A restart puts the environment and the working directory back as the shell
started the kernel. The statements above that changed them when they ran,
itself or in a function they call, run again before the cell.
"""

from __future__ import annotations

import os
import sys

import pytest

from cash.backends.file_backend import FileBackend
from tests._cell_driver import run_cash_cell

VAR = "CASH_TEST_PROCESS_STATE_MODE"


@pytest.fixture
def clean_backend(tmp_path):
    """A file backend in place of the in-memory one: the record a later
    kernel reads is metadata alone, which only a file tier keeps."""
    backend = FileBackend(cache_dir=str(tmp_path / "cache"))
    yield backend
    backend.clear()


@pytest.fixture
def stores_everything(cash_instance):
    """Statements are stored however cheap, as on a slow machine: the cells
    above are then restored, not run, unless something says they must run."""
    cash_instance.config.min_execution_time_to_cache_seconds = 0.0
    cash_instance.config.call_cost_floor_seconds = 0.0


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    """A directory holding ``d.txt`` and ``sub/d.txt``, the kernel's working
    directory, and a module whose functions change the process."""
    (tmp_path / "d.txt").write_text("TOP", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "d.txt").write_text("SUB", encoding="utf-8")
    name = f"_process_lib_{os.getpid()}_{id(tmp_path)}"
    (tmp_path / f"{name}.py").write_text(
        "import os\n"
        "def go(d):\n    os.chdir(d)\n"
        f"def setmode(m):\n    os.environ[{VAR!r}] = m\n"
        f"def mode():\n    return os.environ.get({VAR!r}, 'unset')\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(VAR, raising=False)
    yield tmp_path, name
    sys.modules.pop(name, None)


def _restart(magics, workdir) -> None:
    """What a kernel restart leaves: no session state, no variables, the
    module not loaded, and the process as the shell started it."""
    path, name = workdir
    magics.tracking_state.reset_session_state()
    simulator = magics._upstream_checker.simulator
    simulator.cache.reset()
    simulator.probe.reset()
    user_ns = magics.shell.user_ns
    for var in [n for n in user_ns if not n.startswith("_") and n not in ("get_ipython", "exit", "quit")]:
        user_ns.pop(var, None)
    sys.modules.pop(name, None)
    os.environ.pop(VAR, None)
    os.chdir(path)


def _run_above_then_restart(magics, workdir, cells) -> None:
    """Run every cell but the reader, then restart: the reader is new, so
    nothing restores it and it runs where the rebuild leaves the process."""
    for cell in cells[:-1]:
        run_cash_cell(magics, cell, cells=cells)
    _restart(magics, workdir)


ENVIRONMENT = {
    "a_notebook_helper": ["import os\nimport {m}", "def setup():\n    os.environ[{v!r}] = 'b'", "setup()"],
    "a_helper_defined_and_called_in_one_cell": [
        "import os\nimport {m}",
        "def setup():\n    os.environ[{v!r}] = 'b'\nsetup()",
    ],
    "a_helper_calling_a_helper": [
        "import os\nimport {m}",
        "def inner():\n    os.environ[{v!r}] = 'b'\ndef setup():\n    inner()",
        "setup()",
    ],
    "written_in_the_cell": ["import os\nimport {m}", "os.environ[{v!r}] = 'b'"],
    "a_module_setter": ["import os\nimport {m}", "{m}.setmode('b')"],
}


@pytest.mark.parametrize("case", list(ENVIRONMENT), ids=list(ENVIRONMENT))
def test_an_environment_variable_set_above_is_set_again(cash_magics, clean_backend, stores_everything, workdir, case):
    _, name = workdir
    cells = [cell.format(m=name, v=VAR) for cell in [*ENVIRONMENT[case], "y = 1", "x = {m}.mode()"]]
    _run_above_then_restart(cash_magics, workdir, cells)

    run_cash_cell(cash_magics, cells[-1], cells=cells)

    assert os.environ.get(VAR) == "b"
    assert cash_magics.shell.user_ns["x"] == "b"


DIRECTORY = {
    "written_in_the_cell": ["import os\nimport {m}", "os.chdir({sub!r})"],
    "a_module_helper": ["import os\nimport {m}", "{m}.go({sub!r})"],
    "a_notebook_helper": ["import os\nimport {m}", "def enter(d):\n    os.chdir(d)\nenter({sub!r})"],
}


@pytest.mark.parametrize("case", list(DIRECTORY), ids=list(DIRECTORY))
def test_a_directory_changed_above_is_changed_again(cash_magics, clean_backend, stores_everything, workdir, case):
    path, name = workdir
    sub = str(path / "sub")
    cells = [cell.format(m=name, sub=sub) for cell in [*DIRECTORY[case], "y = 1", "x = open('d.txt').read()"]]
    _run_above_then_restart(cash_magics, workdir, cells)

    run_cash_cell(cash_magics, cells[-1], cells=cells)

    assert os.getcwd() == sub
    assert cash_magics.shell.user_ns["x"] == "SUB"


def test_a_setting_that_ran_in_this_kernel_is_not_run_again(cash_magics, clean_backend, stores_everything, workdir):
    """Without a restart the process keeps what the cell set: running the
    reader does not run the setting again over a change made since."""
    _, name = workdir
    cells = ["import os", f"os.environ[{VAR!r}] = 'b'", f"x = os.environ.get({VAR!r})"]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    os.environ[VAR] = "changed by hand"

    run_cash_cell(cash_magics, "y = 2", cells=[*cells, "y = 2"])

    assert os.environ[VAR] == "changed by hand"


def test_a_setting_never_run_is_not_run_by_the_rebuild(cash_magics, clean_backend, stores_everything, workdir):
    """Control: a setting edited and not run yet reaches no reader, as
    without a restart; only one that ran is put back."""
    _, name = workdir
    cells = ["import os", f"os.environ[{VAR!r}] = 'b'", "y = 1", f"x = os.environ.get({VAR!r}, 'unset')"]
    _run_above_then_restart(cash_magics, workdir, cells)
    edited = [cells[0], f"os.environ[{VAR!r}] = 'c'", *cells[2:]]

    run_cash_cell(cash_magics, "z = 1", cells=[*edited, "z = 1"])

    assert os.environ.get(VAR) != "c"
