"""An edit of a module ``%autoreload`` reloads reaches the cells that read it.

``%load_ext autoreload`` / ``%autoreload 2`` / ``import mylib`` in one cell,
then ``x = mylib.get(10)``: after editing ``get`` to return ``x + 1``,
autoreload picked up the edit but cash served the result of the old code.
The same for a module only ``%aimport`` loads.
"""

import os
import uuid

import pytest
from nbclient.exceptions import CellExecutionError

pytestmark = [pytest.mark.integration, pytest.mark.timeout(240)]

V1 = "import time\ndef get(x):\n    time.sleep(0.3)\n    return x\n"
V2 = "import time\ndef get(x):\n    time.sleep(0.3)\n    return x + 1\n"


def _edit(path):
    before = os.stat(path).st_mtime
    path.write_text(V2, encoding="utf-8")
    os.utime(path, (before + 2, before + 2))


@pytest.mark.parametrize(
    "cells",
    [
        ["%load_ext autoreload\n%autoreload 2\nimport mylib", "x = mylib.get(10)"],
        ["%load_ext autoreload\n%autoreload 2\nfrom mylib import get", "x = get(10)"],
        ["%load_ext autoreload\n%autoreload 1\n%aimport mylib", "x = mylib.get(10)"],
    ],
    ids=["import_beside_the_magics", "from_import_beside_the_magics", "aimport"],
)
def test_the_reader_runs_the_edited_code(nb_runner, cells):
    # A name of its own: a warm kernel keeps the extension, and what it
    # recorded of a `mylib` another test loaded, between tests.
    name = f"arlib_{uuid.uuid4().hex[:8]}"
    path = nb_runner.work_dir / f"{name}.py"
    path.write_text(V1, encoding="utf-8")
    nb_runner.create_notebook([cell.replace("mylib", name) for cell in cells])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert nb_runner.peek("x") == "10"

    _edit(path)
    nb_runner.run_cell(2)
    assert nb_runner.peek("x") == "11", nb_runner.get_raw_output(2)

    nb_runner.run_all()
    assert nb_runner.peek("x") == "11", nb_runner.get_raw_output(2)


TICKED = (
    "import os\nimport time\n\nimport cash\n\n\n{extra}"
    "@cash.cache\n"
    "def get(x):\n"
    "    fd = os.open({log!r}, os.O_WRONLY | os.O_APPEND | os.O_CREAT)  # @cash:assume-safe\n"
    "    os.write(fd, b'x')\n"
    "    os.close(fd)\n"
    "    time.sleep(0.3)\n"
    "    return {body}\n"
)


def test_what_ran_after_the_edit_is_reused_by_a_fresh_process(nb_runner, tmp_path):
    """The extension's own reload compiled the edited functions on their own:
    ``get``'s source read as the module's first lines (the cash wrapper's
    file, for a decorated one) and ``add`` had a copy of the module's
    namespace for its globals. The kernel keyed ``get(10)`` otherwise than a
    fresh process does, so a script, and the kernel after a restart, ran it
    again."""
    from tests._scripts import run_python

    log = tmp_path / "runs.log"
    name = f"arlib_{uuid.uuid4().hex[:8]}"
    path = nb_runner.work_dir / f"{name}.py"
    path.write_text(TICKED.format(extra="", log=str(log), body="x"), encoding="utf-8")
    nb_runner.create_notebook([f"%load_ext autoreload\n%autoreload 2\nimport {name}", f"x = {name}.get(10)"])
    nb_runner.start_kernel()
    nb_runner.run_all()

    before = os.stat(path).st_mtime
    edited = TICKED.format(extra="def add(x):\n    return x + 2\n\n\n", log=str(log), body="add(x)")
    path.write_text(edited, encoding="utf-8")
    os.utime(path, (before + 2, before + 2))
    nb_runner.run_cell(2)
    assert nb_runner.peek("x") == "12", nb_runner.get_raw_output(2)
    nb_runner.settle_writes()
    assert len(log.read_bytes()) == 2
    cache_dir = nb_runner.peek("__import__('cash._active', fromlist=['x']).default_cash().backend.local_dir")

    done = run_python(
        "-c", f"import {name}; print({name}.get(10))", cwd=nb_runner.work_dir, cache_dir=cache_dir.strip("'")
    )

    assert done.stdout.split() == ["12"]
    assert len(log.read_bytes()) == 2, "the script did not reuse what the kernel computed after the edit"


def test_a_call_that_raised_after_an_edit_leaves_its_argument_cacheable(nb_runner, tmp_path):
    """An edit with a typo made ``hist = elo.run_elo(played)`` raise; with
    the helper's source read as none, ``played`` was taken for changed by
    the failed call and lost its lineage, so once the typo was fixed every
    statement reading it ran uncached ("Input variable missing lineage")
    and ran again after the restart."""
    log = tmp_path / "runs.log"
    name = f"arelo_{uuid.uuid4().hex[:8]}"
    path = nb_runner.work_dir / f"{name}.py"
    body = (
        "import os, time\n\n\n"
        "def run_elo(matches, k=20.0):\n"
        f"    fd = os.open({str(log)!r}, os.O_WRONLY | os.O_APPEND | os.O_CREAT)  # @cash:assume-safe\n"
        "    os.write(fd, b'x')\n"
        "    os.close(fd)\n"
        "    time.sleep(0.3)\n"
        "    return float(sum(matches)) * k{tail}\n"
    )
    path.write_text(body.format(tail=""), encoding="utf-8")
    nb_runner.create_notebook(
        [
            f"import cash\n%cash_on\n%load_ext autoreload\n%autoreload 2\nimport {name} as elo",
            "played = list(range(5))",
            "hist = elo.run_elo(played)",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()

    before = os.stat(path).st_mtime
    for n, tail in enumerate([" + goal_diff", " + 1.0"], start=1):
        path.write_text(body.format(tail=tail), encoding="utf-8")
        os.utime(path, (before + 2 * n, before + 2 * n))
        if n == 1:
            with pytest.raises(CellExecutionError):
                nb_runner.run_cell(3)
        else:
            nb_runner.run_cell(3)
    assert nb_runner.peek("hist") == "201.0"
    nb_runner.settle_writes()
    runs = len(log.read_bytes())

    nb_runner.restart()
    nb_runner.run_all()

    assert nb_runner.peek("hist") == "201.0"
    assert len(log.read_bytes()) == runs, "the fixed call was not cached"
