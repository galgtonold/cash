"""An edit of a module ``%autoreload`` reloads reaches the cells that read it.

``%load_ext autoreload`` / ``%autoreload 2`` / ``import mylib`` in one cell,
then ``x = mylib.get(10)``: after editing ``get`` to return ``x + 1``,
autoreload picked up the edit but cash served the result of the old code.
The same for a module only ``%aimport`` loads.
"""

import os
import uuid

import pytest

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
