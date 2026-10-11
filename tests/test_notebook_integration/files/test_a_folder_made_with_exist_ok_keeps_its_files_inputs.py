"""A cell function that makes sure its data folder exists still sees edits there.

``os.makedirs(d, exist_ok=True)`` on a folder that is already there creates
nothing, but its audit event fires before the syscall: the folder was taken for
the cell's own output and the file read under it was not a dependency, so a
later Run All kept the old text.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.files, pytest.mark.timeout(120)]


def test_an_edit_under_a_folder_made_with_exist_ok_is_seen(nb_runner, tmp_path):
    data = tmp_path / "data" / "x.txt"
    data.parent.mkdir()
    data.write_text("one", encoding="utf-8")
    path = str(data).replace("\\", "/")
    nb_runner.create_notebook(
        [
            "import os, time\n"
            "def load(p):\n"
            "    os.makedirs(os.path.dirname(p), exist_ok=True)\n"
            "    time.sleep(0.2)\n"
            "    with open(p) as fh:\n"
            "        return fh.read()",
            f"text = load('{path}')",
            "print(text)",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert nb_runner.get_output(3).strip() == "one"

    data.write_text("two", encoding="utf-8")
    nb_runner.run_all()
    assert nb_runner.get_output(3).strip() == "two"
    assert nb_runner.peek("text") == repr("two")
