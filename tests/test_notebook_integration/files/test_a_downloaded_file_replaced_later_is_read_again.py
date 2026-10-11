"""A cell function that downloads its data when it is missing sees a replaced file.

The file the function created was taken for the cell's own output and never
recorded as an input, so after the user replaced it the next Run All still
showed the first download's data.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.files, pytest.mark.timeout(120)]


def test_a_replaced_download_is_read_on_the_next_run_all(nb_runner, tmp_path):
    remote = tmp_path / "remote.txt"
    remote.write_text("one", encoding="utf-8")
    local = tmp_path / "data.txt"
    nb_runner.create_notebook(
        [
            "import os, shutil, time\n"
            "def load(p, src):\n"
            "    if not os.path.exists(p):\n"
            "        shutil.copyfile(src, p)\n"
            "    time.sleep(0.2)\n"
            "    with open(p) as fh:\n"
            "        return fh.read()",
            f"text = load({str(local)!r}, {str(remote)!r})",
            "print(text)",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert nb_runner.get_output(3).strip() == "one"

    local.write_text("two", encoding="utf-8")
    nb_runner.run_all()
    assert nb_runner.get_output(3).strip() == "two"
    assert nb_runner.peek("text") == repr("two")
